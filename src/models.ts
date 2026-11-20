import type {
	Api,
	Model,
	ModelThinkingLevel,
	RefreshModelsContext,
	SimpleStreamOptions,
	StreamOptions,
	ThinkingLevelMap,
} from "@earendil-works/pi-ai";
import type { AcpClient, SessionInstance } from "./acp-client.ts";
import { errorMessage, type AcpSelect, type AcpSessionConfig } from "./acp.ts";
import type { AcpAgentConfig, AcpProviderConfig } from "./config.ts";

export const ACP_PROVIDER_ID = "acp";
const ACP_API = "acp";

/** Until the agent reports its window in `usage_update.size`. */
const DEFAULT_CONTEXT_WINDOW = 128_000;
const PI_THINKING_LEVELS: readonly ModelThinkingLevel[] = ["off", "minimal", "low", "medium", "high", "xhigh", "max"];
/** Agent values that also mean "off". */
const OFF_ALIASES = ["off", "none", "disabled"];

/**
 * Maps Pi thinking levels onto the agent's `thought_level` values by name
 * (case-insensitive); levels the agent lacks are marked unsupported.
 */
function thinkingLevelMap(thoughtLevel: AcpSelect): ThinkingLevelMap {
	const find = (names: string[]) =>
		thoughtLevel.choices.find((choice) => names.includes(choice.value.toLowerCase()))?.value ?? null;
	return Object.fromEntries(
		PI_THINKING_LEVELS.map((level) => [level, find(level === "off" ? OFF_ALIASES : [level])]),
	) as ThinkingLevelMap;
}

function toModel(agentId: string, id: string, name: string, thinking: ThinkingLevelMap | undefined): Model<typeof ACP_API> {
	return {
		id,
		name,
		api: ACP_API,
		provider: ACP_PROVIDER_ID,
		baseUrl: `stdio://${agentId}`, // Unused: the transport is a child process.
		reasoning: thinking !== undefined,
		...(thinking ? { thinkingLevelMap: thinking } : {}),
		input: ["text", "image"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: DEFAULT_CONTEXT_WINDOW,
		maxTokens: 16_384,
	};
}

/** The agent's default model, then one model per model its session offers. */
function agentModels(agent: AcpAgentConfig, config?: AcpSessionConfig): Model<typeof ACP_API>[] {
	const thinking = config?.thoughtLevel && thinkingLevelMap(config.thoughtLevel);
	return [
		toModel(agent.id, agent.id, agent.name, thinking),
		...(config?.model?.choices ?? []).map((choice) =>
			toModel(agent.id, `${agent.id}/${choice.value}`, `${choice.name} (${agent.name})`, thinking),
		),
	];
}

/** Splits `<agent id>/<model id>`; a bare agent id selects the agent's default model. */
export function parseModelId(id: string, agents: ReadonlyMap<string, AcpAgentConfig>) {
	const agent = agents.get(id);
	if (agent) return { agent, modelId: undefined };
	const slash = id.indexOf("/");
	if (slash < 0) return undefined;
	const owner = agents.get(id.slice(0, slash));
	return owner ? { agent: owner, modelId: id.slice(slash + 1) } : undefined;
}

export function createModelCatalog(
	options: { config: AcpProviderConfig },
	agents: ReadonlyMap<string, AcpAgentConfig>,
	getClient: (agent: AcpAgentConfig) => AcpClient,
	log: ((message: string) => void) | undefined,
) {
	/** Each agent's models from discovery (or the persisted catalog); default model only until then. */
	const discovered = new Map<string, Model<Api>[]>();
	/** The context window each model's agent last reported in `usage_update.size`, by Pi model id. */
	const reportedWindows = new Map<string, number>();
	// WARN: Pi reads `contextWindow` from the selected model and replaces that only on model selection or
	// provider registration, so a reported size takes effect when the catalogue is next published, not mid-session.
	// A models.json `modelOverrides` entry still wins: Pi applies it on top of these models.
	const buildModels = () =>
		options.config.agents.flatMap((agent) =>
			(discovered.get(agent.id) ?? agentModels(agent)).map((model) => {
				const reported = reportedWindows.get(model.id);
				return reported === undefined || reported === model.contextWindow ? model : { ...model, contextWindow: reported };
			}),
		);
	let models = buildModels();

	/** Switches the session to the chosen model and Pi's thinking level, when the agent supports them. */
	async function applySettings(
		acp: AcpClient,
		agent: AcpAgentConfig,
		sessionId: string,
		modelId: string | undefined,
		model: Model<Api>,
		streamOptions: StreamOptions | undefined,
		signal = streamOptions?.signal,
		instance?: SessionInstance,
	): Promise<void> {
		const select = acp.sessionConfig(sessionId)?.model;
		if (modelId !== undefined) {
			if (!select) throw new Error(`${agent.name} does not support choosing a model over ACP.`);
			if (!select.choices.some((choice) => choice.value === modelId)) {
				const offered = select.choices.map((choice) => choice.value).join(", ");
				throw new Error(`${agent.name} does not offer model '${modelId}'. Available: ${offered || "none"}.`);
			}
		}
		// The bare `acp/<agent id>` model means the model the session started with.
		const wantedModel = modelId ?? acp.initialModel(sessionId);
		if (select && wantedModel !== undefined) await acp.setSessionSetting(sessionId, "model", wantedModel, signal, instance);

		// Re-read: switching models can change the offered reasoning levels.
		const thoughtLevel = acp.sessionConfig(sessionId)?.thoughtLevel;
		if (!thoughtLevel) return;
		const level: ModelThinkingLevel = (streamOptions as SimpleStreamOptions | undefined)?.reasoning ?? "off";
		const value = model.thinkingLevelMap?.[level] ?? thinkingLevelMap(thoughtLevel)[level];
		if (typeof value !== "string" || !thoughtLevel.choices.some((choice) => choice.value === value)) {
			log?.(`${agent.id}: no thought_level value for thinking level '${level}'; keeping the agent's setting`);
			return;
		}
		await acp.setSessionSetting(sessionId, "thoughtLevel", value, signal, instance);
	}

	/** Lists one agent's models from a (possibly new) session, without interactive sign-in. */
	async function discoverAgent(agent: AcpAgentConfig, signal: AbortSignal): Promise<void> {
		const acp = getClient(agent);
		const sessionId = await acp.ensureSession({ signal, authenticate: false });
		discovered.set(agent.id, agentModels(agent, acp.sessionConfig(sessionId)));
	}

	/** Restores the persisted catalog, then rediscovers every agent; failed agents keep their old list. */
	const refreshModels = async (context: RefreshModelsContext): Promise<void> => {
		const restored = new Map<string, Model<Api>[]>();
		for (const model of context.stored?.models ?? []) {
			const owner = parseModelId(model.id, agents)?.agent.id;
			if (!owner) continue;
			const stored: unknown = "contextWindow" in model ? model.contextWindow : undefined;
			const window = typeof stored === "number" && Number.isSafeInteger(stored) && stored > 0 ? stored : DEFAULT_CONTEXT_WINDOW;
			// WHY: the persisted catalogue holds only windows this provider published, so a non-default one was
			// reported by the agent earlier and must survive the rediscovery below, which starts from the default.
			if (window !== DEFAULT_CONTEXT_WINDOW && !reportedWindows.has(model.id)) reportedWindows.set(model.id, window);
			if (discovered.has(owner)) continue;
			const list = restored.get(owner) ?? [];
			list.push({ ...(model as Model<Api>), contextWindow: window });
			restored.set(owner, list);
		}
		for (const [owner, list] of restored) discovered.set(owner, list);
		if (context.allowNetwork) {
			await Promise.all(
				options.config.agents.map((agent) =>
					discoverAgent(agent, context.signal).catch((error: unknown) =>
						log?.(`${agent.id}: model discovery failed: ${errorMessage(error)}`),
					),
				),
			);
		}
		const next = buildModels();
		await context.publish({ persist: { models: next, checkedAt: Date.now() }, update: () => (models = next) });
	};

	return { reportedWindows, getModels: () => models, refreshModels, applySettings };
}
