/**
 * The `acp` provider: a Pi `Provider` whose models delegate the whole agent
 * loop to configured ACP agents.
 *
 * Each agent is the model `acp/<agent id>`, which runs with the agent's own
 * default model. Models the agent offers are discovered by opening a session
 * (`refreshModels`) and listed as `acp/<agent id>/<model id>`; picking one
 * switches the session to it before the prompt is sent. Pi's thinking level
 * maps onto the agent's `thought_level` setting when it has one.
 *
 * ACP agents execute their own tools and keep conversation history
 * server-side, so each Pi model turn sends only the newest user message and
 * streams back thought chunks, tool notices, and the reply text. ACP reports
 * no per-turn token usage, so usage is zero.
 */

import {
	createAssistantMessageEventStream,
	type Api,
	type AssistantMessage,
	type AssistantMessageEventStream,
	type Model,
	type ModelThinkingLevel,
	type Provider,
	type RefreshModelsContext,
	type SimpleStreamOptions,
	type StreamOptions,
	type ThinkingLevelMap,
	type TranscriptContext,
	type Usage,
} from "@earendil-works/pi-ai";
import { AcpClient, AcpRpcError, RPC_INVALID_PARAMS, RPC_METHOD_NOT_FOUND, type ServerRequest } from "./acp-client.ts";
import {
	ACP_AUTH_REQUIRED,
	autoApprove,
	type AcpAgentInfo,
	type AcpSelect,
	type AcpSessionConfig,
	parsePermissionRequest,
	type AcpPermissionOutcome,
	type AcpPermissionRequest,
	type AcpSessionUpdate,
	type AcpStopReason,
} from "./acp.ts";
import type { AcpAgentConfig, AcpProviderConfig } from "./config.ts";

export const ACP_PROVIDER_ID = "acp";
const ACP_API = "acp";

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

function baseModel(id: string, name: string, agent: AcpAgentConfig, thinking: ThinkingLevelMap | undefined): Model<typeof ACP_API> {
	return {
		id,
		name,
		api: ACP_API,
		provider: ACP_PROVIDER_ID,
		baseUrl: `stdio://${agent.id}`, // Unused: the transport is a child process.
		reasoning: thinking !== undefined,
		...(thinking ? { thinkingLevelMap: thinking } : {}),
		input: ["text"],
		cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0 },
		contextWindow: 128_000,
		maxTokens: 16_384,
	};
}

/** The agent's default model, then one model per model its session offers. */
export function agentModels(agent: AcpAgentConfig, config?: AcpSessionConfig): Model<typeof ACP_API>[] {
	const thinking = config?.thoughtLevel && thinkingLevelMap(config.thoughtLevel);
	return [
		baseModel(agent.id, agent.name, agent, thinking),
		...(config?.model?.choices ?? []).map((choice) =>
			baseModel(`${agent.id}/${choice.value}`, `${choice.name} (${agent.name})`, agent, thinking),
		),
	];
}

/** Splits `<agent id>/<model id>`; a bare agent id selects the agent's default model. */
function parseModelId(id: string, agents: ReadonlyMap<string, AcpAgentConfig>) {
	const agent = agents.get(id);
	if (agent) return { agent, modelId: undefined };
	const slash = id.indexOf("/");
	if (slash < 0) return undefined;
	const owner = agents.get(id.slice(0, slash));
	return owner ? { agent: owner, modelId: id.slice(slash + 1) } : undefined;
}

const zeroUsage = (): Usage => ({
	input: 0,
	output: 0,
	cacheRead: 0,
	cacheWrite: 0,
	totalTokens: 0,
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
});

/** Decides a `session/request_permission`; `signal` aborts when the turn is cancelled. */
export type PermissionPolicy = (
	request: AcpPermissionRequest,
	context: { agent: AcpAgentConfig; signal: AbortSignal | undefined },
) => Promise<AcpPermissionOutcome>;

export interface AcpProviderOptions {
	config: AcpProviderConfig;
	/** Defaults to auto-approval. */
	permissionPolicy?: PermissionPolicy;
	log?: ((message: string) => void) | undefined;
}

export interface AcpProviderHandle {
	provider: Provider;
	/** Starts fresh ACP sessions in `cwd` on the next prompt (a new Pi session). */
	resetSession(cwd: string): void;
	/** Kills all agent processes; the provider respawns them if used again. */
	dispose(): void;
}

const errorMessage = (error: unknown): string => (error instanceof Error ? error.message : String(error));

/** Text of the newest user message; images are noted since prompts here are text-only. */
function latestUserPrompt(messages: TranscriptContext["messages"]): string | null {
	const message = messages.findLast((candidate) => candidate.role === "user");
	if (!message) return null;
	if (typeof message.content === "string") return message.content.trim() ? message.content : null;
	const texts = message.content.flatMap((block) => (block.type === "text" ? [block.text] : []));
	const images = message.content.length - texts.length;
	if (images > 0) texts.push(`[${images} image attachment(s) could not be forwarded to the ACP agent]`);
	const text = texts.join("\n");
	return text.trim() ? text : null;
}

/** First line of tool output, capped so huge outputs don't flood the transcript. */
function summarize(text: string, max = 200): string {
	const line = text.trim().split("\n", 1)[0] ?? "";
	return line.length > max ? `${line.slice(0, max)}…` : line;
}

/**
 * Builds the streamed `AssistantMessage`: keeps one text or thinking block
 * open at a time and emits balanced start/delta/end events for it.
 */
function createMessageWriter(stream: AssistantMessageEventStream, message: AssistantMessage) {
	let open: { type: "text" | "thinking"; index: number } | null = null;

	const close = (): void => {
		if (!open) return;
		const contentIndex = open.index;
		const block = message.content[contentIndex];
		if (block?.type === "text") stream.push({ type: "text_end", contentIndex, content: block.text, partial: message });
		if (block?.type === "thinking") {
			stream.push({ type: "thinking_end", contentIndex, content: block.thinking, partial: message });
		}
		open = null;
	};

	const append = (type: "text" | "thinking", delta: string): void => {
		if (!delta) return;
		if (open?.type !== type) {
			close();
			const contentIndex = message.content.length;
			if (type === "text") {
				message.content.push({ type: "text", text: "" });
				stream.push({ type: "text_start", contentIndex, partial: message });
			} else {
				message.content.push({ type: "thinking", thinking: "" });
				stream.push({ type: "thinking_start", contentIndex, partial: message });
			}
			open = { type, index: contentIndex };
		}
		const contentIndex = open.index;
		const block = message.content[contentIndex];
		if (block?.type === "text") {
			block.text += delta;
			stream.push({ type: "text_delta", contentIndex, delta, partial: message });
		} else if (block?.type === "thinking") {
			block.thinking += delta;
			stream.push({ type: "thinking_delta", contentIndex, delta, partial: message });
		}
	};

	let settled = false;
	const done = (reason: "stop" | "length"): void => {
		if (settled) return;
		settled = true;
		close();
		message.stopReason = reason;
		stream.push({ type: "done", reason, message });
		stream.end(message);
	};
	const fail = (reason: "error" | "aborted", error: string): void => {
		if (settled) return;
		settled = true;
		close();
		message.stopReason = reason;
		message.errorMessage = error;
		stream.push({ type: "error", reason, error: message });
		stream.end(message);
	};

	return { append, done, fail };
}

/** Renders tool activity as short inline notices in the reply text. */
function createToolRenderer(append: (text: string) => void) {
	const titles = new Map<string, string>();
	return (update: Extract<AcpSessionUpdate, { type: "tool_call" | "tool_call_update" }>): void => {
		if (update.title) titles.set(update.toolCallId, update.title);
		const title = titles.get(update.toolCallId) ?? "tool";
		if (update.type === "tool_call") {
			append(`\n⚙️ Executing [${title}]...\n`);
			return;
		}
		if (update.status !== "completed" && update.status !== "failed") return;
		const mark = update.status === "completed" ? "✓" : "✗";
		const summary = summarize(update.summary);
		append(`${mark} ${title}${summary ? `: ${summary}` : ""}\n`);
	};
}

/** Explains `auth_required`, listing the methods the agent advertised. */
function authHint(agent: AcpAgentConfig, info: AcpAgentInfo | null): string {
	const methods = info?.authMethods ?? [];
	if (methods.length === 0) return `${agent.name} requires authentication. Sign in with the agent's own CLI, then retry.`;
	const list = methods
		.map((method) => `'${method.id}' (${method.name}${method.description ? `: ${method.description}` : ""})`)
		.join(", ");
	return agent.authMethod
		? `${agent.name} still requires authentication after trying method '${agent.authMethod}'. Available methods: ${list}.`
		: `${agent.name} requires authentication. Sign in with the agent's own CLI, or set "authMethod" for this agent in ~/.pi/agent/settings.json to one of: ${list}.`;
}

export function createAcpProvider(options: AcpProviderOptions): AcpProviderHandle {
	const policy: PermissionPolicy = options.permissionPolicy ?? (async (request) => autoApprove(request));
	const agents = new Map(options.config.agents.map((agent) => [agent.id, agent]));
	/** Each agent's models from discovery (or the persisted catalog); default model only until then. */
	const discovered = new Map<string, Model<Api>[]>();
	const buildModels = () =>
		options.config.agents.flatMap((agent) => discovered.get(agent.id) ?? agentModels(agent));
	let models = buildModels();
	/** Each agent's model when its current session started: what `acp/<agent id>` means. */
	const sessionDefaults = new Map<string, string | undefined>();
	let cwd = process.cwd();
	/** One agent process per model, spawned on first use. */
	const clients = new Map<string, AcpClient>();

	const answerServerRequest = async (
		agent: AcpAgentConfig,
		{ method, params, signal }: ServerRequest,
	): Promise<unknown> => {
		if (method !== "session/request_permission") {
			throw new AcpRpcError(`pi-acp-provider does not support '${method}'`, RPC_METHOD_NOT_FOUND);
		}
		const request = parsePermissionRequest(params);
		if (!request) throw new AcpRpcError("Malformed session/request_permission params", RPC_INVALID_PARAMS);
		// A cancelled turn must answer every pending permission request with `cancelled`.
		if (signal?.aborted) return { outcome: { outcome: "cancelled" } };
		const outcome = await policy(request, { agent, signal });
		return { outcome: signal?.aborted ? { outcome: "cancelled" } : outcome };
	};

	const getClient = (agent: AcpAgentConfig): AcpClient => {
		let client = clients.get(agent.id);
		if (!client) {
			client = new AcpClient({
				agent,
				cwd,
				handshakeTimeoutMs: options.config.handshakeTimeoutMs,
				onServerRequest: (request) => answerServerRequest(agent, request),
				log: options.log && ((message) => options.log?.(`${agent.id}: ${message}`)),
			});
			clients.set(agent.id, client);
		}
		return client;
	};

	async function runTurn(
		stream: AssistantMessageEventStream,
		model: Model<Api>,
		context: TranscriptContext,
		streamOptions: StreamOptions | undefined,
	): Promise<void> {
		const signal = streamOptions?.signal;
		const message: AssistantMessage = {
			role: "assistant",
			content: [],
			api: model.api,
			provider: model.provider,
			model: model.id,
			usage: zeroUsage(),
			stopReason: "pending",
			timestamp: Date.now(),
		};
		const writer = createMessageWriter(stream, message);
		const target = parseModelId(model.id, agents);
		if (!target) {
			writer.fail("error", `Unknown ACP agent '${model.id}'. Configure it under "acp.agents" in ~/.pi/agent/settings.json.`);
			return;
		}
		const { agent } = target;
		const acp = getClient(agent);
		const abortedOrError = (error: unknown): void => {
			if (signal?.aborted) writer.fail("aborted", "Request was aborted");
			else if (error instanceof AcpRpcError && error.code === ACP_AUTH_REQUIRED) {
				writer.fail("error", authHint(agent, acp.info));
			} else writer.fail("error", errorMessage(error));
		};

		const promptText = latestUserPrompt(context.messages);
		if (promptText === null) {
			writer.fail("error", `No user message found to send to ${agent.name}.`);
			return;
		}

		// Setup failures (spawn, handshake, session) end the stream before `start`.
		let sessionId: string;
		try {
			sessionId = await acp.ensureSession({ signal });
			await applySettings(acp, agent, sessionId, target.modelId, model, streamOptions);
		} catch (error) {
			abortedOrError(error);
			return;
		}

		stream.push({ type: "start", partial: message });
		const renderTool = createToolRenderer((text) => writer.append("text", text));
		const onProviderStreamEvent = streamOptions?.onProviderStreamEvent;
		try {
			const defaultPayload = { sessionId, prompt: [{ type: "text", text: promptText }] };
			const payload = (await streamOptions?.onPayload?.(defaultPayload, model)) ?? defaultPayload;
			const result = await acp.prompt({
				sessionId,
				payload,
				signal,
				onRawUpdate:
					onProviderStreamEvent &&
					((params) => {
						Promise.resolve(onProviderStreamEvent(params, model)).catch((error: unknown) =>
							options.log?.(`onProviderStreamEvent failed: ${errorMessage(error)}`),
						);
					}),
				onUpdate: (update) => {
					switch (update.type) {
						case "agent_message_chunk":
							writer.append("text", update.text);
							return;
						case "agent_thought_chunk":
							writer.append("thinking", update.text);
							return;
						case "tool_call":
						case "tool_call_update":
							renderTool(update);
							return;
						default: {
							const _exhaustive: never = update;
							void _exhaustive;
						}
					}
				},
			});
			if (result.kind === "aborted") writer.fail("aborted", "Request was aborted");
			else finishTurn(writer, agent, result.stopReason);
		} catch (error) {
			abortedOrError(error);
		}
	}

	/** Switches the session to the chosen model and Pi's thinking level, when the agent supports them. */
	async function applySettings(
		acp: AcpClient,
		agent: AcpAgentConfig,
		sessionId: string,
		modelId: string | undefined,
		model: Model<Api>,
		streamOptions: StreamOptions | undefined,
	): Promise<void> {
		const signal = streamOptions?.signal;
		const config = acp.sessionConfig(sessionId);
		if (!sessionDefaults.has(sessionId)) sessionDefaults.set(sessionId, config?.model?.currentValue);
		const wantedModel = modelId ?? sessionDefaults.get(sessionId);
		if (wantedModel !== undefined && config?.model) {
			if (modelId !== undefined && !config.model.choices.some((choice) => choice.value === modelId)) {
				const offered = config.model.choices.map((choice) => choice.value).join(", ");
				throw new Error(`${agent.name} does not offer model '${modelId}'. Available: ${offered || "none"}.`);
			}
			await acp.setSessionSetting(sessionId, "model", wantedModel, signal);
		} else if (modelId !== undefined) {
			throw new Error(`${agent.name} does not support choosing a model over ACP.`);
		}

		// Re-read: switching models can change the offered reasoning levels.
		const thoughtLevel = acp.sessionConfig(sessionId)?.thoughtLevel;
		if (!thoughtLevel) return;
		const level: ModelThinkingLevel = (streamOptions as SimpleStreamOptions | undefined)?.reasoning ?? "off";
		const value = model.thinkingLevelMap?.[level] ?? thinkingLevelMap(thoughtLevel)[level];
		if (typeof value !== "string" || !thoughtLevel.choices.some((choice) => choice.value === value)) {
			options.log?.(`${agent.id}: no thought_level value for thinking level '${level}'; keeping the agent's setting`);
			return;
		}
		await acp.setSessionSetting(sessionId, "thoughtLevel", value, signal);
	}

	/** Lists one agent's models from a (possibly new) session, without interactive sign-in. */
	async function discoverAgent(agent: AcpAgentConfig, signal: AbortSignal): Promise<void> {
		const acp = getClient(agent);
		const sessionId = await acp.ensureSession({ signal, authenticate: false });
		const config = acp.sessionConfig(sessionId);
		if (!sessionDefaults.has(sessionId)) sessionDefaults.set(sessionId, config?.model?.currentValue);
		discovered.set(agent.id, agentModels(agent, config));
	}

	/** Restores the persisted catalog, then rediscovers every agent; failed agents keep their old list. */
	const refreshModels = async (context: RefreshModelsContext): Promise<void> => {
		const restored = new Map<string, Model<Api>[]>();
		for (const model of context.stored?.models ?? []) {
			const owner = parseModelId(model.id, agents)?.agent.id;
			if (owner && !discovered.has(owner)) restored.set(owner, [...(restored.get(owner) ?? []), model as Model<Api>]);
		}
		for (const [owner, list] of restored) discovered.set(owner, list);
		if (context.allowNetwork) {
			await Promise.all(
				options.config.agents.map((agent) =>
					discoverAgent(agent, context.signal).catch((error: unknown) =>
						options.log?.(`${agent.id}: model discovery failed: ${errorMessage(error)}`),
					),
				),
			);
		}
		const next = buildModels();
		await context.publish({ persist: { models: next, checkedAt: Date.now() }, update: () => (models = next) });
	};

	const stream = (
		model: Model<Api>,
		context: TranscriptContext,
		streamOptions?: StreamOptions,
	): AssistantMessageEventStream => {
		const events = createAssistantMessageEventStream();
		runTurn(events, model, context, streamOptions).catch((error: unknown) =>
			options.log?.(`stream crashed: ${errorMessage(error)}`),
		);
		return events;
	};

	const provider: Provider = {
		id: ACP_PROVIDER_ID,
		name: "ACP agents",
		auth: {
			apiKey: {
				// Ambient: each agent manages its own credentials, so there is no
				// key for Pi to store and the provider always reads as configured.
				name: "Agent-managed authentication",
				resolve: async () => ({ auth: {}, source: "ACP agent" }),
			},
		},
		getModels: () => models,
		refreshModels,
		stream,
		streamSimple: stream,
	};

	return {
		provider,
		resetSession(nextCwd: string): void {
			cwd = nextCwd;
			for (const client of clients.values()) client.resetSession(nextCwd);
		},
		dispose(): void {
			for (const client of clients.values()) client.dispose();
			clients.clear();
		},
	};
}

function finishTurn(
	writer: ReturnType<typeof createMessageWriter>,
	agent: AcpAgentConfig,
	stopReason: AcpStopReason,
): void {
	switch (stopReason) {
		case "end_turn":
			writer.done("stop");
			return;
		case "max_tokens":
		case "max_turn_requests":
			writer.done("length");
			return;
		case "refusal":
			writer.fail("error", `${agent.name} refused to continue this turn.`);
			return;
		case "cancelled":
			writer.fail("aborted", `${agent.name} cancelled the turn.`);
			return;
		default: {
			const _exhaustive: never = stopReason;
			void _exhaustive;
		}
	}
}
