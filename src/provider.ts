/**
 * The `acp` provider: a Pi `Provider` whose models delegate the agent loop
 * to configured ACP agents.
 *
 * Each agent is the model `acp/<agent id>`, which runs with the agent's own
 * default model. Models the agent offers are discovered by opening a session
 * (`refreshModels`) and listed as `acp/<agent id>/<model id>`; picking one
 * switches the session to it before the prompt is sent. Pi's thinking level
 * maps onto the agent's `thought_level` setting when it has one.
 *
 * Pi's side of the session:
 * - Tools: Pi's current tools are served to the agent over MCP (see
 *   `mcp-server.ts`). A call becomes a real Pi tool call: the provider ends
 *   the Pi message there, Pi runs the tool with its own approvals, hooks and
 *   display, and the result goes back to the agent's waiting MCP request.
 *   The agent's own tool calls are shown as `acp_tool` rows (or inline text).
 *   One ACP turn therefore spans several Pi messages.
 * - System prompt: sent once per ACP session, on `session/new` when the agent
 *   supports it, else in front of the first prompt; a later change is sent
 *   with the next prompt.
 * - History: the agent keeps its own conversation, so each prompt carries only
 *   what it has not seen. A session that missed part of Pi's history (a
 *   resumed or forked Pi session, a model switch, a rewritten branch) gets a
 *   transcript of that part first.
 * - Queue: every queued user message is sent. A steering message that arrives
 *   while the agent works goes into the running turn with `_session/steering`
 *   when the agent supports it, else with the next Pi tool result it is
 *   waiting on, else as the next prompt once the running one ends. A steer
 *   that prompt never delivered goes with the next prompt Pi sends.
 * - Isolated requests (compaction and branch summaries, `cacheRetention:
 *   "none"`) run in a separate, throwaway agent process.
 */

import { randomUUID } from "node:crypto";
import {
	createAssistantMessageEventStream,
	getCurrentSystemPrompt,
	getCurrentTools,
	type Api,
	type AssistantMessage,
	type AssistantMessageEventStream,
	type Message,
	type Model,
	type Provider,
	type StreamOptions,
	type ToolResultMessage,
	type TranscriptContext,
	type UserMessage,
} from "@earendil-works/pi-ai";
import { AcpClient, AcpRpcError, abortable, type ServerRequest, type SessionInstance, type SessionSetup } from "./acp-client.ts";
import {
	RPC_INVALID_PARAMS,
	RPC_METHOD_NOT_FOUND,
	autoApprove,
	autoReject,
	emptyToolContent,
	errorMessage,
	isFinalStatus,
	isRecord,
	parsePermissionRequest,
	structuredToolIdentities,
	type AcpAgentInfo,
	type AcpCommand,
	type AcpPermissionOutcome,
	type AcpPermissionRequest,
} from "./acp.ts";
import { PI_MCP_SERVER, deepMerge, type AcpAgentConfig, type AcpProviderConfig } from "./config.ts";
import { addUsage, createMessageWriter, failTurn, finishTurn, zeroUsage, type MessageWriter } from "./message-writer.ts";
import { ACP_PROVIDER_ID, createModelCatalog, parseModelId } from "./models.ts";
import { PiMcpServer, toMcpTool, type McpCall, type McpCallResult, type McpContent, type McpTool } from "./mcp-server.ts";
import { ACP_TOOL_NAME, type AwaitTool } from "./native-tools.ts";
import {
	PendingSteers,
	buildPromptBlocks,
	diverged,
	keepConfirmed,
	newSyncState,
	partsIn,
	requestUsers,
	tailStart,
	type PromptPart,
	type SyncState,
} from "./session-sync.ts";
import { LiveTurn, PI_TOOL_NAME, createTurnLifecycle, piToolRef } from "./live-turn.ts";
import { conversation, messageKey, renderTranscript, tagged, userBlocks, type AcpContentBlock } from "./transcript.ts";

export { piToolRef } from "./live-turn.ts";
export { ACP_PROVIDER_ID } from "./models.ts";

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
	/**
	 * Whether to show agent tool calls as native Pi tool rows (see `native-tools.ts`). Checked when a
	 * prompt starts; it needs the mirror tool to be active. Otherwise tool activity is inline text.
	 */
	nativeTools?: () => boolean;
	/** How long a cancelled prompt gets to answer before its ACP session is replaced (default 5 s). */
	cancelGraceMs?: number;
}

export interface AcpProviderHandle {
	provider: Provider;
	/** Starts fresh ACP sessions in `cwd` on the next prompt (a new Pi session). */
	resetSession(cwd: string): void;
	/** Id of the agent that serves `modelId` (`<agent id>` or `<agent id>/<model id>`), if any. */
	agentFor(modelId: string): string | undefined;
	/** Slash commands the agent's current session advertised; empty until it has started. */
	commands(agentId: string): Promise<AcpCommand[]>;
	/** Resolves a mirrored tool call once its agent reports a final status (the mirror tool's `execute`). */
	awaitTool: AwaitTool;
	/** Kills all agent processes; the provider respawns them if used again. */
	dispose(): void;
}

/** One key from an agent id and a conversation or session id. */
const turnKey = (agentId: string, id: string): string => `${agentId}\u0000${id}`;

interface SessionRequest {
	agent: AcpAgentConfig;
	acp: AcpClient;
	key: string;
	conversationId: string;
	modelId: string | undefined;
	model: Model<Api>;
	streamOptions: StreamOptions | undefined;
	messages: readonly Message[];
	systemPrompt: string;
	signal: AbortSignal;
}

interface OpenedSession {
	sessionId: string;
	instance: SessionInstance | undefined;
	state: SyncState;
	info: AcpAgentInfo;
}

interface IsolatedRequest {
	stream: AssistantMessageEventStream;
	writer: MessageWriter;
	message: AssistantMessage;
	agent: AcpAgentConfig;
	modelId: string | undefined;
	model: Model<Api>;
	context: TranscriptContext;
	streamOptions: StreamOptions;
}

/** Pi tool result content for the agent's MCP request. */
function toMcpResult(message: ToolResultMessage): McpCallResult {
	const content: McpContent[] = message.content.map((block) =>
		block.type === "text" ? { type: "text", text: block.text } : { type: "image", data: block.data, mimeType: block.mimeType },
	);
	if (content.length === 0) content.push({ type: "text", text: message.isError ? "The tool failed." : "Done." });
	return message.isError ? { content, isError: true } : { content };
}

function sameJson(left: unknown, right: unknown): boolean {
	try {
		return JSON.stringify(left) === JSON.stringify(right);
	} catch {
		return false;
	}
}

export function createAcpProvider(options: AcpProviderOptions): AcpProviderHandle {
	const { log } = options;
	const policy: PermissionPolicy = options.permissionPolicy ?? (async (request) => autoApprove(request));
	const agents = new Map(options.config.agents.map((agent) => [agent.id, agent]));
	const { reportedWindows, getModels, refreshModels, applySettings } = createModelCatalog(options, agents, (agent) => getClient(agent), log);
	let cwd = process.cwd();
	/** One client (and agent process) per agent, spawned on first use. */
	const clients = new Map<string, AcpClient>();
	/** What each ACP session has seen, by `turnKey(agent id, session id)`: session ids collide across agents. */
	const sync = new Map<string, SyncState>();
	/** The tools served to each agent in each conversation, by `LiveTurn.key`, from that conversation's latest Pi context. */
	const served = new Map<string, McpTool[]>();
	/** The `served` key each minted MCP nonce lists and calls tools from, until it is revoked. */
	const nonceServed = new Map<string, string>();
	const revokeNonce = (nonce: string): void => {
		nonceServed.delete(nonce);
		mcp?.revoke(nonce);
	};
	let mcp: PiMcpServer | null = null;
	/** The ACP session each bound MCP nonce belongs to; a call is accepted only for the live turn's session. */
	const nonceSessions = new Map<string, { agentId: string; sessionId: string; instance: SessionInstance | undefined }>();
	/** The MCP nonce minted for each `session/new` params object, until the session is bound or fails. */
	const mintedNonces = new WeakMap<Record<string, unknown>, string>();
	/** The `SessionSetup.key` (Pi tool wiring) each main-client session was opened with, by client and session id. */
	const sessionWiring = new WeakMap<AcpClient, Map<string, string>>();
	const wiringOf = (client: AcpClient): Map<string, string> => {
		let wiring = sessionWiring.get(client);
		if (!wiring) sessionWiring.set(client, (wiring = new Map()));
		return wiring;
	};
	/** The ACP turns in progress, by `LiveTurn.key`; each spans several Pi messages. */
	const lives = new Map<string, LiveTurn>();
	/**
	 * The request that owns each `LiveTurn.key`, from before its setup until its turn ends; the controller
	 * becomes the turn's `controller`. Aborting it supersedes the request, set up or not.
	 */
	const claims = new Map<string, AbortController>();
	const pendingSteers = new PendingSteers();
	// WARN: session ids repeat once an agent respawns, and a turn parked at a Pi tool call outlives its
	// process; routing and sync lookups must match the session instance, never the id alone.
	const isCurrent = (turn: LiveTurn): boolean => turn.instance !== undefined && turn.instance === turn.acp.instance(turn.sessionId);
	const liveOn = (client: AcpClient, sessionId: string | undefined): LiveTurn | undefined =>
		[...lives.values()].find((turn) => turn.acp === client && turn.sessionId === sessionId && isCurrent(turn));
	const syncOf = (turn: LiveTurn): SyncState | undefined => (isCurrent(turn) ? sync.get(turnKey(turn.agent.id, turn.sessionId)) : undefined);
	const { enqueuePrompt, startPrompt, dispatch, notifyWaiters, attach, awaitTool } = createTurnLifecycle({ lives, claims, pendingSteers, syncOf, reportedWindows, log });

	const answerServerRequest = async (
		agent: AcpAgentConfig,
		client: AcpClient,
		isolated: boolean,
		{ method, params, sessionId, signal }: ServerRequest,
	): Promise<unknown> => {
		if (method !== "session/request_permission") {
			throw new AcpRpcError(`pi-acp-provider does not support '${method}'`, RPC_METHOD_NOT_FOUND);
		}
		const request = parsePermissionRequest(params);
		if (!request) throw new AcpRpcError("Malformed session/request_permission params", RPC_INVALID_PARAMS);
		// A cancelled turn must answer every pending permission request with `cancelled`.
		if (signal?.aborted) return { outcome: { outcome: "cancelled" } };
		// WHY: a summary needs no tools and Pi shows none of an isolated session's calls, so nothing there is
		// granted (fail closed); this is the only guard for agents without `ownToolsOffMeta` (Codex).
		if (isolated) return { outcome: autoReject(request) };
		// Pi's own tools pass Pi's approval when Pi runs them, so the agent's check is redundant.
		if (isPiToolPermission(agent, client, sessionId, params)) return { outcome: autoApprove(request) };
		const outcome = await policy(request, { agent, signal });
		return { outcome: signal?.aborted ? { outcome: "cancelled" } : outcome };
	};

	/**
	 * Whether a permission request is for a call of Pi's own tool, judged by structure only: it
	 * comes from the main client on the live turn's session, which holds a bound MCP nonce, and names a
	 * tool call the agent reported and has not reported finished; every structured identity of the request
	 * and of the call's reports names one and the same tool Pi serves.
	 * WARN: titles, `piOwned`, `mirrored` and `state` are display state an agent or a tool's input can steer;
	 * never use them here.
	 */
	function isPiToolPermission(agent: AcpAgentConfig, client: AcpClient, sessionId: string | undefined, params: unknown): boolean {
		const turn = liveOn(client, sessionId);
		if (
			!turn ||
			turn.agent.id !== agent.id ||
			!turn.accepting ||
			turn.outcome ||
			turn.controller.signal.aborted
		) {
			return false;
		}
		if (![...nonceSessions.values()].some((owner) => owner.agentId === agent.id && owner.sessionId === sessionId && owner.instance === turn.instance)) return false;
		const toolCall = isRecord(params) && isRecord(params.toolCall) ? params.toolCall : undefined;
		const id = typeof toolCall?.toolCallId === "string" ? toolCall.toolCallId : undefined;
		const entry = id === undefined ? undefined : turn.tools.get(id);
		if (!toolCall || !entry || isFinalStatus(entry.reportedStatus)) return false;
		const tools = new Set(
			[...structuredToolIdentities(toolCall), ...entry.identities].map((identity) =>
				typeof identity === "string" ? (PI_TOOL_NAME.exec(identity)?.[1] ?? null) : null,
			),
		);
		const [tool, ...others] = tools;
		return (
			typeof tool === "string" &&
			others.length === 0 &&
			(served.get(turn.key) ?? []).some((candidate) => candidate.name === tool)
		);
	}

	/** Forgets an ended session of `agent`'s main client and revokes its MCP nonces. */
	function endSession(agent: AcpAgentConfig, client: AcpClient, sessionId: string): void {
		sync.delete(turnKey(agent.id, sessionId));
		wiringOf(client).delete(sessionId);
		for (const [nonce, owner] of nonceSessions) {
			if (owner.agentId !== agent.id || owner.sessionId !== sessionId) continue;
			nonceSessions.delete(nonce);
			revokeNonce(nonce);
		}
	}

	const newClient = (
		agent: AcpAgentConfig,
		clientCwd: string,
		role: { isolated: true } | { onSessionEnd: (client: AcpClient, sessionId: string) => void },
	): AcpClient => {
		const isolated = "isolated" in role;
		const onSessionEnd = "onSessionEnd" in role ? role.onSessionEnd : undefined;
		const client: AcpClient = new AcpClient({
			agent,
			cwd: clientCwd,
			handshakeTimeoutMs: options.config.handshakeTimeoutMs,
			cancelGraceMs: options.cancelGraceMs,
			onServerRequest: (request) => answerServerRequest(agent, client, isolated, request),
			onSessionEnd: onSessionEnd && ((sessionId) => onSessionEnd(client, sessionId)),
			log: log && ((message) => log(`${agent.id}: ${message}`)),
		});
		return client;
	};

	const getClient = (agent: AcpAgentConfig): AcpClient => {
		let client = clients.get(agent.id);
		if (!client) {
			// WARN: only the main client reports ended sessions: session ids are unique per process at
			// best, so an isolated client's ended session could share an id with a live main one.
			client = newClient(agent, cwd, { onSessionEnd: (owner, sessionId) => endSession(agent, owner, sessionId) });
			clients.set(agent.id, client);
		}
		return client;
	};

	/**
	 * Pi's MCP server for `agent`'s next session, started on first use; null when the agent does not take
	 * Pi's tools, or when the server cannot start and the agent may go on with its own (`piTools: "both"`).
	 * A failed start is retried on the next turn.
	 * WARN: a `piTools: "pi"` agent throws instead (fail closed): its own tools are switched off in favour of
	 * Pi's, and starting it without them would fall back to built-ins Pi neither approves nor shows.
	 */
	async function mcpServer(agent: AcpAgentConfig): Promise<PiMcpServer | null> {
		if (agent.piTools === "off") return null;
		mcp ??= new PiMcpServer({ tools: (_agentId, nonce) => served.get(nonceServed.get(nonce) ?? "") ?? [], call: handleMcpCall, log });
		try {
			await mcp.start();
			return mcp;
		} catch (error) {
			log?.(`Pi's MCP server failed to start: ${errorMessage(error)}`);
			if (agent.piTools === "pi") {
				throw new Error(
					`Pi's tool server (MCP server "${PI_MCP_SERVER}") failed to start: ${errorMessage(error)}. ${agent.name} works with Pi's tools only, so this turn stops instead of using its built-in tools. Send the message again to retry.`,
				);
			}
			return null;
		}
	}

	/** Pi's current tools as served to `agent` in the conversation `servedKey`; tells connected agents when the list changed. */
	function updateServedTools(agent: AcpAgentConfig, servedKey: string, context: TranscriptContext): void {
		if (agent.piTools === "off") return;
		const tools = getCurrentTools(context.messages).flatMap((tool) => {
			if (tool.name === ACP_TOOL_NAME) return [];
			const mcpTool = toMcpTool(tool);
			if (!mcpTool) log?.(`${agent.id}: tool '${tool.name}' has no object schema and is not offered over MCP`);
			return mcpTool ? [mcpTool] : [];
		});
		const previous = served.get(servedKey);
		served.set(servedKey, tools);
		if (previous && !sameJson(previous, tools)) mcp?.notifyToolsChanged(agent.id);
	}

	/**
	 * `session/new` params for Pi's tools and system prompt, chosen from what the agent advertises; the
	 * session's MCP nonce serves the tools of the conversation `servedKey`.
	 */
	function sessionSetup(
		agent: AcpAgentConfig,
		client: AcpClient,
		server: PiMcpServer | null,
		servedKey: string,
		systemPrompt: string,
		onPrompt: () => void,
	): SessionSetup {
		const key = `pi:mcp=${server && agent.piTools !== "off" ? agent.mcpTransport : "none"}`;
		return {
			key,
			params: (info) => {
				const params: Record<string, unknown> = {};
				const piOnly = agent.piTools === "pi" && agent.ownToolsOffMeta !== undefined;
				if (server && agent.piTools !== "off") {
					const transport = agent.mcpTransport === "http" && info.capabilities.mcpHttp ? "http" : "stdio";
					const nonce = server.mint(agent.id);
					mintedNonces.set(params, nonce);
					nonceServed.set(nonce, servedKey);
					params.mcpServers = [server.serverFor(agent.id, transport, nonce)];
				}
				if (agent.systemPrompt === "session" && info.capabilities.systemPrompt && systemPrompt) {
					params.systemPrompt = { mode: "append", content: [{ type: "text", text: systemPrompt }] };
					onPrompt();
				}
				const meta =
					piOnly && params.mcpServers
						? deepMerge(agent.sessionMeta ?? {}, agent.ownToolsOffMeta ?? {})
						: agent.sessionMeta;
				if (meta && Object.keys(meta).length > 0) params._meta = meta;
				return params;
			},
			settled: (params, sessionId) => {
				if (sessionId !== undefined) wiringOf(client).set(sessionId, key);
				const nonce = mintedNonces.get(params);
				if (nonce === undefined) return;
				mintedNonces.delete(params);
				if (sessionId === undefined) revokeNonce(nonce);
				else nonceSessions.set(nonce, { agentId: agent.id, sessionId, instance: client.instance(sessionId) });
			},
		};
	}

	/**
	 * `session/new` params for an isolated request: no Pi tools, and the agent's own tools off whenever it has
	 * a switch for them, whatever `piTools` says, because a summary needs no tools (fail closed).
	 * WARN: agents without `ownToolsOffMeta` (Codex) get the bare params isolated sessions always had, without
	 * `sessionMeta`, and keep their built-in tools; ACP has no generic switch. Their permission requests are
	 * still rejected in `answerServerRequest`, but a tool the agent runs without asking is not stopped.
	 */
	function isolatedSetup(agent: AcpAgentConfig): SessionSetup {
		return {
			key: "isolated",
			params: () => (agent.ownToolsOffMeta ? { _meta: deepMerge(agent.sessionMeta ?? {}, agent.ownToolsOffMeta) } : {}),
		};
	}

	async function openSession(request: SessionRequest): Promise<OpenedSession> {
		const { agent, acp, key, conversationId, modelId, model, streamOptions, messages, systemPrompt, signal } = request;
		const server = await mcpServer(agent);
		signal.throwIfAborted();
		let promptInSession = false;
		const setup = sessionSetup(agent, acp, server, key, systemPrompt, () => (promptInSession = true));
		let sessionId = await acp.ensureSession({ signal, setup, conversation: conversationId });
		let known = sync.get(turnKey(agent.id, sessionId));
		// WHY: the client keeps a used session whatever its key, so one opened while Pi's MCP server was
		// down would never gain Pi's tools; a new session seeded from Pi's history does.
		const rewired = wiringOf(acp).get(sessionId) !== setup.key;
		if (rewired || (known && diverged(known, messages))) {
			log?.(
				rewired
					? `${agent.id}: session ${sessionId} lacks Pi's tool wiring (${setup.key}); starting a new session`
					: `${agent.id}: Pi history diverged from session ${sessionId}; starting a new session`,
			);
			sync.delete(turnKey(agent.id, sessionId));
			acp.resetSession(cwd, conversationId);
			promptInSession = false;
			sessionId = await acp.ensureSession({ signal, setup, conversation: conversationId });
			known = undefined;
		}
		const state = known ?? newSyncState(promptInSession ? systemPrompt : "");
		sync.set(turnKey(agent.id, sessionId), state);
		keepConfirmed(state, messages);
		const info = acp.info ?? (await acp.ensureReady(signal));
		const instance = acp.instance(sessionId);
		await applySettings(acp, agent, sessionId, modelId, model, streamOptions, signal, instance);
		signal.throwIfAborted();
		return { sessionId, instance, state, info };
	}

	function rawUpdateHook(streamOptions: StreamOptions | undefined, model: Model<Api>): ((params: unknown) => void) | undefined {
		const onProviderStreamEvent = streamOptions?.onProviderStreamEvent;
		return (
			onProviderStreamEvent &&
			((params: unknown) => {
				Promise.resolve(onProviderStreamEvent(params, model)).catch((error: unknown) =>
					log?.(`onProviderStreamEvent failed: ${errorMessage(error)}`),
				);
			})
		);
	}

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
		const extra = streamOptions as (StreamOptions & { disableContinuation?: boolean }) | undefined;
		// A cache-warming request: ACP agents keep their own cache, so there is nothing to warm.
		if (extra?.disableContinuation === true && extra.maxTokens === 1) {
			stream.push({ type: "start", partial: message });
			writer.done("stop");
			return;
		}
		if (streamOptions?.cacheRetention === "none") {
			await runIsolated({ stream, writer, message, agent, modelId: target.modelId, model, context, streamOptions });
			return;
		}
		const conversationId = streamOptions?.sessionId ?? "";
		const key = turnKey(agent.id, conversationId);
		updateServedTools(agent, key, context);

		const messages = conversation(context.messages);
		const tail = messages.slice(tailStart(messages));
		const live = lives.get(key);
		if (live && !live.controller.signal.aborted && continuesLiveTurn(live, tail)) {
			syncOf(live)?.ownStamps.add(message.timestamp);
			stream.push({ type: "start", partial: message });
			await continueTurn(live, tail);
			attach(live, writer, signal);
			return;
		}

		live?.unbindAbort?.();
		claims.get(key)?.abort();
		claims.delete(key);
		lives.delete(key);
		if (requestUsers(messages).length === 0) {
			writer.fail("error", `No user message found to send to ${agent.name}.`);
			return;
		}
		// WHY: claimed before the first await, so a later request for the key aborts this one even while
		// it is still setting up, and this one can never publish a turn over the later one's.
		const claim = new AbortController();
		claims.set(key, claim);
		const setupSignal = signal ? AbortSignal.any([signal, claim.signal]) : claim.signal;
		const acp = getClient(agent);
		const abortedOrError = (error: unknown): void => {
			if (claims.get(key) === claim) claims.delete(key);
			failTurn(writer, error, { aborted: setupSignal.aborted, agent, auth: acp.info });
		};

		const systemPrompt = getCurrentSystemPrompt(context.messages);
		let opened: OpenedSession;
		try {
			opened = await openSession({ agent, acp, key, conversationId, modelId: target.modelId, model, streamOptions, messages, systemPrompt, signal: setupSignal });
			// WHY: the request can be superseded while this await resumes, after openSession's own check; it
			// must then end before it prunes pending steers or starts the stream.
			setupSignal.throwIfAborted();
		} catch (error) {
			abortedOrError(error);
			return;
		}
		const { sessionId, instance, state, info } = opened;

		const prompt = buildPromptBlocks({
			systemPromptMode: agent.systemPrompt,
			systemPrompt,
			state,
			messages,
			held: pendingSteers.recover(key, messages, state),
			images: info.capabilities.image,
			replyTimestamp: message.timestamp,
			steers: pendingSteers,
			owner: key,
		});
		if (!prompt) {
			if (claims.get(key) === claim) claims.delete(key);
			writer.fail("error", `No user message left to send to ${agent.name}.`);
			return;
		}

		stream.push({ type: "start", partial: message });
		const onRawUpdate = rawUpdateHook(streamOptions, model);
		let payload: unknown;
		let sent: PromptPart[];
		try {
			const defaultPayload = { sessionId, prompt: prompt.blocks };
			// WHY: a superseded request must end now, not once its hook returns.
			payload = (await abortable(Promise.resolve(streamOptions?.onPayload?.(defaultPayload, model)), setupSignal)) ?? defaultPayload;
			sent = partsIn(prompt.parts, payload);
			setupSignal.throwIfAborted();
			// WHY: checked before the turn is published: while the hook ran, the agent may have restarted
			// and handed this session id to another conversation's new session.
			if (acp.instance(sessionId) !== instance) {
				throw new Error(`${agent.name} session ${sessionId} ended while the request was being prepared; send the message again.`);
			}
		} catch (error) {
			abortedOrError(error);
			return;
		}
		const sentSteers = sent.flatMap((part) => (part.steer === undefined ? [] : [part.steer]));

		const turn = new LiveTurn({
			key,
			agent,
			modelId: model.id,
			acp,
			sessionId,
			instance,
			controller: claim,
			native: options.nativeTools?.() === true,
			steering: info.capabilities.steering,
			onRawUpdate,
		});
		lives.set(key, turn);
		startPrompt(turn, { payload, commit: () => prompt.commit(sent), steers: sentSteers });
		attach(turn, writer, signal);
	}

	/** Whether the messages Pi added answer tool calls of `turn`, so the same ACP turn goes on. */
	function continuesLiveTurn(turn: LiveTurn, tail: readonly Message[]): boolean {
		return tail.some(
			(candidate) =>
				candidate.role === "toolResult" &&
				(turn.issued.has(candidate.toolCallId) || turn.mirrors.has(candidate.toolCallId)),
		);
	}

	/**
	 * Hands Pi's tool results to the agent's waiting MCP requests and delivers
	 * the steering messages Pi queued after them.
	 */
	async function continueTurn(turn: LiveTurn, tail: readonly Message[]): Promise<void> {
		const state = syncOf(turn);
		const results = tail.filter(
			(candidate): candidate is ToolResultMessage => candidate.role === "toolResult" && turn.piCalls.has(candidate.toolCallId),
		);
		const steers = tail.filter(
			(candidate): candidate is UserMessage =>
				candidate.role === "user" && !state?.delivered.has(messageKey(candidate)) && !pendingSteers.has(turn.key, messageKey(candidate)),
		);
		const steerKeys = steers.map(messageKey);
		const image = turn.acp.info?.capabilities.image === true;
		const steerBlocks = steers.flatMap((steer) => userBlocks(steer, image));

		let steerDelivered = steerBlocks.length === 0;
		// The steer goes first, so the agent has it queued before it reads the tool results.
		if (!steerDelivered && turn.steering && turn.running) {
			try {
				steerDelivered = (await turn.acp.steer(turn.sessionId, steerBlocks)) === "steered";
				if (steerDelivered) pendingSteers.deliver(turn.key, state, steerKeys);
			} catch (error) {
				log?.(`${turn.agent.id}: _session/steering failed, falling back: ${errorMessage(error)}`);
			}
		}
		for (const [index, result] of results.entries()) {
			const call = turn.piCalls.get(result.toolCallId);
			if (!call) continue;
			turn.piCalls.delete(result.toolCallId);
			const reply = toMcpResult(result);
			if (!steerDelivered && index === results.length - 1) {
				// WHY: MCP results carry images whatever the agent's prompt capabilities say.
				const blocks = steers.flatMap((steer) => userBlocks(steer, true));
				const texts = blocks.flatMap((block) => (block.type === "text" ? [block.text] : []));
				const images = blocks.flatMap((block) => (block.type === "image" ? [block] : []));
				const attached = images.length > 0 ? `\n[${images.length} image(s) attached after this message]` : "";
				reply.content.push(
					{
						type: "text",
						text: `<user-message>\nThe user sent this while the tool ran. Take it into account now:\n${texts.join("\n")}${attached}\n</user-message>`,
					},
					...images,
				);
				steerDelivered = true;
				pendingSteers.deliver(turn.key, state, steerKeys);
			}
			call.resolve(reply);
		}
		if (steerDelivered) return;
		pendingSteers.hold(turn.key, steers);
		enqueuePrompt(turn, {
			payload: { sessionId: turn.sessionId, prompt: steerBlocks },
			commit: () => pendingSteers.deliver(turn.key, syncOf(turn), steerKeys),
			steers: steerKeys,
		});
	}

	/** An MCP `tools/call` from the agent: becomes a Pi tool call in the running turn. */
	function handleMcpCall(call: McpCall): Promise<McpCallResult> {
		const owner = nonceSessions.get(call.nonce);
		const turn =
			owner &&
			[...lives.values()].find(
				(candidate) => candidate.agent.id === owner.agentId && candidate.instance === owner.instance && isCurrent(candidate),
			);
		if (
			!turn ||
			turn.agent.id !== call.agentId ||
			!turn.accepting ||
			turn.outcome ||
			turn.controller.signal.aborted
		) {
			return Promise.reject(new Error("Pi is not running a turn for this agent, so its tools are unavailable."));
		}
		if (!(served.get(turn.key) ?? []).some((tool) => tool.name === call.name)) {
			return Promise.reject(new Error(`Pi has no tool named '${call.name}'.`));
		}
		// The agent's own report of this call is shown by Pi's tool row instead.
		for (const entry of [...turn.tools.values()].reverse()) {
			if (entry.piOwned || isFinalStatus(entry.state.status)) continue;
			if (piToolRef(undefined, entry.title) === call.name || sameJson(entry.rawInput, call.arguments)) {
				entry.piOwned = true;
				break;
			}
		}
		// A mirrored row Pi is waiting on would block it from running this call, which the agent waits on.
		for (const entry of turn.tools.values()) {
			if (entry.waiters.size === 0) continue;
			if (entry.piOwned) entry.state = { ...entry.state, status: "completed", content: { ...emptyToolContent(), text: [`Handed to Pi as ${call.name}.`] } };
			notifyWaiters(entry, true);
		}
		const id = `pi_${randomUUID()}`;
		turn.issued.add(id);
		syncOf(turn)?.ours.add(id);
		return new Promise<McpCallResult>((resolve, reject) => {
			const onAbort = (): void => {
				turn.piCalls.delete(id);
				reject(new Error("The tool call was cancelled."));
			};
			call.signal.addEventListener("abort", onAbort, { once: true });
			turn.piCalls.set(id, {
				name: call.name,
				resolve: (result) => {
					call.signal.removeEventListener("abort", onAbort);
					resolve(result);
				},
			});
			dispatch(turn, { type: "pi_tool_call", id, name: call.name, arguments: call.arguments });
		});
	}

	/** Runs a request outside the conversation (a summary) in a throwaway agent process. */
	async function runIsolated({ stream, writer, message, agent, modelId, model, context, streamOptions }: IsolatedRequest): Promise<void> {
		const signal = streamOptions.signal;
		const client = newClient(agent, cwd, { isolated: true });
		try {
			const sessionId = await client.ensureSession({ signal, setup: isolatedSetup(agent) });
			await applySettings(client, agent, sessionId, modelId, model, streamOptions);
			const info = client.info ?? (await client.ensureReady(signal));
			const blocks: AcpContentBlock[] = [];
			const systemPrompt = getCurrentSystemPrompt(context.messages);
			if (systemPrompt) blocks.push(tagged("system-instructions", systemPrompt, "Instructions for this request:"));
			const messages = conversation(context.messages);
			const users = messages.filter((candidate): candidate is UserMessage => candidate.role === "user");
			const others = messages.filter((candidate) => candidate.role !== "user");
			if (others.length > 0) blocks.push(tagged("conversation-history", renderTranscript(messages), "The conversation:"));
			else for (const user of users) blocks.push(...userBlocks(user, info.capabilities.image));
			stream.push({ type: "start", partial: message });
			let contextUsed: number | undefined;
			const result = await client.prompt({
				sessionId,
				payload: { sessionId, prompt: blocks },
				signal,
				onUpdate: (update) => {
					if (update.type === "agent_message_chunk") writer.append("text", update.text);
					else if (update.type === "usage_update") {
						contextUsed = update.used;
						if (update.size !== undefined) reportedWindows.set(model.id, update.size);
					}
				},
			});
			if (result.kind === "aborted") writer.fail("aborted", "Request was aborted");
			else {
				const usage = zeroUsage();
				addUsage(usage, result.usage, contextUsed);
				finishTurn(writer, agent, result.stopReason, usage);
			}
		} catch (error) {
			failTurn(writer, error, { aborted: signal?.aborted ?? false, agent });
		} finally {
			client.dispose();
		}
	}

	const stream = (
		model: Model<Api>,
		context: TranscriptContext,
		streamOptions?: StreamOptions,
	): AssistantMessageEventStream => {
		const events = createAssistantMessageEventStream();
		runTurn(events, model, context, streamOptions).catch((error: unknown) =>
			log?.(`stream crashed: ${errorMessage(error)}`),
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
		getModels,
		refreshModels,
		stream,
		streamSimple: stream,
	};

	return {
		provider,
		resetSession(nextCwd: string): void {
			cwd = nextCwd;
			for (const claim of claims.values()) claim.abort();
			claims.clear();
			lives.clear();
			pendingSteers.clear();
			for (const client of clients.values()) client.resetSession(nextCwd);
		},
		agentFor: (modelId) => parseModelId(modelId, agents)?.agent.id,
		awaitTool,
		commands: async (agentId) => (await clients.get(agentId)?.currentCommands()) ?? [],
		dispose(): void {
			for (const claim of claims.values()) claim.abort();
			claims.clear();
			lives.clear();
			for (const client of clients.values()) client.dispose();
			clients.clear();
			sync.clear();
			pendingSteers.clear();
			nonceSessions.clear();
			nonceServed.clear();
			mcp?.dispose();
			mcp = null;
		},
	};
}
