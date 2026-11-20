import { randomUUID } from "node:crypto";
import { getCurrentTools, type TranscriptContext } from "@earendil-works/pi-ai";
import type { AcpClient, SessionInstance, SessionSetup } from "./acp-client.ts";
import { emptyToolContent, errorMessage, isFinalStatus, isRecord, structuredToolIdentities } from "./acp.ts";
import { PI_MCP_SERVER, deepMerge, type AcpAgentConfig } from "./config.ts";
import { PiMcpServer, toMcpTool, type McpCall, type McpCallResult, type McpTool } from "./mcp-server.ts";
import { ACP_TOOL_NAME } from "./native-tools.ts";
import type { SyncState } from "./session-sync.ts";
import { PI_TOOL_NAME, piToolRef, type LiveTurn, type createTurnLifecycle } from "./live-turn.ts";

type TurnLifecycle = ReturnType<typeof createTurnLifecycle>;

export interface PiToolBridgeDeps {
	lives: Map<string, LiveTurn>;
	isCurrent: (turn: LiveTurn) => boolean;
	syncOf: (turn: LiveTurn) => SyncState | undefined;
	wiringOf: (client: AcpClient) => Map<string, string>;
	dispatch: TurnLifecycle["dispatch"];
	notifyWaiters: TurnLifecycle["notifyWaiters"];
	log: ((message: string) => void) | undefined;
}

function sameJson(left: unknown, right: unknown): boolean {
	try {
		return JSON.stringify(left) === JSON.stringify(right);
	} catch {
		return false;
	}
}

export function createPiToolBridge({ lives, isCurrent, syncOf, wiringOf, dispatch, notifyWaiters, log }: PiToolBridgeDeps) {
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
	const liveOn = (client: AcpClient, sessionId: string | undefined): LiveTurn | undefined =>
		[...lives.values()].find((turn) => turn.acp === client && turn.sessionId === sessionId && isCurrent(turn));

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

	function revokeSessionNonces(agent: AcpAgentConfig, sessionId: string): void {
		for (const [nonce, owner] of nonceSessions) {
			if (owner.agentId !== agent.id || owner.sessionId !== sessionId) continue;
			nonceSessions.delete(nonce);
			revokeNonce(nonce);
		}
	}

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

	return {
		isPiToolPermission,
		revokeSessionNonces,
		mcpServer,
		updateServedTools,
		sessionSetup,
		dispose(): void {
			nonceSessions.clear();
			nonceServed.clear();
			mcp?.dispose();
			mcp = null;
		},
	};
}
