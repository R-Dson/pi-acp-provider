import { randomUUID } from "node:crypto";
import type { ToolCall, Usage, UserMessage } from "@earendil-works/pi-ai";
import { AcpRpcError, type AcpClient, type PromptResult, type SessionInstance } from "./acp-client.ts";
import { emptyToolContent, isFinalStatus, type AcpAgentInfo, type AcpSessionUpdate } from "./acp.ts";
import { PI_MCP_SERVER, type AcpAgentConfig } from "./config.ts";
import { addUsage, failTurn, finishTurn, zeroUsage, type MessageWriter } from "./message-writer.ts";
import type { McpCallResult } from "./mcp-server.ts";
import { ACP_TOOL_NAME, compactInput, type AcpToolArgs, type AcpToolState, type AwaitTool } from "./native-tools.ts";
import { MAX_STEER_REJECTIONS, noteStreamed, type PendingSteers, type SyncState } from "./session-sync.ts";

/**
 * Tool names and titles agents give calls to Pi's MCP server, as observed:
 * Claude `mcp__pi__read`, Codex `mcp.pi.read`, Vibe `mcp_pi.read`,
 * Antigravity `mcp_pi_read` (titled "Run mcp_pi_read"), and `pi/read`.
 */
const PI_TOOL_REF = new RegExp(`(?:^|[\\s:(\\[])(?:mcp(?:__|[._])${PI_MCP_SERVER}(?:__|[._/])|${PI_MCP_SERVER}/)([\\w-]+)`);
/** A whole structured tool name (never a title) for a tool of Pi's MCP server, in the namings above. */
export const PI_TOOL_NAME = new RegExp(`^(?:mcp(?:__|[._])${PI_MCP_SERVER}(?:__|[._/])|${PI_MCP_SERVER}/)(.+)$`);

/** The text of a user message, images left out. */
function plainText(message: UserMessage): string {
	if (typeof message.content === "string") return message.content;
	return message.content.flatMap((block) => (block.type === "text" ? [block.text] : [])).join("\n");
}

/** First line of tool output, capped so huge outputs don't flood the transcript. */
function summarize(text: string, max = 200): string {
	const line = text.trim().split("\n", 1)[0] ?? "";
	return line.length > max ? `${line.slice(0, max)}…` : line;
}

type TurnOutcome = { result: PromptResult } | { error: unknown; auth: AcpAgentInfo | null };

/** Whether the prompt ran to the agent's `end_turn`, so the turn may take its next prompt. */
const endedCleanly = (outcome: TurnOutcome | null): boolean =>
	outcome !== null && "result" in outcome && outcome.result.kind === "finished" && outcome.result.stopReason === "end_turn";

/** A Pi tool call the agent made over MCP, waiting for Pi's result. */
type PiToolEvent = { type: "pi_tool_call"; id: string; name: string; arguments: Record<string, unknown> };
type TurnEvent = AcpSessionUpdate | PiToolEvent;

interface ToolWaiter {
	resolve: (state: AcpToolState) => void;
	onPartial: (state: AcpToolState) => void;
}

interface ToolEntry {
	title: string | undefined;
	/** The mirror tool's arguments, fixed by the first update so the tool row does not change. */
	args: AcpToolArgs | undefined;
	rawInput: unknown;
	/** Every structured identity the agent's reports of this call carried (`structuredToolIdentities`), never the title. */
	identities: unknown[];
	/** The status the agent last reported; `state.status` can also be set by Pi's handoff of a call it runs. */
	reportedStatus: AcpToolState["status"];
	/** Whether a Pi message already shows this call as a tool row. */
	mirrored: boolean;
	/** A call to Pi's MCP server: Pi shows and runs it as its own tool call, so it is not mirrored. */
	piOwned: boolean;
	state: AcpToolState;
	waiters: Set<ToolWaiter>;
}

interface PiCall {
	name: string;
	resolve: (result: McpCallResult) => void;
}

/** One `session/prompt` of a turn and what it delivers once the agent accepts it. */
interface TurnPrompt {
	payload: unknown;
	/**
	 * Records what the prompt delivered; runs once, on the first update attributed to the prompt or its
	 * `finished` result. WARN: nothing the prompt carries may be recorded before this: a rejected or
	 * aborted prompt must leave the sync state as it was, so the retry sends it all again.
	 */
	commit: () => void;
	/** Keys of the pending steers the prompt carries: rejected with it when the agent rejects it. */
	steers: readonly string[];
}

export type LiveTurnSetup = Pick<
	LiveTurn,
	"key" | "agent" | "modelId" | "acp" | "sessionId" | "instance" | "controller" | "native" | "steering" | "onRawUpdate"
>;

export class LiveTurn {
	/** `turnKey(agent id, Pi's StreamOptions.sessionId or "")`: the turn's slot in `lives`, `claims`, `served` and `pendingSteers`. */
	key: string;
	agent: AcpAgentConfig;
	modelId: string;
	acp: AcpClient;
	sessionId: string;
	/** The session instance the turn set up; its prompts are refused once that one ended. */
	instance: SessionInstance | undefined;
	controller: AbortController;
	/**
	 * Removes the listener that aborts the turn with the signal of the Pi message last attached.
	 * WARN: it must outlive that message: while Pi runs a Pi tool call between messages, it is the only
	 * link that cancels the ACP turn. Only a new `attach`, `settleTurn` or the turn's replacement drops it.
	 */
	unbindAbort: (() => void) | undefined = undefined;
	native: boolean;
	steering: boolean;
	/** Events that arrived while no Pi message was attached (Pi was running tools). */
	buffer: TurnEvent[] = [];
	attached: { consume: (event: TurnEvent) => void; finish: (outcome: TurnOutcome) => void } | null = null;
	outcome: TurnOutcome | null = null;
	/** Whether a `session/prompt` of this turn is in flight. */
	running: boolean = false;
	/**
	 * Whether the session's traffic belongs to this turn: false until its first prompt is sent,
	 * because until then the session may still be draining a cancelled prompt.
	 */
	accepting: boolean = false;
	/** Prompts to send, in order, once the running one ends (steering the agent could not take mid-turn). */
	queued: TurnPrompt[] = [];
	tools: Map<string, ToolEntry> = new Map();
	/**
	 * Mirrored calls by the id Pi knows them by: unique per call, since ACP ids are unique per session at
	 * best and Pi's tool results and `awaitTool` must find the one turn they belong to.
	 */
	mirrors: Map<string, ToolEntry> = new Map();
	/** Pi tool calls still waiting for Pi's result; the agent may cancel one or end the turn first. */
	piCalls: Map<string, PiCall> = new Map();
	/** Every Pi tool call this turn handed to Pi: Pi answers each, even once its MCP request is gone. */
	issued: Set<string> = new Set();
	usage: Usage = zeroUsage();
	/** The agent's latest `usage_update.used`: what its context holds now. */
	contextUsed: number | undefined = undefined;
	onRawUpdate: ((params: unknown) => void) | undefined;
	/** Told to the user when the turn ends in an error, after the error itself. */
	notice: string | undefined = undefined;

	constructor(setup: LiveTurnSetup) {
		this.key = setup.key;
		this.agent = setup.agent;
		this.modelId = setup.modelId;
		this.acp = setup.acp;
		this.sessionId = setup.sessionId;
		this.instance = setup.instance;
		this.controller = setup.controller;
		this.native = setup.native;
		this.steering = setup.steering;
		this.onRawUpdate = setup.onRawUpdate;
	}
}

/** Renders tool activity as short inline notices in the reply text. */
function createToolRenderer(append: (text: string) => void) {
	const titles = new Map<string, string>();
	return (update: Extract<AcpSessionUpdate, { type: "tool_call" | "tool_call_update" }>): void => {
		if (update.title) titles.set(update.toolCallId, update.title);
		const title = titles.get(update.toolCallId) ?? "tool";
		if (update.type === "tool_call") append(`\n⚙️ Executing [${title}]...\n`);
		if (!isFinalStatus(update.status)) return;
		const mark = update.status === "completed" ? "✓" : "✗";
		const summary = summarize(update.summary);
		append(`${mark} ${title}${summary ? `: ${summary}` : ""}\n`);
	};
}

/** The Pi tool an agent tool call refers to, from its tool name or title, if it is one of Pi's. */
export function piToolRef(toolName: string | undefined, title: string | undefined): string | undefined {
	for (const text of [toolName, title]) {
		const match = text ? PI_TOOL_REF.exec(text) : null;
		if (match?.[1]) return match[1];
	}
	return undefined;
}

export interface TurnLifecycleDeps {
	lives: Map<string, LiveTurn>;
	claims: Map<string, AbortController>;
	pendingSteers: PendingSteers;
	syncOf: (turn: LiveTurn) => SyncState | undefined;
	reportedWindows: Map<string, number>;
	log: ((message: string) => void) | undefined;
}

export function createTurnLifecycle({ lives, claims, pendingSteers, syncOf, reportedWindows, log }: TurnLifecycleDeps) {
	/**
	 * Sends `prompt` as the turn's next prompt: now if the agent finished, else once the running prompt ends.
	 * A turn that ended otherwise drops it; its steers stay pending for the next prompt.
	 */
	function enqueuePrompt(turn: LiveTurn, prompt: TurnPrompt): void {
		if (!turn.running && endedCleanly(turn.outcome)) {
			turn.outcome = null;
			startPrompt(turn, prompt);
		} else if (turn.running) turn.queued.push(prompt);
	}

	function startPrompt(turn: LiveTurn, prompt: TurnPrompt): void {
		turn.running = true;
		let accepted = false;
		const accept = (): void => {
			if (accepted || turn.controller.signal.aborted) return;
			accepted = true;
			prompt.commit();
		};
		turn.acp
			.prompt({
				sessionId: turn.sessionId,
				instance: turn.instance,
				payload: prompt.payload,
				signal: turn.controller.signal,
				onRawUpdate: turn.onRawUpdate,
				onSent: () => (turn.accepting = true),
				onUpdate: (update) => {
					accept();
					dispatch(turn, update);
				},
			})
			.then(
				(result) => {
					if (result.kind === "finished") accept();
					settleTurn(turn, { result });
				},
				(error: unknown) => {
					if (!accepted && error instanceof AcpRpcError) rejectPrompt(turn, prompt);
					settleTurn(turn, { error, auth: turn.acp.info });
				},
			);
	}

	/** Counts the agent's rejection of `prompt` against its steers; tells the user about the ones dropped. */
	function rejectPrompt(turn: LiveTurn, prompt: TurnPrompt): void {
		const dropped = pendingSteers.reject(turn.key, prompt.steers);
		if (dropped.length === 0) return;
		const quoted = dropped.map((steer) => `"${summarize(plainText(steer.message), 80)}"`).join(", ");
		turn.notice = `Pi dropped ${dropped.length} steering message(s) that ${turn.agent.name} rejected ${MAX_STEER_REJECTIONS} times: ${quoted}. Send again what still matters.`;
		log?.(`${turn.agent.id}: ${turn.notice}`);
	}

	/** Records an update, then hands it to the attached message or buffers it for the next one. */
	function dispatch(turn: LiveTurn, event: TurnEvent): void {
		if (event.type === "usage_update") {
			turn.contextUsed = event.used;
			if (event.size !== undefined) reportedWindows.set(turn.modelId, event.size);
			return;
		}
		if (event.type === "tool_call" || event.type === "tool_call_update") {
			let entry = turn.tools.get(event.toolCallId);
			if (!entry) {
				entry = {
					title: undefined,
					args: undefined,
					rawInput: undefined,
					identities: [],
					reportedStatus: undefined,
					mirrored: false,
					piOwned: false,
					state: { status: undefined, content: emptyToolContent(), rawOutput: undefined },
					waiters: new Set(),
				};
				turn.tools.set(event.toolCallId, entry);
			}
			if (event.title) entry.title = event.title;
			if (event.status) entry.state.status = entry.reportedStatus = event.status;
			if (event.content) entry.state.content = event.content;
			if (event.rawOutput !== undefined) entry.state.rawOutput = event.rawOutput;
			if (event.rawInput !== undefined) entry.rawInput = event.rawInput;
			entry.identities.push(...event.toolIdentities);
			if (!entry.mirrored && piToolRef(event.toolName, entry.title)) entry.piOwned = true;
			if (!entry.args) {
				const input = compactInput(event.rawInput);
				entry.args = {
					agent: turn.agent.id,
					title: entry.title ?? "tool",
					...(event.kind ? { kind: event.kind } : {}),
					...(event.locations?.length ? { locations: event.locations } : {}),
					...(input !== undefined ? { input } : {}),
				};
			}
			if (entry.mirrored) notifyWaiters(entry);
		}
		if (turn.attached) turn.attached.consume(event);
		else turn.buffer.push(event);
	}

	function notifyWaiters(entry: ToolEntry, force = false): void {
		const final = force || isFinalStatus(entry.state.status);
		for (const waiter of [...entry.waiters]) {
			if (final) {
				entry.waiters.delete(waiter);
				waiter.resolve(entry.state);
			} else waiter.onPartial(entry.state);
		}
	}

	/** Makes `signal` abort the turn, in place of the signal it was bound to. */
	function bindAbort(turn: LiveTurn, signal: AbortSignal | undefined): void {
		turn.unbindAbort?.();
		turn.unbindAbort = undefined;
		if (signal?.aborted) turn.controller.abort();
		if (!signal || signal.aborted) return;
		const onAbort = (): void => turn.controller.abort();
		signal.addEventListener("abort", onAbort, { once: true });
		turn.unbindAbort = () => signal.removeEventListener("abort", onAbort);
	}

	/** Streams the turn into one Pi message until the next tool call or the end of the turn. */
	function attach(turn: LiveTurn, writer: MessageWriter, signal: AbortSignal | undefined): void {
		bindAbort(turn, turn.outcome ? undefined : signal);
		const renderTool = createToolRenderer((text) => writer.append("text", text));
		let ended = false;
		const endAt = (toolCall: ToolCall): void => {
			ended = true;
			turn.attached = null;
			writer.toolUse(toolCall);
			noteStreamed(syncOf(turn), writer.message);
		};
		const consume = (event: TurnEvent): void => {
			if (ended) return;
			if (event.type === "agent_message_chunk") writer.append("text", event.text);
			else if (event.type === "agent_thought_chunk") writer.append("thinking", event.text);
			else if (event.type === "pi_tool_call") {
				if (!turn.piCalls.has(event.id)) return;
				endAt({ type: "toolCall", id: event.id, name: event.name, arguments: event.arguments as ToolCall["arguments"] });
			} else if (event.type === "tool_call" || event.type === "tool_call_update") {
				const entry = turn.tools.get(event.toolCallId);
				if (!entry || entry.piOwned) return;
				if (!turn.native) {
					renderTool(event);
					return;
				}
				if (entry.mirrored) return;
				entry.mirrored = true;
				const id = `acp_${randomUUID()}`;
				turn.mirrors.set(id, entry);
				syncOf(turn)?.ours.add(id);
				endAt({
					type: "toolCall",
					id,
					name: ACP_TOOL_NAME,
					arguments: (entry.args ?? { agent: turn.agent.id, title: entry.title ?? "tool" }) as unknown as ToolCall["arguments"],
				});
			}
		};
		const finish = (outcome: TurnOutcome): void => {
			if (ended) return;
			ended = true;
			turn.attached = null;
			if (lives.get(turn.key) === turn) lives.delete(turn.key);
			if (claims.get(turn.key) === turn.controller) claims.delete(turn.key);
			if ("error" in outcome)
				failTurn(writer, outcome.error, { aborted: turn.controller.signal.aborted, agent: turn.agent, auth: outcome.auth, notice: turn.notice });
			else if (outcome.result.kind === "aborted") writer.fail("aborted", "Request was aborted");
			else finishTurn(writer, turn.agent, outcome.result.stopReason, turn.usage);
			noteStreamed(syncOf(turn), writer.message);
		};
		while (turn.buffer.length > 0 && !ended) consume(turn.buffer.shift()!);
		if (ended) return;
		if (turn.outcome) {
			finish(turn.outcome);
			return;
		}
		turn.attached = { consume, finish };
	}

	function settleTurn(turn: LiveTurn, outcome: TurnOutcome): void {
		turn.running = false;
		if ("result" in outcome && outcome.result.kind === "finished") addUsage(turn.usage, outcome.result.usage, turn.contextUsed);
		const next = turn.queued.shift();
		if (next && endedCleanly(outcome) &&!turn.controller.signal.aborted) {
			startPrompt(turn, next);
			return;
		}
		// Their steers stay in `pendingSteers`, for the next prompt.
		turn.queued.length = 0;
		bindAbort(turn, undefined);
		turn.outcome = outcome;
		// Mirrored calls the agent never finalized end with what it last reported.
		for (const entry of turn.tools.values()) notifyWaiters(entry, true);
		for (const [id, call] of turn.piCalls) {
			turn.piCalls.delete(id);
			call.resolve({ content: [{ type: "text", text: "The turn ended before Pi ran this tool." }], isError: true });
		}
		turn.attached?.finish(outcome);
	}

	const awaitTool: AwaitTool = (toolCallId, signal, onPartial) => {
		const turn = [...lives.values()].find((candidate) => candidate.mirrors.has(toolCallId));
		const entry = turn?.mirrors.get(toolCallId);
		if (!turn || !entry) {
			return Promise.resolve({ status: "failed", content: { ...emptyToolContent(), text: ["This ACP turn is no longer running."] }, rawOutput: undefined });
		}
		if (turn.outcome || isFinalStatus(entry.state.status)) return Promise.resolve(entry.state);
		// A Pi tool call already waits behind this row; end the row so Pi gets to it.
		if (turn.buffer.some((event) => event.type === "pi_tool_call")) return Promise.resolve(entry.state);
		return new Promise((resolve, reject) => {
			const waiter: ToolWaiter = {
				resolve: (state) => {
					signal?.removeEventListener("abort", onAbort);
					resolve(state);
				},
				onPartial,
			};
			const onAbort = (): void => {
				entry.waiters.delete(waiter);
				turn.controller.abort();
				reject(new Error("Request was aborted"));
			};
			if (signal?.aborted) return onAbort();
			signal?.addEventListener("abort", onAbort, { once: true });
			entry.waiters.add(waiter);
		});
	};

	return { enqueuePrompt, startPrompt, dispatch, notifyWaiters, attach, awaitTool };
}
