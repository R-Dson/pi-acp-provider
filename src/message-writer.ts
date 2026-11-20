import type { AssistantMessage, AssistantMessageEventStream, ToolCall, Usage } from "@earendil-works/pi-ai";
import type { AcpAgentInfo, AcpStopReason, AcpUsage } from "./acp.ts";
import type { AcpAgentConfig } from "./config.ts";

export const zeroUsage = (): Usage => ({
	input: 0,
	output: 0,
	cacheRead: 0,
	cacheWrite: 0,
	totalTokens: 0,
	cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 },
});

/**
 * WARN: an ACP prompt result sums the usage of every model call in the turn, and each call re-reads the
 * whole context, so the sum overstates the context many times. Pi reads `totalTokens` as the context
 * size, so it takes the agent's `usage_update.used` instead when the agent sent one.
 * WARN: pi-ai's `isContextOverflow` also reads `input + cacheRead` of a finished message as its prompt
 * size and auto-compacts when that exceeds the window, so the summed prompt-side counts must not
 * outgrow `contextUsed` either; a long tool loop otherwise looks like an overflow every turn.
 */
export function addUsage(total: Usage, usage: AcpUsage | undefined, contextUsed?: number): void {
	if (usage) {
		total.input += usage.input;
		total.output += usage.output;
		total.cacheRead += usage.cacheRead;
		total.cacheWrite += usage.cacheWrite;
	}
	if (contextUsed === undefined) {
		total.totalTokens = total.input + total.output + total.cacheRead + total.cacheWrite;
		return;
	}
	total.totalTokens = contextUsed;
	const prompt = total.input + total.cacheRead + total.cacheWrite;
	if (prompt <= contextUsed) return;
	// WHY: scaling keeps the agent's input/cache split (its cache hit ratio) while making the parts add up
	// to the real context; ACP reports no per-call breakdown to pick the last call's numbers from instead.
	const scale = contextUsed / prompt;
	total.input = Math.floor(total.input * scale);
	total.cacheWrite = Math.floor(total.cacheWrite * scale);
	total.cacheRead = contextUsed - total.input - total.cacheWrite;
}

/**
 * Builds the streamed `AssistantMessage`: keeps one text or thinking block
 * open at a time and emits balanced start/delta/end events for it.
 */
export function createMessageWriter(stream: AssistantMessageEventStream, message: AssistantMessage) {
	/** The block being streamed and its content so far. */
	let open: { type: "text" | "thinking"; contentIndex: number; content: string } | null = null;

	const close = (): void => {
		if (!open) return;
		const { type, contentIndex, content } = open;
		stream.push({ type: `${type}_end`, contentIndex, content, partial: message });
		open = null;
	};

	const append = (type: "text" | "thinking", delta: string): void => {
		if (!delta) return;
		if (open?.type !== type) {
			close();
			open = { type, contentIndex: message.content.length, content: "" };
			message.content.push(type === "text" ? { type, text: "" } : { type, thinking: "" });
			stream.push({ type: `${type}_start`, contentIndex: open.contentIndex, partial: message });
		}
		open.content += delta;
		const block = message.content[open.contentIndex];
		if (block?.type === "text") block.text = open.content;
		else if (block?.type === "thinking") block.thinking = open.content;
		stream.push({ type: `${type}_delta`, contentIndex: open.contentIndex, delta, partial: message });
	};

	let settled = false;
	/** Closes the open block and ends the stream; later calls are ignored. */
	const settle = (push: () => void): void => {
		if (settled) return;
		settled = true;
		close();
		push();
		stream.end(message);
	};
	/** Ends the message at a tool call Pi runs or shows as a tool row; the ACP turn goes on in the next message. */
	const toolUse = (toolCall: ToolCall): void =>
		settle(() => {
			const contentIndex = message.content.length;
			message.content.push(toolCall);
			stream.push({ type: "toolcall_start", contentIndex, partial: message });
			stream.push({ type: "toolcall_end", contentIndex, toolCall, partial: message });
			message.stopReason = "toolUse";
			stream.push({ type: "done", reason: "toolUse", message });
		});
	const done = (reason: "stop" | "length", usage?: Usage): void =>
		settle(() => {
			if (usage) message.usage = usage;
			message.stopReason = reason;
			stream.push({ type: "done", reason, message });
		});
	const fail = (reason: "error" | "aborted", error: string): void =>
		settle(() => {
			message.stopReason = reason;
			message.errorMessage = error;
			stream.push({ type: "error", reason, error: message });
		});

	return { message, append, toolUse, done, fail };
}

export type MessageWriter = ReturnType<typeof createMessageWriter>;

/** Explains `auth_required`, listing the methods the agent advertised. */
export function authHint(agent: AcpAgentConfig, info: AcpAgentInfo | null): string {
	const methods = info?.authMethods ?? [];
	if (methods.length === 0) return `${agent.name} requires authentication. Sign in with the agent's own CLI, then retry.`;
	const list = methods
		.map((method) => `'${method.id}' (${method.name}${method.description ? `: ${method.description}` : ""})`)
		.join(", ");
	return agent.authMethod
		? `${agent.name} still requires authentication after trying method '${agent.authMethod}'. Available methods: ${list}.`
		: `${agent.name} requires authentication. Sign in with the agent's own CLI, or set "authMethod" for this agent in ~/.pi/agent/settings.json to one of: ${list}.`;
}

/** Ends the message according to the agent's `stopReason`. */
export function finishTurn(writer: MessageWriter, agent: AcpAgentConfig, stopReason: AcpStopReason, usage?: Usage): void {
	switch (stopReason) {
		case "end_turn":
			writer.done("stop", usage);
			return;
		case "max_tokens":
		case "max_turn_requests":
			writer.done("length", usage);
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
