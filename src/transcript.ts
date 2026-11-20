/**
 * Pi transcript ↔ ACP prompt conversions.
 *
 * An ACP agent keeps its own conversation, so the provider sends only what
 * the agent has not seen: the queued user messages, and, for a session that
 * missed part of Pi's history (a resumed or forked Pi session, a switch from
 * another model, a rewritten branch), a plain-text transcript of that part.
 */

import type { Message, ToolResultMessage, UserMessage } from "@earendil-works/pi-ai";

/** An ACP prompt `ContentBlock`. */
export type AcpContentBlock = { type: "text"; text: string } | { type: "image"; data: string; mimeType: string };

/** Cap for a rendered transcript; the oldest part is dropped first. */
const MAX_TRANSCRIPT_CHARS = 120_000;
const MAX_TOOL_TEXT_CHARS = 2_000;

/** Messages that are conversation, not prompt or tool declarations. */
export function conversation(messages: readonly Message[]): Message[] {
	return messages.filter((message) => message.role !== "system");
}

/** Cheap identity of a message: role, timestamp and content size. */
export function messageKey(message: Message): string {
	const timestamp = "timestamp" in message ? message.timestamp : 0;
	let size = 0;
	const content = message.content;
	if (typeof content === "string") size = content.length;
	else for (const block of content) size += JSON.stringify(block).length;
	return `${message.role}:${timestamp}:${size}`;
}

/** User content as ACP blocks; images become a note when the agent cannot take them. */
export function userBlocks(message: UserMessage, images: boolean): AcpContentBlock[] {
	if (typeof message.content === "string") return message.content ? [{ type: "text", text: message.content }] : [];
	const blocks: AcpContentBlock[] = [];
	let dropped = 0;
	for (const block of message.content) {
		if (block.type === "text") {
			if (block.text) blocks.push({ type: "text", text: block.text });
		} else if (images) blocks.push({ type: "image", data: block.data, mimeType: block.mimeType });
		else dropped++;
	}
	if (dropped > 0) blocks.push({ type: "text", text: `[${dropped} image attachment(s) could not be forwarded to the ACP agent]` });
	return blocks;
}

function clip(text: string, max: number): string {
	return text.length > max ? `${text.slice(0, max)}… [${text.length - max} more characters]` : text;
}

function textOf(content: UserMessage["content"] | ToolResultMessage["content"]): string {
	if (typeof content === "string") return content;
	return content.map((block) => (block.type === "text" ? block.text : "[image]")).join("\n");
}

/** One message as transcript text; null for messages that carry nothing to show. */
function renderMessage(message: Message): string | null {
	switch (message.role) {
		case "user": {
			const text = textOf(message.content).trim();
			return text ? `[user]\n${text}` : null;
		}
		case "assistant": {
			const parts = message.content.flatMap((block) => {
				if (block.type === "text") return block.text.trim() ? [block.text.trim()] : [];
				if (block.type === "toolCall") return [`(called tool ${block.name} ${clip(JSON.stringify(block.arguments), 500)})`];
				return [];
			});
			return parts.length > 0 ? `[assistant]\n${parts.join("\n")}` : null;
		}
		case "toolResult": {
			const text = clip(textOf(message.content).trim(), MAX_TOOL_TEXT_CHARS);
			return `[tool result: ${message.toolName}${message.isError ? ", error" : ""}]\n${text || "(no output)"}`;
		}
		default:
			return null;
	}
}

/** Plain-text transcript of `messages`, newest last, capped from the oldest end. */
export function renderTranscript(messages: readonly Message[]): string {
	return transcriptOf(messages).text;
}

/** `renderTranscript`, with the messages the text shows: not those with nothing to show or cut by the cap. */
export function transcriptOf(messages: readonly Message[]): { text: string; shown: Message[] } {
	const parts = messages.flatMap((message) => {
		const text = renderMessage(message);
		return text === null ? [] : [{ message, text }];
	});
	let total = 0;
	let start = parts.length;
	while (start > 0 && total + (parts[start - 1]?.text.length ?? 0) <= MAX_TRANSCRIPT_CHARS) {
		total += (parts[start - 1]?.text.length ?? 0) + 2;
		start--;
	}
	const omitted = start > 0 ? `[${start} earlier message(s) omitted]\n\n` : "";
	const shown = parts.slice(start);
	return { text: omitted + shown.map((part) => part.text).join("\n\n"), shown: shown.map((part) => part.message) };
}

/** Wraps `body` in a tag so the agent can tell context Pi adds from what the user typed. */
export function tagged(tag: string, body: string, intro?: string): AcpContentBlock {
	return { type: "text", text: `<${tag}>\n${intro ? `${intro}\n\n` : ""}${body}\n</${tag}>` };
}
