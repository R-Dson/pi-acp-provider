import type { AssistantMessage, Message, UserMessage } from "@earendil-works/pi-ai";
import { isRecord } from "./acp.ts";
import type { SystemPromptMode } from "./config.ts";
import { ACP_PROVIDER_ID } from "./models.ts";
import { messageKey, tagged, transcriptOf, userBlocks, type AcpContentBlock } from "./transcript.ts";

export const MAX_STEER_REJECTIONS = 2;

export const LATE_STEER_NOTE = "Delivered late: the user sent the following earlier, and it did not reach you then.";

export interface SyncState {
	ownStamps: Set<number>;
	/** WARN: moves only to own replies found in Pi's history when a prompt is accepted, so it lags the newest reply; see `lastStreamed`. */
	anchor: number | undefined;
	/** WHY: compaction replaces the first message, so a changed key means the session's history is gone. */
	first: string | undefined;
	/** WHY: results of the session's own mirrored and Pi tool calls are not news to it, so they never go in a seed. */
	ours: Set<string>;
	delivered: Set<string>;
	/**
	 * WHY: kept apart from `anchor`, which moves only past the session's own replies, so a seed of foreign
	 * history is not sent again while no own reply follows it.
	 */
	seeded: Set<string>;
	/**
	 * WHY: unlike `anchor`, it covers the latest reply, so a fork that keeps an earlier one is still seen.
	 * `error` replies are left out because Pi omits a retried one from context; `aborted` ones it keeps.
	 */
	lastStreamed: number | undefined;
	/**
	 * WHY: a seed's transcript leaves out messages with nothing to show and ones its cap cuts; only the
	 * rendered ones were read, so only they may become `kept`.
	 */
	shown: Set<string>;
	/**
	 * WHY: a Pi `context` handler may add a message of any role for one request only, and nothing tells it
	 * from persisted history, so a message counts only once it is seen again. An edit that drops one before
	 * that goes unnoticed.
	 */
	kept: Set<string>;
	systemPrompt: string;
}

export const newSyncState = (systemPrompt: string): SyncState => ({
	ownStamps: new Set(),
	anchor: undefined,
	first: undefined,
	ours: new Set(),
	delivered: new Set(),
	seeded: new Set(),
	lastStreamed: undefined,
	shown: new Set(),
	kept: new Set(),
	systemPrompt,
});

export function tailStart(messages: readonly Message[]): number {
	for (let index = messages.length - 1; index >= 0; index--) if (messages[index]?.role === "assistant") return index + 1;
	return 0;
}

/**
 * WHY: an assistant message that ended in `error` answered nothing, so the users before it still wait for a
 * reply; the ones the session already got are filtered out as delivered. An `aborted` one counts as an
 * answer: the user stopped that request, so it must not run again.
 */
function requestStart(messages: readonly Message[]): number {
	for (let index = messages.length - 1; index >= 0; index--) {
		const candidate = messages[index];
		if (candidate?.role === "assistant" && candidate.stopReason !== "error") return index + 1;
	}
	return 0;
}

export function requestUsers(messages: readonly Message[]): UserMessage[] {
	return messages.slice(requestStart(messages)).filter((candidate): candidate is UserMessage => candidate.role === "user");
}

export function isOwnReply(state: SyncState, message: Message): boolean {
	return message.role === "assistant" && message.provider === ACP_PROVIDER_ID && state.ownStamps.has(message.timestamp);
}

function isOwn(state: SyncState, message: Message): boolean {
	if (isOwnReply(state, message) || state.seeded.has(messageKey(message))) return true;
	if (message.role === "toolResult") return state.ours.has(message.toolCallId);
	return message.role === "user" && state.delivered.has(messageKey(message));
}

export function diverged(state: SyncState, messages: readonly Message[]): boolean {
	if (state.first !== undefined && messages[0] && messageKey(messages[0]) !== state.first) return true;
	const hasReply = (timestamp: number): boolean =>
		messages.some((candidate) => candidate.role === "assistant" && candidate.timestamp === timestamp);
	if (state.anchor !== undefined && !hasReply(state.anchor)) return true;
	if (state.lastStreamed !== undefined && !hasReply(state.lastStreamed)) return true;
	const keys = new Set(messages.map(messageKey));
	return [...state.kept].some((key) => !keys.has(key));
}

export function keepConfirmed(state: SyncState, messages: readonly Message[]): void {
	for (const message of messages) {
		const key = messageKey(message);
		if (state.delivered.has(key) || state.shown.has(key)) state.kept.add(key);
	}
}

export function noteStreamed(state: SyncState | undefined, message: AssistantMessage): void {
	if (!state?.ownStamps.has(message.timestamp) || message.stopReason === "error") return;
	state.lastStreamed = message.timestamp;
}

export interface PendingSteer {
	message: UserMessage;
	key: string;
	rejections: number;
}

export const isDropped = (steer: PendingSteer): boolean => steer.rejections >= MAX_STEER_REJECTIONS;

/** WHY: kept apart from any `SyncState`, so they outlive a replaced or re-seeded session and reach its next prompt. */
export class PendingSteers {
	readonly #byOwner = new Map<string, PendingSteer[]>();

	has(owner: string, key: string): boolean {
		return this.#byOwner.get(owner)?.some((steer) => steer.key === key) === true;
	}

	hold(owner: string, steers: readonly UserMessage[]): void {
		const held = this.#byOwner.get(owner) ?? [];
		for (const message of steers) {
			const key = messageKey(message);
			if (!held.some((steer) => steer.key === key)) held.push({ message, key, rejections: 0 });
		}
		this.#byOwner.set(owner, held);
	}

	deliver(owner: string, state: SyncState | undefined, keys: readonly string[]): void {
		for (const key of keys) state?.delivered.add(key);
		const held = this.#byOwner.get(owner);
		if (held) this.#byOwner.set(owner, held.filter((steer) => !keys.includes(steer.key)));
	}

	reject(owner: string, keys: readonly string[]): PendingSteer[] {
		const dropped: PendingSteer[] = [];
		for (const steer of this.#byOwner.get(owner) ?? []) {
			if (!keys.includes(steer.key)) continue;
			steer.rejections++;
			if (steer.rejections === MAX_STEER_REJECTIONS) dropped.push(steer);
		}
		return dropped;
	}

	/** WARN: dropped steers stay listed, unsent, so that they stay out of history seeds as well. */
	recover(owner: string, messages: readonly Message[], state: SyncState): PendingSteer[] {
		const branch = new Set(messages.map(messageKey));
		const kept = (this.#byOwner.get(owner) ?? []).filter((steer) => branch.has(steer.key) && !state.delivered.has(steer.key));
		this.#byOwner.set(owner, kept);
		return kept;
	}

	clear(): void {
		this.#byOwner.clear();
	}
}

export interface PromptPart {
	blocks: AcpContentBlock[];
	/** WHY: fixed before an `onPayload` hook can change the blocks, so `partsIn` can see a rewrite. */
	sent: string[];
	record: () => void;
	omit: () => void;
	steer: string | undefined;
}

function promptPart(blocks: AcpContentBlock[], record: () => void, options: { omit?: () => void; steer?: string } = {}): PromptPart {
	return { blocks, sent: blocks.map((block) => JSON.stringify(block)), record, omit: options.omit ?? (() => {}), steer: options.steer };
}

/**
 * WHY: an `onPayload` hook may keep, rebuild or rewrite blocks. A kept block is the same object and must
 * be unchanged; a rebuilt one is matched by value only when no other block of the prompt has that value,
 * since it cannot tell which of two identical blocks it stands for.
 * WARN: a part a hook rewrote, or an ambiguous one, counts as not sent: wrongly sending again is
 * recoverable, wrongly recording delivery loses the content.
 */
export function partsIn(parts: readonly PromptPart[], payload: unknown): PromptPart[] {
	const prompt: unknown[] = isRecord(payload) && Array.isArray(payload.prompt) ? payload.prompt : [];
	const ours = new Set<unknown>(parts.flatMap((part) => part.blocks));
	const kept = new Set<unknown>();
	const rebuilt = new Map<string, number>();
	for (const block of prompt) {
		if (ours.has(block)) kept.add(block);
		else {
			const text = JSON.stringify(block);
			rebuilt.set(text, (rebuilt.get(text) ?? 0) + 1);
		}
	}
	const owners = new Map<string, number>();
	for (const text of parts.flatMap((part) => part.sent)) owners.set(text, (owners.get(text) ?? 0) + 1);
	return parts.filter((part) =>
		part.blocks.every((block, index) => {
			const text = part.sent[index]!;
			if (kept.has(block)) return JSON.stringify(block) === text;
			return owners.get(text) === 1 && rebuilt.get(text) === 1;
		}),
	);
}

export interface PromptInput {
	systemPromptMode: SystemPromptMode;
	systemPrompt: string;
	state: SyncState;
	/** WARN: system messages must be filtered out (`conversation`): they would move the first-message key and the tail. */
	messages: readonly Message[];
	/** WARN: must be what `steers.recover` returned for these `messages`, or off-branch or delivered steers are sent. */
	held: readonly PendingSteer[];
	images: boolean;
	replyTimestamp: number;
	steers: PendingSteers;
	owner: string;
}

export interface PromptBuild {
	parts: PromptPart[];
	blocks: AcpContentBlock[];
	/**
	 * WARN: run it only once the agent accepts the prompt, with the parts `partsIn` found in the sent
	 * payload; until then the build has changed no state, so a rejected prompt is retried in full.
	 */
	commit: (sent: readonly PromptPart[]) => void;
}

export function buildPromptBlocks(input: PromptInput): PromptBuild | null {
	const { systemPromptMode, systemPrompt, state, messages, held, images, replyTimestamp, steers, owner } = input;
	const tail = messages.slice(tailStart(messages));
	const users = requestUsers(messages);
	const parts: PromptPart[] = [];
	if (systemPromptMode !== "off" && systemPrompt !== "" && state.systemPrompt !== systemPrompt) {
		const intro = state.systemPrompt
			? "Pi, the host application you run in, updated its instructions. They replace the earlier ones:"
			: "Instructions from Pi, the host application you run in. Follow them together with your own:";
		parts.push(promptPart([tagged("system-instructions", systemPrompt, intro)], () => (state.systemPrompt = systemPrompt)));
	}
	const isHeld = (candidate: Message): boolean => held.some((steer) => steer.key === messageKey(candidate));
	const late = held.filter((steer) => !isDropped(steer));
	const requestUserSet = new Set<Message>(users);
	const history = messages.slice(0, messages.length - tail.length);
	const anchorIndex = state.anchor === undefined ? -1 : history.findLastIndex((candidate) => candidate.role === "assistant" && candidate.timestamp === state.anchor);
	// WHY: pending steers and the users of an unanswered request are requests the agent never got, so
	// they go as user blocks, never as history.
	const unseen = history
		.slice(anchorIndex + 1)
		.filter((candidate) => !isOwn(state, candidate) && !isHeld(candidate) && !requestUserSet.has(candidate));
	let heldHistory = history.length;
	const { text: transcript, shown } = transcriptOf(unseen);
	if (transcript) {
		const intro = "Earlier messages of this Pi conversation that you have not seen, oldest first. Treat them as history, not as new requests:";
		parts.push(
			promptPart(
				[tagged("conversation-history", transcript, intro)],
				() => {
					for (const seen of unseen) state.seeded.add(messageKey(seen));
					for (const seen of shown) state.shown.add(messageKey(seen));
				},
				// WHY: the anchor must stay before history the session never got, or later prompts skip it.
				{ omit: () => (heldHistory = history.indexOf(unseen[0]!)) },
			),
		);
	}
	if (late.length > 0) parts.push(promptPart([{ type: "text", text: LATE_STEER_NOTE }], () => {}));
	for (const steer of late) {
		const record = (): void => steers.deliver(owner, state, [steer.key]);
		parts.push(promptPart(userBlocks(steer.message, images), record, { steer: steer.key }));
	}
	const sendable = users.filter((user) => !isHeld(user));
	const fresh = sendable.filter((user) => !state.delivered.has(messageKey(user)));
	// WHY: with nothing new, the request is Pi retrying the last one; a recovered steer is new enough.
	const sentUsers = fresh.length > 0 || late.length > 0 ? fresh : sendable.slice(-1);
	if (sentUsers.length === 0 && late.length === 0) return null;
	for (const user of sentUsers) {
		// WHY: once the agent answers, a user the hook left out falls behind the reply and would reach the
		// next prompt only as history; held, it goes as a late user block instead.
		const omit = (): void => steers.hold(owner, [user]);
		parts.push(promptPart(userBlocks(user, images), () => state.delivered.add(messageKey(user)), { omit }));
	}
	const commit = (sent: readonly PromptPart[]): void => {
		for (const part of parts) (sent.includes(part) ? part.record : part.omit)();
		const newestOwn = history.slice(0, heldHistory).findLast((candidate) => isOwnReply(state, candidate));
		if (newestOwn) state.anchor = newestOwn.timestamp;
		state.first = messages[0] ? messageKey(messages[0]) : undefined;
		state.ownStamps.add(replyTimestamp);
	};
	return { parts, blocks: parts.flatMap((part) => part.blocks), commit };
}
