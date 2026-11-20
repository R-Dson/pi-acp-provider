import test from "node:test";
import assert from "node:assert/strict";
import type { AssistantMessage, Message, StopReason, UserMessage } from "@earendil-works/pi-ai";
import { zeroUsage } from "../src/message-writer.ts";
import { ACP_PROVIDER_ID } from "../src/models.ts";
import {
	LATE_STEER_NOTE,
	PendingSteers,
	buildPromptBlocks,
	diverged,
	keepConfirmed,
	newSyncState,
	noteStreamed,
	partsIn,
	type PromptBuild,
	type PromptInput,
	type SyncState,
} from "../src/session-sync.ts";
import { messageKey } from "../src/transcript.ts";

let clock = 1_000;
const user = (text: string): UserMessage => ({ role: "user", content: [{ type: "text", text }], timestamp: ++clock });
const reply = (text: string, options: { provider?: string; stopReason?: StopReason } = {}): AssistantMessage => ({
	role: "assistant",
	content: [{ type: "text", text }],
	api: "acp",
	provider: options.provider ?? ACP_PROVIDER_ID,
	model: "fake",
	usage: zeroUsage(),
	stopReason: options.stopReason ?? "stop",
	timestamp: ++clock,
});

const OWNER = "fake\u0000conversation";

function build(state: SyncState, messages: readonly Message[], overrides: Partial<PromptInput> = {}): PromptBuild {
	const steers = overrides.steers ?? new PendingSteers();
	const result = buildPromptBlocks({
		systemPromptMode: "prompt",
		systemPrompt: "",
		state,
		messages,
		held: steers.recover(OWNER, messages, state),
		images: false,
		replyTimestamp: ++clock,
		steers,
		owner: OWNER,
		...overrides,
	});
	assert.ok(result, "expected a prompt");
	return result;
}

const texts = (prompt: PromptBuild): string[] => prompt.blocks.flatMap((block) => (block.type === "text" ? [block.text] : []));
const acceptAll = (prompt: PromptBuild): void => prompt.commit(prompt.parts);
function streamed(state: SyncState, prompt: PromptBuild, answer: AssistantMessage): void {
	state.ownStamps.add(answer.timestamp);
	acceptAll(prompt);
	noteStreamed(state, answer);
}

test("a foreign reply in history is seeded once, and the request's users go as user blocks", () => {
	const state = newSyncState("");
	const u1 = user("first");
	const foreign = reply("from another model", { provider: "other" });
	const u2 = user("second");
	const first = build(state, [u1, foreign, u2]);
	const [seed, ...rest] = texts(first);
	assert.match(seed!, /^<conversation-history>\n/);
	assert.match(seed!, /\[user\]\nfirst/);
	assert.match(seed!, /\[assistant\]\nfrom another model/);
	assert.doesNotMatch(seed!, /second/);
	assert.deepEqual(rest, ["second"]);

	acceptAll(first);
	assert.ok(state.seeded.has(messageKey(foreign)));
	assert.ok(state.shown.has(messageKey(foreign)));
	assert.ok(state.delivered.has(messageKey(u2)));

	const u3 = user("third");
	const again = build(state, [u1, foreign, u2, reply("own reply", { provider: "other" }), u3]);
	assert.equal(texts(again).filter((text) => text.startsWith("<conversation-history>")).length, 1);
	assert.doesNotMatch(texts(again)[0]!, /from another model/);
});

test("a seed the payload left out is sent again with the next prompt", () => {
	const state = newSyncState("");
	const messages = [user("old"), reply("foreign", { provider: "other" }), user("new")];
	const first = build(state, messages);
	const kept = partsIn(first.parts, { prompt: first.blocks.slice(1) });
	first.commit(kept);
	assert.equal(state.seeded.size, 0);
	assert.match(texts(build(state, messages))[0]!, /^<conversation-history>/);
});

test("an omitted seed keeps the anchor before the history it carried", () => {
	const setUp = (): { state: SyncState; messages: Message[]; own: AssistantMessage } => {
		const state = newSyncState("");
		const u1 = user("one");
		state.delivered.add(messageKey(u1));
		const own = reply("own reply");
		state.ownStamps.add(own.timestamp);
		return { state, messages: [u1, reply("foreign", { provider: "other" }), own, user("two")], own };
	};

	const accepted = setUp();
	acceptAll(build(accepted.state, accepted.messages));
	assert.equal(accepted.state.anchor, accepted.own.timestamp, "with the seed sent, the anchor moves to the own reply");

	const omitted = setUp();
	const prompt = build(omitted.state, omitted.messages);
	assert.match(texts(prompt)[0]!, /^<conversation-history>[\s\S]*foreign/);
	prompt.commit(prompt.parts.slice(1));
	assert.equal(omitted.state.anchor, undefined);
	assert.match(texts(build(omitted.state, omitted.messages))[0]!, /^<conversation-history>[\s\S]*foreign/);
});

test("own replies move the anchor, so later prompts carry only what follows them", () => {
	const state = newSyncState("");
	const u1 = user("one");
	const first = build(state, [u1]);
	const a1 = reply("answer one");
	streamed(state, first, a1);
	const u2 = user("two");
	const second = build(state, [u1, a1, u2]);
	assert.deepEqual(texts(second), ["two"]);
	acceptAll(second);
	assert.equal(state.anchor, a1.timestamp);
	assert.equal(state.first, messageKey(u1));
});

test("with nothing new, a retried request sends its last user again", () => {
	const state = newSyncState("");
	const u1 = user("one");
	const failed = reply("failed", { stopReason: "error" });
	streamed(state, build(state, [u1]), failed);
	assert.deepEqual(texts(build(state, [u1, failed])), ["one"]);
});

test("noteStreamed records own replies Pi keeps, and diverged sees a fork that keeps an earlier one", () => {
	const state = newSyncState("");
	const u1 = user("one");
	const a1 = reply("answer one");
	streamed(state, build(state, [u1]), a1);
	assert.equal(state.lastStreamed, a1.timestamp);

	const u2 = user("two");
	const a2 = reply("answer two");
	streamed(state, build(state, [u1, a1, u2]), a2);
	assert.equal(state.lastStreamed, a2.timestamp);
	assert.equal(state.anchor, a1.timestamp);

	const u3 = user("three");
	assert.equal(diverged(state, [u1, a1, u2, a2, u3]), false);
	assert.equal(diverged(state, [u1, a1, u3]), true, "the anchor A1 is still there, but the newest reply A2 is gone");
});

test("noteStreamed ignores error replies and replies the session did not stream", () => {
	const state = newSyncState("");
	const failed = reply("failed", { stopReason: "error" });
	state.ownStamps.add(failed.timestamp);
	noteStreamed(state, failed);
	noteStreamed(state, reply("foreign"));
	noteStreamed(undefined, reply("no session"));
	assert.equal(state.lastStreamed, undefined);
	const aborted = reply("stopped", { stopReason: "aborted" });
	state.ownStamps.add(aborted.timestamp);
	noteStreamed(state, aborted);
	assert.equal(state.lastStreamed, aborted.timestamp);
});

test("diverged sees a replaced first message", () => {
	const state = newSyncState("");
	const u1 = user("one");
	acceptAll(build(state, [u1]));
	assert.equal(diverged(state, [u1, user("two")]), false);
	assert.equal(diverged(state, [user("summary"), user("two")]), true);
});

test("kept holds only delivered messages a later request still held", () => {
	const state = newSyncState("");
	const added = user("added for one request");
	const u1 = user("one");
	acceptAll(build(state, [added, u1]));
	const a1 = reply("answer");
	const u2 = user("two");
	const later = [u1, a1, u2];
	state.first = messageKey(u1);
	keepConfirmed(state, later);
	assert.ok(state.kept.has(messageKey(u1)));
	assert.ok(!state.kept.has(messageKey(added)));
	assert.ok(!state.kept.has(messageKey(a1)), "a message never delivered or shown is not kept");
	assert.equal(diverged(state, later), false, "a message seen once and never again is not history");
	assert.equal(diverged(state, [a1, u2]), true, "a kept user is gone");
});

test("diverged sees a kept user removed while the first message stays", () => {
	const state = newSyncState("");
	const u0 = user("zero");
	const a0 = reply("answer zero", { provider: "other" });
	const u1 = user("one");
	const u2 = user("two");
	state.first = messageKey(u0);
	state.delivered.add(messageKey(u1));
	keepConfirmed(state, [u0, a0, u1]);
	assert.ok(state.kept.has(messageKey(u1)));
	assert.equal(diverged(state, [u0, a0, u1, u2]), false);
	assert.equal(diverged(state, [u0, a0, u2]), true);
});

test("a pending steer goes as a late user block, never inside the seed", () => {
	const state = newSyncState("");
	const steers = new PendingSteers();
	const steer = user("STEER");
	steers.hold(OWNER, [steer]);
	const u1 = user("one");
	const messages = [u1, steer, reply("foreign", { provider: "other" }), user("next")];
	const prompt = build(state, messages, { steers });
	const [seed, note, sentSteer, next] = texts(prompt);
	assert.match(seed!, /^<conversation-history>/);
	assert.doesNotMatch(seed!, /STEER/);
	assert.equal(note, LATE_STEER_NOTE);
	assert.equal(sentSteer, "STEER");
	assert.equal(next, "next");

	assert.ok(steers.has(OWNER, messageKey(steer)), "held until the prompt is accepted");
	acceptAll(prompt);
	assert.ok(!steers.has(OWNER, messageKey(steer)));
	assert.ok(state.delivered.has(messageKey(steer)));
});

test("pending steers off the active branch or already delivered are forgotten", () => {
	const state = newSyncState("");
	const steers = new PendingSteers();
	const offBranch = user("gone");
	const delivered = user("delivered");
	const kept = user("kept");
	steers.hold(OWNER, [offBranch, delivered, kept]);
	state.delivered.add(messageKey(delivered));
	assert.deepEqual(
		steers.recover(OWNER, [delivered, kept], state).map((steer) => steer.message),
		[kept],
	);
	assert.ok(!steers.has(OWNER, messageKey(offBranch)));
});

test("pending steers are kept per owner", () => {
	const steers = new PendingSteers();
	const steer = user("STEER");
	steers.hold(OWNER, [steer]);
	assert.ok(!steers.has("fake\u0000other", messageKey(steer)));
	steers.hold(OWNER, [steer]);
	assert.equal(steers.recover(OWNER, [steer], newSyncState("")).length, 1, "holding twice keeps one entry");
});

test("a steer rejected twice is dropped: neither sent nor seeded", () => {
	const state = newSyncState("");
	const steers = new PendingSteers();
	const steer = user("STEER");
	steers.hold(OWNER, [steer]);
	assert.deepEqual(steers.reject(OWNER, [messageKey(steer)]), []);
	assert.deepEqual(
		steers.reject(OWNER, [messageKey(steer)]).map((dropped) => dropped.message),
		[steer],
	);
	const prompt = build(state, [user("one"), steer, reply("foreign", { provider: "other" }), user("next")], { steers });
	assert.ok(texts(prompt).every((text) => !text.includes("STEER")));
	assert.ok(!texts(prompt).includes(LATE_STEER_NOTE));
});

test("there is nothing to send when the only user is a dropped steer", () => {
	const state = newSyncState("");
	const steers = new PendingSteers();
	const steer = user("STEER");
	steers.hold(OWNER, [steer]);
	steers.reject(OWNER, [messageKey(steer)]);
	steers.reject(OWNER, [messageKey(steer)]);
	const result = buildPromptBlocks({
		systemPromptMode: "prompt",
		systemPrompt: "",
		state,
		messages: [steer],
		held: steers.recover(OWNER, [steer], state),
		images: false,
		replyTimestamp: ++clock,
		steers,
		owner: OWNER,
	});
	assert.equal(result, null);
});

test("a user the payload left out is held as a pending steer", () => {
	const state = newSyncState("");
	const steers = new PendingSteers();
	const u1 = user("one");
	const u2 = user("two");
	const prompt = build(state, [u1, u2], { steers });
	prompt.commit(prompt.parts.slice(0, 1));
	assert.ok(state.delivered.has(messageKey(u1)));
	assert.ok(!state.delivered.has(messageKey(u2)));
	assert.ok(steers.has(OWNER, messageKey(u2)));
});

test("nothing is recorded until the prompt is committed", () => {
	const state = newSyncState("");
	const before = structuredClone(state);
	build(state, [user("one"), reply("foreign", { provider: "other" }), user("two")], { systemPrompt: "Be brief." });
	assert.deepEqual(state, before);
});

test("the system prompt goes once, again when it changes, and never when off or already in the session", () => {
	const state = newSyncState("");
	const u1 = user("one");
	const first = build(state, [u1], { systemPrompt: "Be brief." });
	assert.match(texts(first)[0]!, /^<system-instructions>\nInstructions from Pi, [^\n]*\n\nBe brief\.\n<\/system-instructions>$/);
	acceptAll(first);
	assert.equal(state.systemPrompt, "Be brief.");

	assert.deepEqual(texts(build(state, [u1, user("two")], { systemPrompt: "Be brief." })), ["two"]);
	assert.match(texts(build(state, [u1, user("two")], { systemPrompt: "Be terse." }))[0]!, /^<system-instructions>\nPi, [^\n]* updated its instructions/);

	assert.deepEqual(texts(build(newSyncState(""), [u1], { systemPrompt: "Be brief.", systemPromptMode: "off" })), ["one"]);
	assert.deepEqual(texts(build(newSyncState("Be brief."), [u1], { systemPrompt: "Be brief.", systemPromptMode: "session" })), ["one"]);
	assert.deepEqual(texts(build(newSyncState("Be brief."), [u1], { systemPrompt: "" })), ["one"]);
});

test("system instructions the payload left out are sent again", () => {
	const state = newSyncState("");
	const u1 = user("one");
	const first = build(state, [u1], { systemPrompt: "Be brief." });
	first.commit(first.parts.slice(1));
	assert.equal(state.systemPrompt, "");
	assert.match(texts(build(state, [u1, user("two")], { systemPrompt: "Be brief." }))[0]!, /^<system-instructions>/);
});

test("partsIn counts a rewritten block as not sent, and a rebuilt identical one as sent", () => {
	const state = newSyncState("");
	const prompt = build(state, [user("one"), user("two")]);
	const rebuilt = prompt.blocks.map((block) => ({ ...block }));
	assert.deepEqual(partsIn(prompt.parts, { prompt: rebuilt }), prompt.parts);
	const rewritten = [prompt.blocks[0], { type: "text", text: "changed" }];
	assert.deepEqual(partsIn(prompt.parts, { prompt: rewritten }), prompt.parts.slice(0, 1));
});
