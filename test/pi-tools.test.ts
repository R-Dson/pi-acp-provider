import test from "node:test";
import assert from "node:assert/strict";
import { getEventListeners } from "node:events";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { fileURLToPath } from "node:url";
import { normalizeContext } from "@earendil-works/pi-ai/utils/transcript";
import type { AssistantMessage, AssistantMessageEvent, AssistantMessageEventStream, Message, Model, Api, Tool } from "@earendil-works/pi-ai";
import { Type } from "typebox";
import { createAcpProvider, piToolRef, type AcpProviderHandle, type AcpProviderOptions } from "../src/provider.ts";
import type { AcpAgentConfig } from "../src/config.ts";
import { toToolResult } from "../src/native-tools.ts";
import { flattenRootCombinators, toMcpTool } from "../src/mcp-server.ts";

const fakePath = fileURLToPath(new URL("./fake-agent.mjs", import.meta.url));

function agent(overrides: Partial<AcpAgentConfig> = {}): AcpAgentConfig {
	return {
		id: "fake",
		name: "Fake agent",
		command: process.execPath,
		args: [fakePath],
		env: {},
		authMethod: undefined,
		piTools: "both",
		mcpTransport: "stdio",
		systemPrompt: "prompt",
		sessionMeta: undefined,
		ownToolsOffMeta: undefined,
		...overrides,
	};
}

const handles: AcpProviderHandle[] = [];
test.afterEach(() => {
	for (const handle of handles.splice(0)) handle.dispose();
});

function makeProvider(agents: AcpAgentConfig[], extra: Omit<AcpProviderOptions, "config"> = {}): AcpProviderHandle {
	const handle = createAcpProvider({
		config: { agents, autoApproveTools: true, handshakeTimeoutMs: 8000, debug: false, nativeTools: true },
		...extra,
	});
	handles.push(handle);
	return handle;
}

const echoTool: Tool = {
	name: "echo",
	description: "Echoes text",
	parameters: Type.Object({ text: Type.String() }),
};

let clock = Date.now();
const user = (text: string): Message => ({ role: "user", content: [{ type: "text", text }], timestamp: ++clock });

async function collect(stream: AssistantMessageEventStream): Promise<AssistantMessageEvent[]> {
	const events: AssistantMessageEvent[] = [];
	for await (const event of stream) events.push(event);
	return events;
}

function model(handle: AcpProviderHandle): Model<Api> {
	const first = handle.provider.getModels()[0];
	assert.ok(first);
	return first;
}

const textOf = (message: AssistantMessage) => message.content.flatMap((block) => (block.type === "text" ? [block.text] : [])).join("");

/** One provider call with Pi's `tools` (default: echo); returns the final event, which must be `done`. */
async function step(handle: AcpProviderHandle, messages: Message[], systemPrompt?: string, options: Record<string, unknown> & { tools?: Tool[] } = {}) {
	const { tools = [echoTool], ...streamOptions } = options;
	const context = normalizeContext({ ...(systemPrompt ? { systemPrompt } : {}), tools, messages });
	const last = (await collect(handle.provider.streamSimple(model(handle), context, streamOptions))).at(-1);
	assert.ok(last?.type === "done", `expected done, got ${JSON.stringify(last)}`);
	return last.message;
}

/** Answers every tool call of `message` the way Pi would: echo runs, acp_tool waits on the agent. */
async function answerTools(handle: AcpProviderHandle, message: AssistantMessage): Promise<Message[]> {
	const results: Message[] = [];
	for (const block of message.content) {
		if (block.type !== "toolCall") continue;
		const content =
			block.name === "echo"
				? [{ type: "text" as const, text: `echoed ${(block.arguments as { text: string }).text}` }]
				: toToolResult(await handle.awaitTool(block.id, undefined, () => {})).content;
		results.push({ role: "toolResult", toolCallId: block.id, toolName: block.name, content, isError: false, timestamp: ++clock } as Message);
	}
	return results;
}

for (const transport of ["stdio", "http"] as const) {
	test(`a Pi tool called over ${transport} MCP runs as a Pi tool call and its result reaches the agent`, async () => {
		const handle = makeProvider([agent({ mcpTransport: transport, env: transport === "http" ? { FAKE_MCP_HTTP: "1" } : {} })]);
		const messages: Message[] = [user('MCPCALL:echo:{"text":"hi"}')];
		const first = await step(handle, messages);
		assert.equal(first.stopReason, "toolUse");
		const calls = first.content.filter((block) => block.type === "toolCall");
		assert.equal(calls.length, 1, "the agent's own report of the MCP call is not mirrored as acp_tool");
		assert.equal(calls[0]?.type === "toolCall" && calls[0].name, "echo");
		assert.deepEqual(calls[0]?.type === "toolCall" && calls[0].arguments, { text: "hi" });
		messages.push(first, ...(await answerTools(handle, first)));
		const last = await step(handle, messages);
		assert.equal(last.stopReason, "stop");
		assert.match(textOf(last), /TOOLS<<<echo>>> RESULT<<<echoed hi>>>/);
	});
}

const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

/**
 * Runs `body` with TMPDIR at `dir`, where Pi's MCP server makes its stdio socket directory.
 * WARN: POSIX only; `os.tmpdir()` ignores TMPDIR on Windows, and the MCP server uses a named pipe there.
 */
async function withTmpdir<T>(dir: string, body: () => Promise<T>): Promise<T> {
	const saved = process.env.TMPDIR;
	process.env.TMPDIR = dir;
	try {
		return await body();
	} finally {
		if (saved === undefined) delete process.env.TMPDIR;
		else process.env.TMPDIR = saved;
	}
}

for (const [knob, label, call] of [
	// The cancellation must have reached the provider, which rejects the call; not a call left pending.
	["FAKE_MCP_CANCEL", "cancelled its MCP call", /CALL<<<The tool call was cancelled\.>>>/],
	["FAKE_MCP_GIVEUP", "ended the turn without waiting", /CALL<<<error:gave up>>>/],
] as const) {
	test(`a Pi tool result after the agent ${label} ends the turn with the agent's text`, async () => {
		const handle = makeProvider([agent({ env: { [knob]: "1" } })]);
		const messages: Message[] = [user('MCPCALL:echo:{"text":"hi"}')];
		const first = await step(handle, messages);
		assert.equal(first.stopReason, "toolUse");
		// Pi is still running the tool when the agent stops waiting for it.
		await sleep(300);
		messages.push(first, ...(await answerTools(handle, first)));
		const last = await step(handle, messages);
		assert.equal(last.stopReason, "stop");
		assert.match(textOf(last), /AGENT-WENT-ON/);
		assert.match(textOf(last), call);
	});
}

/** Streams `messages` and aborts on the first thinking delta; returns the aborted message. */
async function abortedStep(handle: AcpProviderHandle, messages: Message[]): Promise<AssistantMessage> {
	const controller = new AbortController();
	const context = normalizeContext({ tools: [echoTool], messages });
	let last: AssistantMessageEvent | undefined;
	for await (const event of handle.provider.streamSimple(model(handle), context, { signal: controller.signal })) {
		last = event;
		if (event.type === "thinking_delta") controller.abort();
	}
	assert.ok(last?.type === "error" && last.reason === "aborted", `expected aborted, got ${JSON.stringify(last)}`);
	return last.error;
}

test("aborting Pi's run while it runs a Pi tool call aborts the ACP turn and answers the MCP request with an error", async () => {
	const handle = makeProvider([agent()]);
	const controller = new AbortController();
	const messages: Message[] = [user('MCPCALL:echo:{"text":"hi"}')];
	const first = await step(handle, messages, undefined, { signal: controller.signal });
	assert.equal(first.stopReason, "toolUse");
	const call = first.content.find((block) => block.type === "toolCall");
	assert.ok(call?.type === "toolCall");
	controller.abort();
	const aborted = { role: "toolResult", toolCallId: call.id, toolName: call.name, content: [{ type: "text", text: "Operation aborted" }], isError: true, timestamp: ++clock } as Message;
	messages.push(first, aborted, user("ECHO CALLREPORT after"));
	const next = textOf(await step(handle, messages));
	assert.match(next, /^ECHO\[sess-1\]<<<ECHO CALLREPORT after>>> CANCELLEDCALL<<<The turn ended before Pi ran this tool\.>>>$/);
});

test("a turn spanning several Pi messages leaves no abort listener on Pi's signal once it ends", async () => {
	const handle = makeProvider([agent()]);
	const controller = new AbortController();
	const messages: Message[] = [user('MCPCALL:echo:{"text":"hi"}')];
	const first = await step(handle, messages, undefined, { signal: controller.signal });
	assert.equal(first.stopReason, "toolUse");
	messages.push(first, ...(await answerTools(handle, first)));
	assert.equal((await step(handle, messages, undefined, { signal: controller.signal })).stopReason, "stop");
	assert.equal(getEventListeners(controller.signal, "abort").length, 0);
});

test("a cancelled prompt's late updates, permission requests and MCP calls stay out of the next turn", async () => {
	const asked: string[] = [];
	const handle = makeProvider([agent({ env: { FAKE_CANCEL_LAG: "200" } })], {
		permissionPolicy: async (request) => {
			asked.push(request.title);
			return { outcome: "selected", optionId: "allow" };
		},
	});
	const messages: Message[] = [user("SLOW work")];
	messages.push(await abortedStep(handle, messages), user("ECHO next"));
	const next = await step(handle, messages);
	assert.equal(next.stopReason, "stop");
	const text = textOf(next);
	assert.doesNotMatch(text, /LATE-CHUNK/);
	assert.match(text, /LAG<<<permission=cancelled;mcp=rejected>>>/);
	assert.deepEqual(asked, []);
	assert.deepEqual((await handle.commands("fake")).map((command) => command.name), ["context", "session"]);
});

for (const transport of ["stdio", "http"] as const) {
	test(`a session whose cancel never drains is replaced and its later ${transport} MCP, permission and update traffic is refused`, async () => {
		const asked: string[] = [];
		const handle = makeProvider(
			[agent({ mcpTransport: transport, env: { FAKE_IGNORE_CANCEL: "1", ...(transport === "http" ? { FAKE_MCP_HTTP: "1" } : {}) } })],
			{
				cancelGraceMs: 500,
				permissionPolicy: async (request) => {
					asked.push(request.title);
					return { outcome: "selected", optionId: "allow" };
				},
			},
		);
		const messages: Message[] = [user("SLOW work")];
		messages.push(await abortedStep(handle, messages), user("ECHO next"));
		const next = await step(handle, messages);
		assert.equal(next.stopReason, "stop");
		const text = textOf(next);
		assert.match(text, /^ECHO\[sess-2\]/, "the next prompt runs on a new session");
		assert.match(text, /<conversation-history>[\s\S]*SLOW work[\s\S]*<\/conversation-history>/, "the new session is re-seeded");
		assert.doesNotMatch(text, /STALE-CHUNK/);
		assert.match(text, /STALE<<<permission=cancelled;mcp=rejected>>>/);
		assert.deepEqual(asked, []);
	});
}

/** A provider whose permission policy records each request's title and allows it. */
function recordingProvider(
	env: Record<string, string>,
	asked: string[],
	overrides: Partial<AcpAgentConfig> = {},
	extra: Omit<AcpProviderOptions, "config" | "permissionPolicy"> = {},
) {
	return makeProvider([agent({ env, ...overrides })], {
		...extra,
		permissionPolicy: async (request) => {
			asked.push(request.title);
			return { outcome: "selected", optionId: "allow" };
		},
	});
}

/** Runs one MCPCALL turn that asks permission for its echo call; returns the final reply text. */
async function askedEchoTurn(handle: AcpProviderHandle, tools?: Tool[]): Promise<string> {
	const messages: Message[] = [user('MCPCALL:echo:{"text":"hi"} ASKPERM')];
	const first = await step(handle, messages, undefined, tools ? { tools } : {});
	assert.equal(first.stopReason, "toolUse");
	messages.push(first, ...(await answerTools(handle, first)));
	return textOf(await step(handle, messages, undefined, tools ? { tools } : {}));
}

const shoutTool: Tool = { name: "shout", description: "Shouts text", parameters: Type.Object({ text: Type.String() }) };

test("a permission request naming one served Pi tool for a call reported as another goes to the policy", async () => {
	const asked: string[] = [];
	const handle = recordingProvider({ FAKE_MCP_NAME: "mcp__pi__echo", FAKE_PERMISSION_NAME: "mcp__pi__shout" }, asked);
	const text = await askedEchoTurn(handle, [echoTool, shoutTool]);
	assert.match(text, /PERM<<<allow>>> TOOLS<<<echo,shout>>> RESULT<<<echoed hi>>>/);
	assert.deepEqual(asked, ["mcp__pi__echo"]);
});

test("a structurally named Pi tool call shown as a native row is approved without the policy", async () => {
	const asked: string[] = [];
	const handle = recordingProvider({ FAKE_MCP_TITLE: "Run something", FAKE_PERMISSION_NAME: "mcp__pi__echo" }, asked, {}, { nativeTools: () => true });
	const messages: Message[] = [user('MCPCALL:echo:{"text":"hi"} ASKPERM')];
	const names: string[] = [];
	for (let index = 0; index < 5; index++) {
		const message = await step(handle, messages);
		names.push(...message.content.flatMap((block) => (block.type === "toolCall" ? [block.name] : [])));
		messages.push(message);
		if (message.stopReason !== "toolUse") {
			assert.match(textOf(message), /PERM<<<allow>>> TOOLS<<<echo>>> RESULT<<<echoed hi>>>/);
			assert.deepEqual(names, ["acp_tool", "echo"], "the ordinary title made the report a native row");
			assert.deepEqual(asked, []);
			return;
		}
		messages.push(...(await answerTools(handle, message)));
	}
	assert.fail("the turn did not end");
});

for (const [label, env] of [
	["the permission request's structured name", { FAKE_PERMISSION_NAME: "mcp__pi__echo" }],
	["the structured name the tool call was reported with", { FAKE_MCP_NAME: "mcp__pi__echo", FAKE_PERMISSION_TITLE: "Echo" }],
	["both structured fields of the request, agreeing", { FAKE_PERMISSION_NAME: "mcp__pi__echo", FAKE_PERMISSION_META_NAME: "mcp__pi__echo" }],
] as const) {
	test(`a live Pi tool call named by ${label} is approved without the policy`, async () => {
		const asked: string[] = [];
		const text = await askedEchoTurn(recordingProvider(env, asked));
		assert.match(text, /PERM<<<allow>>> TOOLS<<<echo>>> RESULT<<<echoed hi>>>/);
		assert.deepEqual(asked, []);
	});
}

for (const [label, env, title] of [
	["titled only", {}, "mcp__pi__echo"],
	["named for a tool Pi does not serve", { FAKE_PERMISSION_NAME: "mcp__pi__rm" }, "mcp__pi__echo"],
	["named unlike its tool call", { FAKE_MCP_NAME: "mcp__pi__echo", FAKE_PERMISSION_NAME: "Bash" }, "mcp__pi__echo"],
	["whose ACP name and Claude tool name disagree", { FAKE_PERMISSION_NAME: "mcp__pi__echo", FAKE_PERMISSION_META_NAME: "Bash" }, "mcp__pi__echo"],
	["whose call was reported with disagreeing ACP and Claude tool names", { FAKE_MCP_NAME: "mcp__pi__echo", FAKE_MCP_META_NAME: "Bash" }, "mcp__pi__echo"],
] as const) {
	test(`a permission request for a Pi tool call ${label} goes to the policy`, async () => {
		const asked: string[] = [];
		const text = await askedEchoTurn(recordingProvider(env, asked));
		assert.match(text, /PERM<<<allow>>> TOOLS<<<echo>>> RESULT<<<echoed hi>>>/);
		assert.deepEqual(asked, [title]);
	});
}

test("a pending call whose waiting row Pi's handoff of another call ended is still approved by its structured name", async () => {
	const asked: string[] = [];
	const handle = recordingProvider({ FAKE_PERMISSION_NAME: "mcp__pi__echo" }, asked, {}, { nativeTools: () => true });
	const messages: Message[] = [user('MCPCALL:echo:{"text":"hi"} OVERLAP')];
	const rows: string[] = [];
	for (let index = 0; index < 5; index++) {
		const message = await step(handle, messages);
		messages.push(message);
		if (message.stopReason !== "toolUse") {
			assert.match(textOf(message), /OVERLAPPERM<<<allow>>> TOOLS<<<echo>>> RESULT<<<echoed hi>>>/);
			assert.match(rows[0] ?? "", /Handed to Pi as echo/, "the handoff ended r1's row with a synthetic status");
			assert.deepEqual(asked, []);
			return;
		}
		const results = await answerTools(handle, message);
		for (const result of results) if (result.role === "toolResult" && result.toolName === "acp_tool") rows.push(JSON.stringify(result.content));
		messages.push(...results);
	}
	assert.fail("the turn did not end");
});

test("a Pi-looking permission request without a reported, pending tool call goes to the policy", async () => {
	const asked: string[] = [];
	const handle = recordingProvider({ FAKE_PERMISSION_TITLE: "Read pi/notes.md", FAKE_PERMISSION_NAME: "mcp__pi__echo" }, asked, { piTools: "both" });
	const message = await step(handle, [user("PERMISSION")]);
	assert.match(textOf(message), /\[approved=allow\]/);
	assert.deepEqual(asked, ["Read pi/notes.md"]);
});

test("a Pi-named permission request on a session without Pi's MCP server goes to the policy", { skip: process.platform === "win32" }, async () => {
	const asked: string[] = [];
	const handle = recordingProvider({ FAKE_PERMISSION_NAME: "mcp__pi__echo" }, asked);
	const message = await withTmpdir(join(tmpdir(), `missing-${process.pid}`), () => step(handle, [user('MCPCALL:echo:{"text":"hi"} ASKPERM')]));
	assert.match(textOf(message), /NO-MCP-SERVER/);
	assert.deepEqual(asked, ["mcp__pi__echo"]);
});

test("an unrecognized report of a Pi tool call that Pi mirrors does not block the call", async () => {
	const handle = makeProvider([agent({ env: { FAKE_MCP_TITLE: "Run something" } })], { nativeTools: () => true });
	const messages: Message[] = [user('MCPCALL:echo:{"text":"hi"}')];
	const names: string[] = [];
	for (let index = 0; index < 5; index++) {
		const message = await step(handle, messages);
		names.push(...message.content.flatMap((block) => (block.type === "toolCall" ? [block.name] : [])));
		messages.push(message);
		if (message.stopReason !== "toolUse") {
			assert.match(textOf(message), /RESULT<<<echoed hi>>>/);
			assert.deepEqual(names, ["acp_tool", "echo"]);
			return;
		}
		messages.push(...(await answerTools(handle, message)));
	}
	assert.fail("the turn did not end");
});

test("pi tools off: no MCP server is offered", async () => {
	const handle = makeProvider([agent({ piTools: "off" })]);
	const message = await step(handle, [user("SESSIONINFO")]);
	assert.deepEqual(JSON.parse(textOf(message)).mcpServers, []);
});

test("pi mode merges the agent's tool switch into the session meta; both mode does not", async () => {
	const off = { vendor: { tools: [] } };
	const piOnly = makeProvider([agent({ piTools: "pi", ownToolsOffMeta: off, sessionMeta: { vendor: { allowed: ["mcp__pi"] } } })]);
	const piInfo = JSON.parse(textOf(await step(piOnly, [user("SESSIONINFO")])));
	assert.deepEqual(piInfo._meta, { vendor: { allowed: ["mcp__pi"], tools: [] } });
	const both = makeProvider([agent({ id: "both", ownToolsOffMeta: off })]);
	const bothInfo = JSON.parse(textOf(await step(both, [user("SESSIONINFO")])));
	assert.equal(bothInfo._meta, undefined);
});

const missingTmpdir = () => join(tmpdir(), `pi-acp-missing-${process.pid}`);

test("a Pi-tools-only agent ends the turn when Pi's MCP server cannot start, and the next turn retries it", { skip: process.platform === "win32" }, async () => {
	const off = { vendor: { tools: [] } };
	const handle = makeProvider([agent({ piTools: "pi", ownToolsOffMeta: off })]);
	const context = normalizeContext({ tools: [echoTool], messages: [user("SESSIONINFO")] });
	const failed = await withTmpdir(missingTmpdir(), () => collect(handle.provider.streamSimple(model(handle), context)));
	const last = failed.at(-1);
	assert.ok(last?.type === "error" && last.reason === "error", `expected an error, got ${JSON.stringify(last)}`);
	assert.match(last.error.errorMessage ?? "", /Pi's tool server \(MCP server "pi"\) failed to start/);
	assert.match(last.error.errorMessage ?? "", /instead of using its built-in tools/);
	const info = JSON.parse(textOf(await step(handle, [user("SESSIONINFO")])));
	assert.equal(info.mcpServers[0]?.name, "pi");
	assert.deepEqual(info._meta, off);
});

test("a used session opened while Pi's MCP server was down is replaced, with Pi's tools and a seed, once it is up", { skip: process.platform === "win32" }, async () => {
	const handle = makeProvider([agent()]);
	const messages: Message[] = [user("SESSIONINFO")];
	const down = await withTmpdir(missingTmpdir(), () => step(handle, messages));
	assert.deepEqual(JSON.parse(textOf(down)).mcpServers, [], "a piTools: both agent goes on without Pi's tools");
	messages.push(down, user("ECHO two"));
	const echo = await step(handle, messages);
	assert.match(textOf(echo), /^ECHO\[sess-2\]/, "the next turn runs on a new session");
	assert.match(textOf(echo), /<conversation-history>[\s\S]*SESSIONINFO[\s\S]*<\/conversation-history>/, "the new session is seeded");
	messages.push(echo, user("SESSIONINFO"));
	const info = JSON.parse(textOf(await step(handle, messages)));
	assert.equal(info.mcpServers[0]?.name, "pi", "the new session has Pi's MCP server");
});

test("another agent's session with the same id does not vouch for a session opened without Pi's tools", { skip: process.platform === "win32" }, async () => {
	const handle = makeProvider([agent(), agent({ id: "other", name: "Other agent" })]);
	const run = async (agentId: string, messages: Message[]) => {
		const target = handle.provider.getModels().find((candidate) => candidate.id === agentId);
		assert.ok(target);
		const last = (await collect(handle.provider.streamSimple(target, normalizeContext({ tools: [echoTool], messages })))).at(-1);
		assert.ok(last?.type === "done", `expected done, got ${JSON.stringify(last)}`);
		return last.message;
	};
	// WHY: one shared first message keeps the sync state the two same-named sessions share from looking diverged.
	const first = user("SESSIONINFO");
	const down = await withTmpdir(missingTmpdir(), () => run("fake", [first]));
	assert.deepEqual(JSON.parse(textOf(down)).mcpServers, []);
	const other = JSON.parse(textOf(await run("other", [first])));
	assert.equal(other.mcpServers[0]?.name, "pi", "the other agent's own sess-1 has Pi's tools");
	const again = await run("fake", [first, down, user("SESSIONINFO")]);
	assert.equal(JSON.parse(textOf(again)).mcpServers[0]?.name, "pi", "the first agent left its unwired sess-1");
});

test("http transport falls back to stdio when the agent does not advertise it", async () => {
	const handle = makeProvider([agent({ mcpTransport: "http" })]);
	const info = JSON.parse(textOf(await step(handle, [user("SESSIONINFO")])));
	assert.equal(info.mcpServers[0].type, undefined);
	assert.equal(info.mcpServers[0].name, "pi");
});

test("prompt mode: the system prompt goes in front of the first prompt only", async () => {
	const handle = makeProvider([agent()]);
	const messages: Message[] = [user("ECHO one")];
	const first = await step(handle, messages, "SYSTEM-RULES");
	assert.match(textOf(first), /<system-instructions>[\s\S]*SYSTEM-RULES[\s\S]*<\/system-instructions>ECHO one/);
	messages.push(first, user("ECHO two"));
	const second = await step(handle, messages, "SYSTEM-RULES");
	assert.doesNotMatch(textOf(second), /system-instructions/);
	messages.push(second, user("ECHO three"));
	const third = await step(handle, messages, "NEW-RULES");
	assert.match(textOf(third), /updated its instructions[\s\S]*NEW-RULES/);
});

test("session mode: the system prompt goes on session/new when the agent supports it", async () => {
	const handle = makeProvider([agent({ systemPrompt: "session", env: { FAKE_SYSPROMPT: "1" } })]);
	const messages: Message[] = [user("SESSIONINFO")];
	const info = JSON.parse(textOf(await step(handle, messages, "SYSTEM-RULES")));
	assert.deepEqual(info.systemPrompt, { mode: "append", content: [{ type: "text", text: "SYSTEM-RULES" }] });
	const echo = await step(handle, [...messages, user("ECHO x")], "SYSTEM-RULES");
	assert.doesNotMatch(textOf(echo), /system-instructions/);
});

test("session mode falls back to the prompt when the agent does not advertise it", async () => {
	const handle = makeProvider([agent({ systemPrompt: "session" })]);
	assert.match(textOf(await step(handle, [user("ECHO x")], "SYSTEM-RULES")), /<system-instructions>/);
});

test("every queued user message is sent, in order", async () => {
	const handle = makeProvider([agent()]);
	const message = await step(handle, [user("ECHO first"), user("second"), user("third")]);
	assert.match(textOf(message), /ECHO first\s*second\s*third>>>/);
});

test("history the session has not seen is sent as a transcript", async () => {
	const handle = makeProvider([agent()]);
	const other: Message = {
		role: "assistant",
		content: [{ type: "text", text: "answer from another model" }],
		api: "x",
		provider: "other",
		model: "m",
		usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } },
		stopReason: "stop",
		timestamp: ++clock,
	} as Message;
	const text = textOf(await step(handle, [user("earlier question"), other, user("ECHO now")]));
	assert.match(text, /<conversation-history>[\s\S]*\[user\]\nearlier question[\s\S]*\[assistant\]\nanswer from another model[\s\S]*<\/conversation-history>ECHO now/);
});

test("a continued conversation sends only the new message; a rewritten one starts a new session", async () => {
	const handle = makeProvider([agent()]);
	const messages: Message[] = [user("ECHO one")];
	const first = await step(handle, messages);
	const session = /ECHO\[(.+?)\]/.exec(textOf(first))?.[1];
	messages.push(first, user("ECHO two"));
	const second = textOf(await step(handle, messages));
	assert.equal(/ECHO\[(.+?)\]/.exec(second)?.[1], session);
	assert.doesNotMatch(second, /conversation-history/);
	// A tree jump back to the start: the session's reply is gone from Pi's history.
	const branched = textOf(await step(handle, [messages[0]!, { ...(first as AssistantMessage), timestamp: ++clock }, user("ECHO other")]));
	assert.notEqual(/ECHO\[(.+?)\]/.exec(branched)?.[1], session);
	assert.match(branched, /<conversation-history>[\s\S]*ECHO one/);
});

const sessionOf = (message: AssistantMessage) => /ECHO\[(.+?)\]/.exec(textOf(message))?.[1];

test("a fork that keeps an earlier reply but drops the latest one starts a new session seeded up to the fork", async () => {
	const handle = makeProvider([agent()]);
	const messages: Message[] = [user("ECHO one")];
	const first = await step(handle, messages);
	messages.push(first, user("ECHO two"));
	const second = await step(handle, messages);
	assert.equal(sessionOf(second), sessionOf(first));
	const forked = await capture(handle, [...messages.slice(0, 2), user("ECHO three")]);
	assert.equal(forked.message.stopReason, "stop");
	assert.notEqual(sessionOf(forked.message), sessionOf(first));
	assert.match(forked.prompt, /^<conversation-history>[\s\S]*ECHO one[\s\S]*<\/conversation-history>\nECHO three$/);
	assert.doesNotMatch(forked.prompt, /ECHO two/);
});

test("a fork that drops an aborted latest reply and its user starts a new session", async () => {
	const handle = makeProvider([agent()]);
	const messages: Message[] = [user("ECHO one")];
	const first = await step(handle, messages);
	messages.push(first, user("SLOW two"));
	messages.push(await abortAtText(handle, messages));
	const forked = await capture(handle, [...messages.slice(0, 2), user("ECHO three")]);
	assert.equal(forked.message.stopReason, "stop");
	assert.notEqual(sessionOf(forked.message), sessionOf(first));
	assert.doesNotMatch(forked.prompt, /SLOW two/);
});

test("a context edit that drops a user a later request still held starts a new session", async () => {
	const handle = makeProvider([agent()]);
	const messages: Message[] = [user("ECHO one")];
	messages.push(await step(handle, messages), user("ECHO two"));
	messages.push(await step(handle, messages), user("ECHO three"));
	const third = await step(handle, messages);
	assert.equal(sessionOf(third), "sess-1");
	const edited = await capture(handle, [...messages.filter((message) => message !== messages[2]), third, user("ECHO four")]);
	assert.equal(edited.message.stopReason, "stop");
	assert.notEqual(sessionOf(edited.message), "sess-1");
});

test("a context edit that drops a seeded foreign reply a later request still held starts a new session", async () => {
	const handle = makeProvider([agent()]);
	const messages: Message[] = [user("earlier question"), foreignReply("answer from another model"), user("ECHO two")];
	const second = await step(handle, messages);
	assert.match(textOf(second), /^ECHO\[sess-1\]<<<<conversation-history>[\s\S]*answer from another model/);
	messages.push(second, user("ECHO three"));
	const third = await step(handle, messages);
	assert.match(textOf(third), /^ECHO\[sess-1\]<<<ECHO three>>>$/);
	const edited = await capture(handle, [...messages.filter((message) => message !== messages[1]), third, user("ECHO four")]);
	assert.equal(edited.message.stopReason, "stop");
	assert.notEqual(sessionOf(edited.message), "sess-1");
	assert.match(edited.prompt, /^<conversation-history>\n[^\n]*\n\n\[user\]\nearlier question\n\n\[user\]\nECHO two\n\n\[assistant\]\nECHO\[sess-1\]/, "the seed goes from the first user straight to the next one");
});

test("user messages a context handler adds for one request only do not start a new session", async () => {
	const handle = makeProvider([agent()]);
	const messages: Message[] = [user("ECHO one")];
	const first = await step(handle, messages);
	messages.push(first, user("ECHO two"));
	const injected = await step(handle, [messages[0]!, user("one-shot note in history"), first, messages[2]!, user("one-shot note after")]);
	assert.match(textOf(injected), /^ECHO\[sess-1\]<<<<conversation-history>[\s\S]*one-shot note in history[\s\S]*ECHO twoone-shot note after>>>$/);
	messages.push(injected, user("ECHO three"));
	const next = textOf(await step(handle, messages));
	assert.match(next, /^ECHO\[sess-1\]<<<ECHO three>>>$/);
});

test("an assistant reply a context handler adds for one request only does not start a new session", async () => {
	const handle = makeProvider([agent()]);
	const messages: Message[] = [user("ECHO one")];
	const first = await step(handle, messages);
	messages.push(first, user("ECHO two"));
	const injected = await step(handle, [messages[0]!, foreignReply("one-shot synthetic reply"), first, messages[2]!]);
	assert.match(textOf(injected), /^ECHO\[sess-1\]<<<<conversation-history>[\s\S]*one-shot synthetic reply[\s\S]*<\/conversation-history>ECHO two>>>$/);
	messages.push(injected, user("ECHO three"));
	const next = textOf(await step(handle, messages));
	assert.match(next, /^ECHO\[sess-1\]<<<ECHO three>>>$/);
});

test("seeded replies the transcript left out do not count as removed history", async () => {
	const handle = makeProvider([agent()]);
	const huge = foreignReply(`HUGE ${"x".repeat(130_000)}`);
	const thinking = { ...(foreignReply("") as AssistantMessage), content: [{ type: "thinking", thinking: "hidden thoughts" }] } as Message;
	const messages: Message[] = [user("earlier question"), huge, thinking, foreignReply("answer from another model"), user("ECHO two")];
	const second = await step(handle, messages);
	assert.match(textOf(second), /^ECHO\[sess-1\]<<<<conversation-history>[\s\S]*\[2 earlier message\(s\) omitted\][\s\S]*answer from another model/);
	assert.doesNotMatch(textOf(second), /HUGE|hidden thoughts/);
	messages.push(second, user("ECHO three"));
	const third = await step(handle, messages);
	assert.match(textOf(third), /^ECHO\[sess-1\]<<<ECHO three>>>$/);
	messages.push(third, user("ECHO four"));
	const edited = textOf(await step(handle, messages.filter((message) => message !== huge && message !== thinking)));
	assert.match(edited, /^ECHO\[sess-1\]<<<ECHO four>>>$/);
});

test("compaction, which replaces the first message, starts a new session", async () => {
	const handle = makeProvider([agent()]);
	const messages: Message[] = [user("ECHO one")];
	const first = await step(handle, messages);
	messages.push(first, user("ECHO two"));
	const second = await step(handle, messages);
	const compacted = await capture(handle, [user("summary of the conversation so far"), second, user("ECHO three")]);
	assert.equal(compacted.message.stopReason, "stop");
	assert.notEqual(sessionOf(compacted.message), sessionOf(first));
	assert.match(compacted.prompt, /^<conversation-history>[\s\S]*summary of the conversation so far[\s\S]*<\/conversation-history>\nECHO three$/);
});

test("a steer after the agent's own tool call is sent as the next prompt once the turn ends", async () => {
	const handle = makeProvider([agent({ piTools: "off" })], { nativeTools: () => true });
	const messages: Message[] = [user("hello")];
	const first = await step(handle, messages);
	assert.equal(first.stopReason, "toolUse");
	messages.push(first, ...(await answerTools(handle, first)), user("ECHO follow-up"));
	const last = await step(handle, messages);
	assert.equal(last.stopReason, "stop");
	assert.match(textOf(last), /Hello world #0[\s\S]*ECHO\[sess-1\]<<<ECHO follow-up>>>/);
});

/** Streams `messages` and aborts on the first text delta; returns the aborted message. */
async function abortAtText(handle: AcpProviderHandle, messages: Message[]): Promise<AssistantMessage> {
	const controller = new AbortController();
	let last: AssistantMessageEvent | undefined;
	for await (const event of handle.provider.streamSimple(model(handle), normalizeContext({ tools: [echoTool], messages }), { signal: controller.signal })) {
		last = event;
		if (event.type === "text_delta") controller.abort();
	}
	assert.ok(last?.type === "error" && last.reason === "aborted", `expected aborted, got ${JSON.stringify(last)}`);
	return last.error;
}

const foreignReply = (text: string): Message =>
	({
		role: "assistant",
		content: [{ type: "text", text }],
		api: "x",
		provider: "other",
		model: "m",
		usage: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, totalTokens: 0, cost: { input: 0, output: 0, cacheRead: 0, cacheWrite: 0, total: 0 } },
		stopReason: "stop",
		timestamp: ++clock,
	}) as Message;

type Payload = { sessionId: string; prompt: { type: string; text?: string }[] };

/**
 * One provider call that records the prompt text it sends, after `hook` (an `onPayload` that may throw
 * or change the payload); returns the final message, whatever its end.
 */
async function capture(
	handle: AcpProviderHandle,
	messages: Message[],
	systemPrompt?: string,
	hook?: (payload: Payload) => Payload | void,
	extra: Record<string, unknown> = {},
) {
	const prompts: string[] = [];
	const context = normalizeContext({ ...(systemPrompt ? { systemPrompt } : {}), tools: [echoTool], messages });
	const options = {
		...extra,
		onPayload: (payload: unknown) => {
			const final = hook?.(payload as Payload) ?? (payload as Payload);
			prompts.push(final.prompt.map((block) => block.text ?? `[${block.type}]`).join("\n"));
			return final === payload ? undefined : final;
		},
	};
	const last = (await collect(handle.provider.streamSimple(model(handle), context, options))).at(-1);
	assert.ok(last?.type === "done" || last?.type === "error", `expected an end, got ${JSON.stringify(last)}`);
	return { message: last.type === "done" ? last.message : last.error, prompt: prompts.join("\n") };
}

/** The M1 context: a system prompt, foreign history, and three queued users. */
function retryContext(): Message[] {
	return [user("earlier question"), foreignReply("FOREIGN-REPLY"), user("ECHO first"), user("second"), user("third")];
}

function assertFullPrompt(prompt: string, extra = ""): void {
	assert.match(prompt, /<system-instructions>[\s\S]*SYSTEM-RULES/);
	assert.match(prompt, /<conversation-history>[\s\S]*FOREIGN-REPLY[\s\S]*<\/conversation-history>/);
	assert.doesNotMatch(prompt, /<conversation-history>[\s\S]*ECHO first[\s\S]*<\/conversation-history>/);
	assert.ok(prompt.endsWith(`ECHO first\nsecond\nthird${extra}`), prompt);
}

async function assertThirdRequestIsBare(handle: AcpProviderHandle, messages: Message[], reply: AssistantMessage): Promise<void> {
	messages.push(reply, user("ECHO fourth"));
	const third = await capture(handle, messages, "SYSTEM-RULES");
	assert.equal(third.message.stopReason, "stop");
	assert.equal(third.prompt, "ECHO fourth", "the third request repeats neither the instructions, the seed nor the users");
}

/**
 * The request after a failed one: the same context (Pi retrying), or one that keeps the failed reply and
 * adds a user message, whose prompt must still carry the failed request's users as requests.
 */
function afterFailure(messages: Message[], failed: AssistantMessage, keepFailed: boolean): string {
	if (!keepFailed) return "";
	messages.push(failed, user("ECHO extra"));
	return "\nECHO extra";
}

for (const keepFailed of [false, true]) {
	const variant = keepFailed ? ", keeping the failed reply," : "";
	test(`a prompt the agent rejects once is retried${variant} with the instructions, the history seed and every queued user`, async () => {
		const handle = makeProvider([agent({ env: { FAKE_FAIL_PROMPT_N: "1" } })]);
		const messages = retryContext();
		const rejected = await capture(handle, messages, "SYSTEM-RULES");
		assert.equal(rejected.message.stopReason, "error");
		assert.match(rejected.message.errorMessage ?? "", /one-shot rejection/);
		const extra = afterFailure(messages, rejected.message, keepFailed);
		const retry = await capture(handle, messages, "SYSTEM-RULES");
		assert.equal(retry.message.stopReason, "stop");
		assertFullPrompt(retry.prompt, extra);
		await assertThirdRequestIsBare(handle, messages, retry.message);
	});

	test(`a prompt dropped by a throwing onPayload hook is sent in full by the next request${variant}`, async () => {
		const handle = makeProvider([agent()]);
		const messages = retryContext();
		const thrown = await capture(handle, messages, "SYSTEM-RULES", () => {
			throw new Error("hook failed");
		});
		assert.equal(thrown.message.stopReason, "error");
		const extra = afterFailure(messages, thrown.message, keepFailed);
		const retry = await capture(handle, messages, "SYSTEM-RULES");
		assert.equal(retry.message.stopReason, "stop");
		assertFullPrompt(retry.prompt, extra);
		await assertThirdRequestIsBare(handle, messages, retry.message);
	});
}

test("a kept failed reply with nothing to show adds no history seed", async () => {
	const handle = makeProvider([agent({ env: { FAKE_FAIL_PROMPT_N: "1" } })]);
	const messages: Message[] = [user("ECHO first")];
	const rejected = await capture(handle, messages);
	assert.equal(rejected.message.stopReason, "error");
	messages.push(rejected.message, user("ECHO second"));
	const retry = await capture(handle, messages);
	assert.equal(retry.message.stopReason, "stop");
	assert.equal(retry.prompt, "ECHO first\nECHO second");
});

test("a history seed a hook removed from two answered prompts reaches the third once", async () => {
	const handle = makeProvider([agent()]);
	const stripSeed = (payload: Payload): void => {
		payload.prompt.splice(0, payload.prompt.length, ...payload.prompt.filter((block) => !/<conversation-history>/.test(block.text ?? "")));
	};
	const messages: Message[] = [user("earlier question"), foreignReply("FOREIGN-REPLY"), user("ECHO one")];
	for (const [index, next] of ["ECHO two", "ECHO three"].entries()) {
		const stripped = await capture(handle, messages, undefined, stripSeed);
		assert.equal(stripped.message.stopReason, "stop");
		assert.equal(stripped.prompt, index === 0 ? "ECHO one" : "ECHO two");
		messages.push(stripped.message, user(next));
	}
	const third = await capture(handle, messages);
	assert.match(third.prompt, /^<conversation-history>[\s\S]*earlier question[\s\S]*FOREIGN-REPLY[\s\S]*<\/conversation-history>\nECHO three$/);
	messages.push(third.message, user("ECHO four"));
	assert.equal((await capture(handle, messages)).prompt, "ECHO four");
});

test("cache-warming requests do not reach the agent", async () => {
	const handle = makeProvider([agent({ command: "/nonexistent/agent" })]);
	const message = await step(handle, [user("hi")], undefined, { maxTokens: 1, disableContinuation: true });
	assert.equal(message.stopReason, "stop");
	assert.deepEqual(message.content, []);
});

test("top-level combinators are flattened for MCP, with the constraint in the description", () => {
	const { schema, note } = flattenRootCombinators({
		type: "object",
		properties: { mode: { type: "string" } },
		oneOf: [
			{ properties: { path: { type: "string" } }, required: ["path"] },
			{ properties: { url: { type: "string" } }, required: ["url"] },
		],
	});
	assert.deepEqual(schema, { type: "object", properties: { mode: { type: "string" }, path: { type: "string" }, url: { type: "string" } } });
	assert.equal(note, "Input constraint: provide parameters for exactly one of: (path) or (url).");
	assert.equal(toMcpTool({ name: "bad", description: "", parameters: { type: "string" } }), null);
	assert.deepEqual(toMcpTool(echoTool)?.inputSchema, { type: "object", properties: { text: { type: "string" } }, required: ["text"] });
});

test("Pi tool calls are recognized in each agent's naming", () => {
	assert.equal(piToolRef("mcp__pi__echo_secret", undefined), "echo_secret");
	assert.equal(piToolRef(undefined, "mcp.pi.echo_secret"), "echo_secret");
	assert.equal(piToolRef(undefined, "mcp_pi.echo_secret"), "echo_secret");
	assert.equal(piToolRef("mcp_pi_echo_secret", "Run mcp_pi_echo_secret"), "echo_secret");
	assert.equal(piToolRef(undefined, "pi/read"), "read");
	for (const title of ["Read pipeline.ts", "mcp__pipeline__run", "api/read", "Run pip install", "Searching for relevant tools"]) {
		assert.equal(piToolRef(undefined, title), undefined, title);
	}
});

test("two conversations of one agent each list, approve and call their own Pi tools", async () => {
	const asked: string[] = [];
	const handle = recordingProvider({ FAKE_MCP_NAME: "mcp__pi__alpha", FAKE_PERMISSION_NAME: "mcp__pi__alpha" }, asked);
	const alpha: Tool = { ...echoTool, name: "alpha" };
	const beta: Tool = { ...echoTool, name: "beta" };
	let release: () => void = () => {};
	const gate = new Promise<void>((resolve) => (release = resolve));
	let paused: () => void = () => {};
	const reached = new Promise<void>((resolve) => (paused = resolve));
	const messages: Message[] = [user('MCPCALL:alpha:{"text":"a"} ASKPERM')];
	const first = step(handle, messages, undefined, {
		tools: [alpha],
		sessionId: "conv-a",
		onPayload: async () => {
			paused();
			await gate;
			return undefined;
		},
	});
	await reached;
	// WHY: conversation b's context lists only beta while conversation a's turn is about to call alpha.
	assert.match(textOf(await step(handle, [user("ECHO b")], undefined, { tools: [beta], sessionId: "conv-b" })), /ECHO\[/);
	release();
	const call = await first;
	assert.equal(call.stopReason, "toolUse", textOf(call));
	const block = call.content.at(-1);
	assert.ok(block?.type === "toolCall" && block.name === "alpha");
	messages.push(call, {
		role: "toolResult",
		toolCallId: block.id,
		toolName: block.name,
		content: [{ type: "text", text: "alpha ran" }],
		isError: false,
		timestamp: ++clock,
	} as Message);
	const last = await step(handle, messages, undefined, { tools: [alpha], sessionId: "conv-a" });
	assert.match(textOf(last), /PERM<<<allow>>> TOOLS<<<alpha>>> RESULT<<<alpha ran>>>/);
	assert.deepEqual(asked, [], "conversation a's own tool is approved without the policy");
});

test("after a respawn reuses a session id, a turn parked at a native tool call does not capture the new session's calls", async () => {
	const asked: string[] = [];
	const handle = recordingProvider({ FAKE_MCP_NAME: "mcp__pi__echo", FAKE_PERMISSION_NAME: "mcp__pi__echo" }, asked, {}, { nativeTools: () => true });
	const parked: Message[] = [user("SLOW go a")];
	const opening = await step(handle, parked, undefined, { sessionId: "conv-a" });
	assert.equal(opening.stopReason, "toolUse");
	const mirrored = opening.content.at(-1);
	assert.ok(mirrored?.type === "toolCall" && mirrored.name === "acp_tool");
	const crashing: Message[] = [user("CRASH now")];
	const crashCall = await step(handle, crashing, undefined, { sessionId: "conv-b" });
	assert.equal(crashCall.stopReason, "toolUse");
	// WHY: the continuation ends only once the turn settles, which is when the agent's exit is seen.
	crashing.push(crashCall, ...(await answerTools(handle, crashCall)));
	const crashed = (await collect(handle.provider.streamSimple(model(handle), normalizeContext({ tools: [echoTool], messages: crashing }), { sessionId: "conv-b" }))).at(-1);
	assert.ok(crashed?.type === "error");
	assert.match(crashed.error.errorMessage ?? "", /exited unexpectedly/);

	const fresh: Message[] = [user('MCPCALL:echo:{"text":"hi"} ASKPERM')];
	const call = await step(handle, fresh, undefined, { sessionId: "conv-c" });
	assert.equal(call.stopReason, "toolUse", textOf(call));
	fresh.push(call, ...(await answerTools(handle, call)));
	const answered = await step(handle, fresh, undefined, { sessionId: "conv-c" });
	assert.match(textOf(answered), /PERM<<<allow>>> TOOLS<<<echo>>> RESULT<<<echoed hi>>>/);
	assert.deepEqual(asked, [], "the new session's own Pi tool is approved without the policy");

	parked.push(opening, ...(await answerTools(handle, opening)));
	const late = (await collect(handle.provider.streamSimple(model(handle), normalizeContext({ tools: [echoTool], messages: parked }), { sessionId: "conv-a" }))).at(-1);
	assert.equal(late?.type, "error", "the parked turn ends with its agent's exit");
	fresh.push(answered, user("ECHO after"));
	const after = await step(handle, fresh, undefined, { sessionId: "conv-c" });
	assert.match(textOf(after), /^ECHO\[sess-1\]<<<ECHO after>>>$/, "conversation c keeps its session, which reuses the parked turn's id");
});
