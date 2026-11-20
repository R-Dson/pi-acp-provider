/**
 * Fake ACP (v1) agent for tests.
 *
 * Behavior is driven by markers in the prompt text and FAKE_* env vars:
 *   - "PERMISSION"  asks session/request_permission (string request id) first, for tool call "t0"
 *                   (never reported), titled FAKE_PERMISSION_TITLE (default "rm -rf build"), with
 *                   ACP `name` FAKE_PERMISSION_NAME when set; FAKE_PERMISSION_NO_REJECT=1 offers
 *                   only allow options
 *   - "SLOW"        delays each streamed update (~150ms)
 *                   (a PERMISSION answer that arrives after session/cancel is reported by the next
 *                   ECHO as "CANCELLEDPERM<<<<optionId|outcome>>>>")
 *   - "CRASH"       exits (code 3) mid-stream after the tool update
 *   - "AUTH"        fails the prompt with the ACP auth_required error (-32000)
 *   - "RPCERROR"    fails the prompt with an internal error
 *   - "MAXTOKENS"   finishes with stopReason "max_tokens"
 *   - "REFUSAL"     finishes with stopReason "refusal"
 *   - "ONESHOT"     first reports a tool call that is already completed
 *   - "DIFF"        first reports an edit tool call with a diff
 *   - "USAGE"       sends usage_update (used 1234, size 200000) and a summed prompt `usage`
 *     FAKE_USAGE_SIZE=<raw JSON number>  sends that literal as the size instead (e.g. 1e999)
 *   - "LONGLOOP"    like USAGE, but a many-call turn whose summed cacheRead is far past the
 *                   128k window while usage_update reports 100000 ("NOUPDATE" omits it)
 *   - FAKE_SPLIT=1  writes stdout in small split chunks (line-buffer test)
 *   - FAKE_MODELS=config  session/new returns configOptions (model + thought_level)
 *     FAKE_MODELS=legacy  session/new returns the legacy `models` field
 *                   Replies end with "[model=<id> thought=<level>]".
 *     FAKE_DEFAULT_MODEL=<id>  the model new sessions start on (default "fast")
 *     FAKE_THOUGHT_CATEGORY=<c>  category of the reasoning option (default "thought_level")
 *     FAKE_SETTING_DELAY=<ms>  applies and answers each session/set_config_option after <ms>
 *   - "/context"    (whole prompt) answers like an agent slash command
 *   - FAKE_PROTOCOL=n answers initialize with protocol version n
 *   - FAKE_AUTH=1   session/new fails with auth_required until `authenticate`
 *                   is called with methodId "fake-login"
 *   - "ECHO"        replies "ECHO[<session id>]<<<full prompt text>>>" and ends the turn
 *   - "SESSIONINFO" replies with JSON of the session's session/new params
 *   - "MCPCALL:<tool>:<json args>"  reports a tool call titled "mcp__pi__<tool>",
 *                   calls <tool> on the session's MCP server (stdio or http),
 *                   then replies "RESULT<<<text>>>" (plus "STEERED<<<text>>>"
 *                   when a _session/steering message arrived meanwhile); result
 *                   items are joined with "|", non-text ones as "[<type>:<mimeType>]"
 *   - "CALLREPORT"  with ECHO: first waits (up to 2s) until the last MCPCALL's tools/call got its
 *                   answer and its prompt got session/cancel, in whichever order they arrived, then
 *                   adds "CANCELLEDCALL<<<text>>>" (the call's answer), "UNCANCELLEDCALL<<<text>>>"
 *                   when no cancel came, or "CANCELLEDCALL<<<none>>>" without an MCPCALL
 *   - FAKE_MCP_TITLE=<t> titles that tool call <t> instead
 *   - FAKE_MCP_NAME=<n>  gives that tool call the structured ACP `name` <n>
 *     FAKE_MCP_META_NAME=<n>  and/or `_meta.claudeCode.toolName` <n>
 *   - "ASKPERM"     with MCPCALL: after reporting the tool call, asks permission for it (titled
 *                   FAKE_PERMISSION_TITLE or like the call, `name` FAKE_PERMISSION_NAME and
 *                   `_meta.claudeCode.toolName` FAKE_PERMISSION_META_NAME when set);
 *                   the reply starts "PERM<<<optionId|outcome>>>" and the call is made only when allowed
 *   - "OVERLAP"     with MCPCALL: first reports a pending call "r1" titled "Run something" with the same
 *                   input, waits 150ms, and once the MCP call is pending asks permission for r1 (shaped
 *                   like ASKPERM's); the reply adds "OVERLAPPERM<<<optionId|outcome>>>"
 *   - FAKE_MCP_CANCEL=1  cancels that MCP call after 50ms (`notifications/cancelled`) and ends the
 *                   turn with "AGENT-WENT-ON CALL<<<what the call returned>>>"
 *   - FAKE_MCP_GIVEUP=1  ends the turn after 50ms without waiting for the call
 *   - FAKE_CANCEL_LAG=<ms>  after session/cancel, sends a chunk "LATE-CHUNK", a command list
 *                   with only "late", a permission request and an MCP echo call, waits <ms>, then answers `cancelled`; the next
 *                   ECHO adds "LAG<<<permission=<outcome>;mcp=<rejected|accepted:...|none>>>>"
 *   - FAKE_IGNORE_CANCEL=1  never answers a cancelled prompt; on cancel it opens an MCP client to
 *                   the session's server. When the next session/prompt arrives (on any session),
 *                   each such lingering prompt sends, on its own session, a chunk "STALE-CHUNK", a
 *                   permission request and an MCP echo call over that client; the next ECHO adds
 *                   "STALE<<<permission=<outcome>;mcp=<rejected|accepted:...|pending|none>>>>"
 *   - "IGNORE-CANCEL"  in a prompt: that prompt alone behaves as under FAKE_IGNORE_CANCEL
 *   - "BADSESSIONPERM"  asks two permission requests, one without a sessionId and one with a
 *                   numeric one, and replies "BADPERM<<<missing=<outcome>;number=<outcome>>>>"
 *   - FAKE_FAIL_SESSION_N=<n>  the nth session/new of this process fails with an internal error,
 *                   after FAKE_FAIL_SESSION_DELAY=<ms> (default 0)
 *   - FAKE_FAIL_PROMPT_N=<n>[,<n>...]  the nth session/prompt of this process (counting every
 *                   prompt, on any session) gets a one-shot internal RPC error instead of running
 *   - FAKE_ORPHAN=<file>  at start, spawns a child in the agent's process group that ignores
 *                   SIGTERM and writes its pid to <file> (POSIX only)
 *   - FAKE_MCP_HTTP=1    advertises HTTP MCP support
 *   - FAKE_SYSPROMPT=1   advertises sessionCapabilities.systemPrompt
 *   - FAKE_STEERING=1    advertises and accepts `_session/steering`
 */

import { spawn } from "node:child_process";
import { createInterface } from "node:readline";

if (process.env.FAKE_ORPHAN) {
	const script = `process.on("SIGTERM", () => {}); require("node:fs").writeFileSync(${JSON.stringify(process.env.FAKE_ORPHAN)}, String(process.pid)); setInterval(() => {}, 1000);`;
	spawn(process.execPath, ["-e", script], { stdio: "ignore" }).unref();
}

const SPLIT = process.env.FAKE_SPLIT === "1";
let authenticated = process.env.FAKE_AUTH !== "1";

function emit(message) {
	const line = JSON.stringify(message) + "\n";
	if (!SPLIT) {
		process.stdout.write(line);
		return;
	}
	for (let index = 0; index < line.length; index += 7) {
		process.stdout.write(line.slice(index, index + 7));
	}
}

const reply = (id, result) => emit({ jsonrpc: "2.0", id, result });
const replyError = (id, code, message) => emit({ jsonrpc: "2.0", id, error: { code, message } });
const update = (sessionId, sessionUpdate) =>
	emit({ jsonrpc: "2.0", method: "session/update", params: { sessionId, update: sessionUpdate } });
const sleep = (ms) => new Promise((resolve) => setTimeout(resolve, ms));

const MODELS = process.env.FAKE_MODELS;
const settings = { model: process.env.FAKE_DEFAULT_MODEL ?? "fast", thought: "medium" };
const configOptions = () => [
	{
		id: "model",
		name: "Model",
		category: "model",
		type: "select",
		currentValue: settings.model,
		options: [
			{ group: "main", name: "Main", options: [{ value: "fast", name: "Fast" }] },
			{ value: "smart", name: "Smart", description: "Slow but clever" },
		],
	},
	{
		id: "effort",
		name: "Reasoning",
		category: process.env.FAKE_THOUGHT_CATEGORY ?? "thought_level",
		type: "select",
		currentValue: settings.thought,
		options: ["none", "low", "medium", "high"].map((value) => ({ value, name: value })),
	},
];
const sessionExtras = () =>
	MODELS === "config"
		? { configOptions: configOptions() }
		: MODELS === "legacy"
			? { models: { currentModelId: settings.model, availableModels: [{ modelId: "fast", name: "Fast" }, { modelId: "smart", name: "Smart" }] } }
			: {};

let sessionCounter = 0;
let sessionNewCounter = 0;
let promptCounter = 0;
/** session/prompt requests this process received, for FAKE_FAIL_PROMPT_N. */
let promptRequests = 0;
const FAIL_PROMPTS = new Set((process.env.FAKE_FAIL_PROMPT_N ?? "").split(",").filter(Boolean).map(Number));
let serverRequestId = 0;
const serverRequestWaiters = new Map();
/** The running prompt of each session. */
const activePrompts = new Map();
/** What the fake saw of a cancelled turn's late traffic (FAKE_CANCEL_LAG), reported by the next ECHO. */
let lagReport = null;
/** The answer a PERMISSION prompt got after it was cancelled, reported by the next ECHO. */
let cancelledPermission = null;
/** The last MCPCALL's tools/call answer and its prompt's session/cancel, reported by a CALLREPORT ECHO. */
let lastCall = null;
const sessionParams = new Map();
const steered = [];

/**
 * A tiny MCP client for one ACP McpServer entry: `request` resolves with the JSON-RPC response,
 * or `{ error }` when the server refuses or the connection closes.
 */
async function openMcp(server) {
	const init = { protocolVersion: "2025-06-18", capabilities: {}, clientInfo: { name: "fake-agent", version: "0" } };
	let client;
	if (server.type === "http") {
		const headers = { "content-type": "application/json", accept: "application/json, text/event-stream" };
		for (const header of server.headers ?? []) headers[header.name] = header.value;
		let mcpSession = null;
		const post = async (body) => {
			try {
				const response = await fetch(server.url, {
					method: "POST",
					headers: mcpSession ? { ...headers, "mcp-session-id": mcpSession } : headers,
					body: JSON.stringify(body),
				});
				mcpSession = response.headers.get("mcp-session-id") ?? mcpSession;
				if (response.status === 202) return null;
				if (!response.ok) return { error: { message: `HTTP ${response.status}` } };
				return await response.json();
			} catch (error) {
				return { error: { message: String(error) } };
			}
		};
		client = {
			request: (id, method, params) => post({ jsonrpc: "2.0", id, method, params }),
			notify: (method, params) => post({ jsonrpc: "2.0", method, params }),
			close() {},
		};
	} else {
		const env = { ...process.env };
		for (const entry of server.env ?? []) env[entry.name] = entry.value;
		const child = spawn(server.command, server.args, { env, stdio: ["pipe", "pipe", "inherit"] });
		const waiters = new Map();
		let closed = false;
		createInterface({ input: child.stdout }).on("line", (line) => {
			const message = JSON.parse(line);
			waiters.get(message.id)?.(message);
			waiters.delete(message.id);
		});
		child.on("exit", () => {
			closed = true;
			for (const resolve of waiters.values()) resolve({ error: { message: "connection closed" } });
			waiters.clear();
		});
		const write = (message) => {
			if (!closed) child.stdin.write(JSON.stringify(message) + "\n");
		};
		child.stdin.on("error", () => {});
		client = {
			request: (id, method, params) =>
				closed
					? Promise.resolve({ error: { message: "connection closed" } })
					: new Promise((resolve) => {
							waiters.set(id, resolve);
							write({ jsonrpc: "2.0", id, method, params });
						}),
			notify: (method, params) => write({ jsonrpc: "2.0", method, params }),
			close: () => child.kill(),
		};
	}
	await client.request(1, "initialize", init);
	await client.notify("notifications/initialized");
	return client;
}

const CALL_ID = 3;

/** Lists the tools, then starts `tools/call`; `result` is the pending call. */
async function startMcpCall(server, name, args) {
	const client = await openMcp(server);
	const list = await client.request(2, "tools/list", {});
	const result = client.request(CALL_ID, "tools/call", { name, arguments: args });
	return { client, tools: list.result?.tools ?? [], result };
}

/** The text of a `tools/call` response, or `error:<message>` when it failed at the protocol level. */
const callText = (response) =>
	response.error
		? `error:${response.error.message}`
		: (response.result.content ?? [])
				.map((block) => (block.type === "text" ? block.text : `[${block.type}${block.mimeType ? `:${block.mimeType}` : ""}]`))
				.join("|");

const CANCEL_LAG = Number(process.env.FAKE_CANCEL_LAG ?? 0);

/** FAKE_CANCEL_LAG: after session/cancel, a chunk, a permission request and an MCP call, then `cancelled`. */
async function lagThenCancel(turn) {
	const { sessionId } = turn;
	await sleep(10);
	update(sessionId, { sessionUpdate: "agent_message_chunk", content: { type: "text", text: "LATE-CHUNK" } });
	update(sessionId, { sessionUpdate: "available_commands_update", availableCommands: [{ name: "late", description: "Late command list" }] });
	const response = await serverRequest("session/request_permission", {
		sessionId,
		toolCall: { toolCallId: "late", title: "late tool" },
		options: [{ optionId: "allow", name: "Allow", kind: "allow_once" }],
	});
	const permission = response.result?.outcome?.outcome ?? "error";
	let mcp = "none";
	const server = (sessionParams.get(sessionId)?.mcpServers ?? []).find((entry) => entry.name === "pi");
	if (server) {
		const { client, result } = await startMcpCall(server, "echo", { text: "late" });
		const call = await result;
		client.close();
		mcp = call.error || call.result.isError ? "rejected" : `accepted:${callText(call)}`;
	}
	lagReport = `permission=${permission};mcp=${mcp}`;
	await sleep(CANCEL_LAG);
	turn.finish("cancelled");
}

const IGNORE_CANCEL = process.env.FAKE_IGNORE_CANCEL === "1";
/** FAKE_IGNORE_CANCEL: cancelled prompts never answered, with the MCP client each opened on cancel. */
const lingering = [];
/** What the lingering prompts' traffic got, reported by the next ECHO. */
let staleReport = null;

function linger(turn) {
	const server = (sessionParams.get(turn.sessionId)?.mcpServers ?? []).find((entry) => entry.name === "pi");
	lingering.push({ sessionId: turn.sessionId, mcp: server ? openMcp(server) : Promise.resolve(null) });
}

/** FAKE_IGNORE_CANCEL: old-session traffic, sent after the next prompt has arrived. */
async function sendStaleTraffic() {
	for (const { sessionId, mcp: opening } of lingering.splice(0)) {
		update(sessionId, { sessionUpdate: "agent_message_chunk", content: { type: "text", text: "STALE-CHUNK" } });
		const response = await serverRequest("session/request_permission", {
			sessionId,
			toolCall: { toolCallId: "stale", title: "stale tool" },
			options: [{ optionId: "allow", name: "Allow", kind: "allow_once" }],
		});
		const permission = response.result?.outcome?.outcome ?? "error";
		const client = await opening;
		let mcp = "none";
		if (client) {
			const call = await Promise.race([
				client.request(CALL_ID, "tools/call", { name: "echo", arguments: { text: "stale" } }),
				sleep(500).then(() => null),
			]);
			client.close();
			mcp = call === null ? "pending" : call.error || call.result.isError ? "rejected" : `accepted:${callText(call)}`;
		}
		staleReport = `permission=${permission};mcp=${mcp}`;
	}
}

function serverRequest(method, params) {
	return new Promise((resolve) => {
		const id = `srv-${++serverRequestId}`;
		serverRequestWaiters.set(id, resolve);
		emit({ jsonrpc: "2.0", id, method, params });
	});
}

/** The structured identity fields named by two env knobs: ACP `name` and `_meta.claudeCode.toolName`. */
function identityFields(nameKnob, metaKnob) {
	const name = process.env[nameKnob];
	const meta = process.env[metaKnob];
	return { ...(name ? { name } : {}), ...(meta ? { _meta: { claudeCode: { toolName: meta } } } : {}) };
}

/** session/request_permission for `toolCallId`, shaped by the FAKE_PERMISSION_* knobs. */
function askPermission(sessionId, toolCallId, title) {
	return serverRequest("session/request_permission", {
		sessionId,
		toolCall: { toolCallId, title: process.env.FAKE_PERMISSION_TITLE ?? title, ...identityFields("FAKE_PERMISSION_NAME", "FAKE_PERMISSION_META_NAME") },
		options: [
			{ optionId: "allow", name: "Allow", kind: "allow_once" },
			{ optionId: "allow-all", name: "Always allow", kind: "allow_always" },
			...(process.env.FAKE_PERMISSION_NO_REJECT === "1" ? [] : [{ optionId: "deny", name: "Deny", kind: "reject_once" }]),
		],
	});
}

async function callReport() {
	if (!lastCall) return "CANCELLEDCALL<<<none>>>";
	const { answer, cancelSeen } = lastCall;
	const cancelled = await Promise.race([cancelSeen.then(() => true), sleep(2000).then(() => false)]);
	const text = await Promise.race([answer, sleep(2000).then(() => "timeout")]);
	return `${cancelled ? "" : "UN"}CANCELLEDCALL<<<${text}>>>`;
}

async function handlePrompt(message) {
	const { sessionId } = message.params;
	await sendStaleTraffic();
	const text = (message.params.prompt ?? []).map((block) => block?.text ?? "").join("");
	const cancelSeen = Promise.withResolvers();
	const turn = {
		ignoreCancel: IGNORE_CANCEL || text.includes("IGNORE-CANCEL"),
		sessionId,
		finished: false,
		/** Set by session/cancel: the turn stops its scripted steps. */
		cancelled: false,
		/** Resolves on session/cancel, even one that arrives after the turn finished. */
		cancelSeen: cancelSeen.promise,
		markCancelSeen: cancelSeen.resolve,
		finish(stopReason) {
			if (this.finished) return;
			this.finished = true;
			reply(message.id, { stopReason });
		},
	};
	activePrompts.set(sessionId, turn);
	const stepDelay = text.includes("SLOW") ? 150 : 5;

	if (text.includes("LONGLOOP")) {
		if (!text.includes("NOUPDATE")) update(sessionId, { sessionUpdate: "usage_update", used: 100000, size: 128000 });
		update(sessionId, { sessionUpdate: "agent_message_chunk", content: { type: "text", text: "ok" } });
		return reply(message.id, {
			stopReason: "end_turn",
			usage: { inputTokens: 2000, outputTokens: 9000, cachedReadTokens: 4869752, cachedWriteTokens: 120000, totalTokens: 5000752 },
		});
	}
	if (text.includes("USAGE")) {
		const rawSize = process.env.FAKE_USAGE_SIZE;
		if (rawSize === undefined) update(sessionId, { sessionUpdate: "usage_update", used: 1234, size: 200000 });
		else {
			const line = JSON.stringify({ jsonrpc: "2.0", method: "session/update", params: { sessionId, update: { sessionUpdate: "usage_update", used: 1234, size: "__SIZE__" } } });
			process.stdout.write(line.replace('"__SIZE__"', rawSize) + "\n");
		}
		update(sessionId, { sessionUpdate: "agent_message_chunk", content: { type: "text", text: "ok" } });
		return reply(message.id, {
			stopReason: "end_turn",
			usage: { inputTokens: 10, outputTokens: 20, cachedReadTokens: 30000, cachedWriteTokens: 40, totalTokens: 30070 },
		});
	}
	if (text === "/context") {
		update(sessionId, { sessionUpdate: "agent_message_chunk", content: { type: "text", text: "context: 42%" } });
		return turn.finish("end_turn");
	}
	if (text.includes("ECHO")) {
		const lag = lagReport === null ? "" : ` LAG<<<${lagReport}>>>`;
		const stale = `${staleReport === null ? "" : ` STALE<<<${staleReport}>>>`}${cancelledPermission === null ? "" : ` CANCELLEDPERM<<<${cancelledPermission}>>>`}${text.includes("CALLREPORT") ? ` ${await callReport()}` : ""}`;
		update(sessionId, { sessionUpdate: "agent_message_chunk", content: { type: "text", text: `ECHO[${sessionId}]<<<${text}>>>${lag}${stale}` } });
		promptCounter++;
		return turn.finish("end_turn");
	}
	if (text.includes("SESSIONINFO")) {
		update(sessionId, { sessionUpdate: "agent_message_chunk", content: { type: "text", text: JSON.stringify(sessionParams.get(sessionId) ?? null) } });
		return turn.finish("end_turn");
	}
	const mcpMatch = /MCPCALL:([\w-]+):(\{.*?\})(?:$|\s)/.exec(text);
	if (mcpMatch) {
		const [, tool, json] = mcpMatch;
		const args = JSON.parse(json);
		const overlap = text.includes("OVERLAP");
		if (overlap) {
			update(sessionId, { sessionUpdate: "tool_call", toolCallId: "r1", title: "Run something", status: "pending", rawInput: args });
			// WHY: long enough for Pi to show r1 as a native row and wait on it before the Pi tool call arrives.
			await sleep(150);
		}
		const title = process.env.FAKE_MCP_TITLE ?? `mcp__pi__${tool}`;
		const name = identityFields("FAKE_MCP_NAME", "FAKE_MCP_META_NAME");
		update(sessionId, { sessionUpdate: "tool_call", toolCallId: "m1", title, ...name, status: "pending", rawInput: args });
		await sleep(20);
		let perm = "";
		if (text.includes("ASKPERM")) {
			const answer = (await askPermission(sessionId, "m1", title)).result?.outcome ?? {};
			const chosen = answer.outcome === "selected" ? answer.optionId : answer.outcome ?? "error";
			perm = `PERM<<<${chosen}>>> `;
			if (chosen !== "allow") {
				update(sessionId, { sessionUpdate: "agent_message_chunk", content: { type: "text", text: perm.trim() } });
				return turn.finish("end_turn");
			}
		}
		const server = (sessionParams.get(sessionId)?.mcpServers ?? []).find((entry) => entry.name === "pi");
		if (!server) {
			update(sessionId, { sessionUpdate: "agent_message_chunk", content: { type: "text", text: "NO-MCP-SERVER" } });
			return turn.finish("end_turn");
		}
		const { client, tools, result: pending } = await startMcpCall(server, tool, args);
		lastCall = { answer: pending.then(callText), cancelSeen: turn.cancelSeen };
		if (process.env.FAKE_MCP_CANCEL === "1" || process.env.FAKE_MCP_GIVEUP === "1") {
			// WHY: long enough for the provider to hand the call to Pi before the agent stops waiting.
			await sleep(50);
			if (process.env.FAKE_MCP_CANCEL === "1") client.notify("notifications/cancelled", { requestId: CALL_ID, reason: "timeout" });
			const settled =
				process.env.FAKE_MCP_CANCEL === "1"
					? await Promise.race([pending, sleep(300).then(() => ({ error: { message: "still pending" } }))])
					: { error: { message: "gave up" } };
			update(sessionId, { sessionUpdate: "tool_call_update", toolCallId: "m1", status: "failed" });
			update(sessionId, { sessionUpdate: "agent_message_chunk", content: { type: "text", text: `AGENT-WENT-ON CALL<<<${callText(settled)}>>>` } });
			turn.finish("end_turn");
			client.close();
			return;
		}
		if (overlap) {
			// WHY: by now the provider has handed the call to Pi, which ends r1's waiting row as a side effect.
			await sleep(50);
			const answer = (await askPermission(sessionId, "r1", "Run something")).result?.outcome ?? {};
			perm += `OVERLAPPERM<<<${answer.outcome === "selected" ? answer.optionId : answer.outcome ?? "error"}>>> `;
		}
		const response = await pending;
		client.close();
		// WHY: a prompt that already answered `cancelled` sends nothing more, or it would reach the next one.
		if (turn.finished) return;
		const result = response.result ?? { content: [{ type: "text", text: callText(response) }], isError: true };
		update(sessionId, { sessionUpdate: "tool_call_update", toolCallId: "m1", status: result.isError ? "failed" : "completed" });
		const resultText = callText(response);
		const extra = steered.length > 0 ? ` STEERED<<<${steered.splice(0).join("|")}>>>` : "";
		update(sessionId, {
			sessionUpdate: "agent_message_chunk",
			content: { type: "text", text: `${perm}TOOLS<<<${tools.map((entry) => entry.name).join(",")}>>> RESULT<<<${resultText}>>>${extra}` },
		});
		promptCounter++;
		return turn.finish("end_turn");
	}
	if (text.includes("BADSESSIONPERM")) {
		const ask = async (sessionField) =>
			(await serverRequest("session/request_permission", {
				...sessionField,
				toolCall: { toolCallId: "bad", title: "bad session" },
				options: [{ optionId: "allow", name: "Allow", kind: "allow_once" }],
			})).result?.outcome?.outcome ?? "error";
		const missing = await ask({});
		const number = await ask({ sessionId: 42 });
		update(sessionId, { sessionUpdate: "agent_message_chunk", content: { type: "text", text: `BADPERM<<<missing=${missing};number=${number}>>>` } });
		return turn.finish("end_turn");
	}
	if (text.includes("AUTH")) return replyError(message.id, -32000, "Authentication required");
	if (text.includes("RPCERROR")) return replyError(message.id, -32603, "fake rpc failure");

	let approved = null;
	if (text.includes("PERMISSION")) {
		const response = await askPermission(sessionId, "t0", "rm -rf build");
		const outcome = response.result?.outcome ?? {};
		approved = outcome.outcome === "selected" ? outcome.optionId : outcome.outcome ?? "error";
		if (turn.cancelled) cancelledPermission = approved;
		if (turn.finished || turn.cancelled) return;
	}

	const steps = [
		{ sessionUpdate: "agent_thought_chunk", content: { type: "text", text: "Considering the request... " } },
		{ sessionUpdate: "tool_call", toolCallId: "t1", title: "grep for secrets", kind: "search", status: "pending" },
		{ sessionUpdate: "tool_call_update", toolCallId: "t1", status: "in_progress" },
		{
			sessionUpdate: "tool_call_update",
			toolCallId: "t1",
			status: "completed",
			content: [{ type: "content", content: { type: "text", text: "3 matches\nsecond line" } }],
		},
		{ sessionUpdate: "plan", entries: [] },
		{ sessionUpdate: "agent_message_chunk", content: { type: "text", text: "Hello " } },
		{
			sessionUpdate: "agent_message_chunk",
			content: { type: "text", text: `world #${promptCounter}${approved !== null ? ` [approved=${approved}]` : ""}${MODELS ? ` [model=${settings.model} thought=${settings.thought}]` : ""}` },
		},
	];
	if (text.includes("ONESHOT")) {
		steps.unshift({
			sessionUpdate: "tool_call",
			toolCallId: "t0",
			title: "read notes",
			status: "completed",
			content: [{ type: "content", content: { type: "text", text: "2 notes" } }],
		});
	}
	if (text.includes("DIFF")) {
		steps.unshift(
			{ sessionUpdate: "tool_call", toolCallId: "e1", title: "Edit notes.txt", kind: "edit", status: "in_progress", locations: [{ path: "/tmp/notes.txt", line: 2 }] },
			{
				sessionUpdate: "tool_call_update",
				toolCallId: "e1",
				status: "completed",
				content: [{ type: "diff", path: "/tmp/notes.txt", oldText: "one\ntwo\n", newText: "one\nTWO\n" }],
			},
		);
	}
	for (const step of steps) {
		if (turn.finished || turn.cancelled) return;
		update(sessionId, step);
		if (step.status === "completed" && text.includes("CRASH")) process.exit(3);
		await sleep(stepDelay);
	}
	promptCounter++;
	turn.finish(text.includes("MAXTOKENS") ? "max_tokens" : text.includes("REFUSAL") ? "refusal" : "end_turn");
}

createInterface({ input: process.stdin }).on("line", (line) => {
	let message;
	try {
		message = JSON.parse(line);
	} catch {
		return;
	}
	if (message.method === undefined) {
		const waiter = serverRequestWaiters.get(message.id);
		serverRequestWaiters.delete(message.id);
		waiter?.(message);
		return;
	}
	switch (message.method) {
		case "initialize":
			reply(message.id, {
				protocolVersion: Number(process.env.FAKE_PROTOCOL ?? 1),
				agentCapabilities: {
					loadSession: false,
					promptCapabilities: { image: true },
					...(process.env.FAKE_MCP_HTTP === "1" ? { mcpCapabilities: { http: true } } : {}),
					...(process.env.FAKE_SYSPROMPT === "1" ? { sessionCapabilities: { systemPrompt: {} } } : {}),
				},
				...(process.env.FAKE_STEERING === "1" ? { _meta: { steering: { supported: true } } } : {}),
				agentInfo: { name: `fake-agent-${process.pid}`, version: "0.0.1" },
				authMethods: [{ id: "fake-login", name: "Fake login", description: "Pretend to sign in" }],
			});
			break;
		case "authenticate":
			if (message.params?.methodId !== "fake-login") {
				replyError(message.id, -32602, "unknown auth method");
				break;
			}
			authenticated = true;
			reply(message.id, {});
			break;
		case "session/new":
			if (!authenticated) replyError(message.id, -32000, "Authentication required");
			else if (++sessionNewCounter === Number(process.env.FAKE_FAIL_SESSION_N)) {
				const { id } = message;
				setTimeout(() => replyError(id, -32603, "fake session/new failure"), Number(process.env.FAKE_FAIL_SESSION_DELAY ?? 0));
			}
			else {
				const sessionId = `sess-${++sessionCounter}`;
				sessionParams.set(sessionId, message.params);
				reply(message.id, { sessionId, ...sessionExtras() });
				update(sessionId, {
					sessionUpdate: "available_commands_update",
					availableCommands: [
						{ name: "context", description: "Show context usage" },
						{ name: "session", description: "Session tools", input: { hint: "info | rename" } },
					],
				});
			}
			break;
		case "session/set_config_option": {
			const { configId, value } = message.params;
			if (MODELS !== "config" || (configId !== "model" && configId !== "effort")) {
				replyError(message.id, -32602, "unknown config option");
				break;
			}
			const { id } = message;
			const apply = () => {
				settings[configId === "model" ? "model" : "thought"] = value;
				reply(id, { configOptions: configOptions() });
			};
			const delay = Number(process.env.FAKE_SETTING_DELAY ?? 0);
			if (delay > 0) setTimeout(apply, delay);
			else apply();
			break;
		}
		case "session/set_model":
			if (MODELS !== "legacy") replyError(message.id, -32601, "Method not found");
			else {
				settings.model = message.params.modelId;
				reply(message.id, {});
			}
			break;
		case "session/prompt":
			if (FAIL_PROMPTS.has(++promptRequests)) replyError(message.id, -32603, `fake one-shot rejection of prompt #${promptRequests}`);
			else void handlePrompt(message);
			break;
		case "_session/steering": {
			if (process.env.FAKE_STEERING !== "1") {
				replyError(message.id, -32601, "Method not found");
				break;
			}
			const steerText = (message.params.prompt ?? []).map((block) => block?.text ?? "").join("");
			const running = activePrompts.get(message.params.sessionId);
			if (running && !running.finished) {
				steered.push(steerText);
				reply(message.id, { outcome: "steered" });
			} else reply(message.id, { outcome: "promptRequired", reason: "noRunningTurn" });
			break;
		}
		case "session/cancel": {
			const running = activePrompts.get(message.params?.sessionId);
			running?.markCancelSeen();
			if (!running || running.finished) break;
			running.cancelled = true;
			if (running.ignoreCancel) linger(running);
			else if (CANCEL_LAG > 0) void lagThenCancel(running);
			else running.finish("cancelled");
			break;
		}
		default:
			if (message.id !== undefined) replyError(message.id, -32601, `Method not found: ${message.method}`);
	}
});
