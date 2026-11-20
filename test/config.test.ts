import test from "node:test";
import assert from "node:assert/strict";
import { parseSettings, resolveConfig } from "../src/config.ts";

test("defaults apply when nothing is configured", () => {
	assert.deepEqual(resolveConfig({ settings: {}, env: {} }), {
		agents: [],
		autoApproveTools: true,
		handshakeTimeoutMs: 30_000,
		debug: false,
		nativeTools: true,
	});
});

test("agents resolve with defaults for optional fields", () => {
	const config = resolveConfig({
		settings: parseSettings({
			agents: {
				copilot: { command: "copilot", args: ["--acp"] },
				gemini: { name: "Gemini", command: "gemini", env: { DEBUG: "0" }, authMethod: "oauth-personal" },
			},
		}),
		env: {},
	});
	const baseline = { piTools: "both", mcpTransport: "stdio", systemPrompt: "prompt", sessionMeta: undefined, ownToolsOffMeta: undefined };
	assert.deepEqual(config.agents, [
		{ id: "copilot", name: "copilot", command: "copilot", args: ["--acp"], env: {}, authMethod: undefined, ...baseline },
		{ id: "gemini", name: "Gemini", command: "gemini", args: [], env: { DEBUG: "0" }, authMethod: "oauth-personal", ...baseline },
	]);
});

test("environment overrides settings", () => {
	const config = resolveConfig({ settings: { autoApproveTools: true }, env: { ACP_AUTO_APPROVE: "false", ACP_DEBUG: "TRUE" } });
	assert.equal(config.autoApproveTools, false);
	assert.equal(config.debug, true);
});

test("invalid settings values and agents are dropped", () => {
	assert.deepEqual(
		parseSettings({
			agents: { nocmd: { args: ["x"] }, bad: { command: "a", args: ["ok", 1], env: { A: 2 } } },
			handshakeTimeoutMs: -5,
			autoApproveTools: "no",
		}),
		{ agents: { bad: { command: "a" } } },
	);
});

test("agents without a preset get the baseline every ACP agent supports", () => {
	const [vibe, agy] = resolveConfig({
		settings: parseSettings({ agents: { vibe: { command: "vibe-acp" }, agy: { command: "agy", args: ["acp"] } } }),
		env: {},
	}).agents;
	for (const agent of [vibe, agy]) {
		assert.equal(agent?.piTools, "both");
		assert.equal(agent?.mcpTransport, "stdio");
		assert.equal(agent?.systemPrompt, "prompt");
		assert.equal(agent?.sessionMeta, undefined);
	}
});

test("claude and codex presets apply by command name, and settings override them", () => {
	const [claude, codex, custom] = resolveConfig({
		settings: parseSettings({
			agents: {
				claude: { command: "/usr/local/bin/claude-agent-acp" },
				codex: { command: "codex-acp" },
				custom: {
					command: "claude-agent-acp",
					piTools: "both",
					env: { EXTRA: "1" },
					sessionMeta: { claudeCode: { options: { model: "opus" } } },
				},
			},
		}),
		env: {},
	}).agents;
	assert.equal(claude?.piTools, "pi");
	assert.equal(claude?.mcpTransport, "http");
	assert.equal(claude?.systemPrompt, "session");
	assert.deepEqual(claude?.ownToolsOffMeta, { claudeCode: { options: { tools: [], strictMcpConfig: true } } });
	assert.equal(claude?.env.ENABLE_CLAUDEAI_MCP_SERVERS, "0");
	assert.equal(codex?.piTools, "both");
	assert.equal(codex?.mcpTransport, "http");
	assert.equal(codex?.systemPrompt, "session");
	assert.equal(custom?.piTools, "both");
	assert.deepEqual(custom?.env, { ENABLE_CLAUDEAI_MCP_SERVERS: "0", EXTRA: "1" });
	assert.deepEqual(custom?.sessionMeta, { claudeCode: { options: { allowedTools: ["mcp__pi"], model: "opus" } } });
});

test("invalid wiring values are dropped", () => {
	assert.deepEqual(parseSettings({ agents: { a: { command: "a", piTools: "all", mcpTransport: "sse", systemPrompt: true } } }), {
		agents: { a: { command: "a" } },
	});
});
