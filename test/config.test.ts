import test from "node:test";
import assert from "node:assert/strict";
import { parseSettings, resolveConfig } from "../src/config.ts";

test("defaults apply when nothing is configured", () => {
	assert.deepEqual(resolveConfig({ settings: {}, env: {} }), {
		agents: [],
		autoApproveTools: true,
		handshakeTimeoutMs: 30_000,
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
	assert.deepEqual(config.agents, [
		{ id: "copilot", name: "copilot", command: "copilot", args: ["--acp"], env: {}, authMethod: undefined },
		{ id: "gemini", name: "Gemini", command: "gemini", args: [], env: { DEBUG: "0" }, authMethod: "oauth-personal" },
	]);
});

test("environment overrides settings", () => {
	const config = resolveConfig({ settings: { autoApproveTools: true }, env: { ACP_AUTO_APPROVE: "false" } });
	assert.equal(config.autoApproveTools, false);
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
