/**
 * Configuration for the ACP provider extension.
 *
 * Agents are declared under `acp.agents` in Pi's `settings.json`
 * (`~/.pi/agent/` by default); each becomes the model `acp/<id>`.
 * Environment variables override the global options.
 *
 * How an agent is wired to Pi (tools, system prompt) starts from the baseline
 * every ACP agent supports: Pi's tools served over a stdio MCP server next to
 * the agent's own tools, and Pi's system prompt sent with the first prompt.
 * Agents known to support more get a preset, matched by command name, and
 * any key set in settings wins over the preset.
 */

import { readFileSync } from "node:fs";
import { basename } from "node:path";
import { isRecord } from "./acp.ts";

/**
 * Which tools the agent works with:
 * - `off`: only its own tools (Pi's are not offered).
 * - `both`: its own tools plus Pi's tools, served over MCP.
 * - `pi`: only Pi's tools; needs `ownToolsOffMeta` that disables the agent's own, else acts as `both`.
 */
export type PiToolsMode = "off" | "both" | "pi";
/** How the agent reaches Pi's MCP server: every ACP agent supports stdio; some also HTTP. */
export type McpTransport = "stdio" | "http";
/**
 * Where Pi's system prompt goes:
 * - `off`: nowhere; the agent uses only its own instructions.
 * - `prompt`: in front of the first prompt of each ACP session.
 * - `session`: ACP `systemPrompt` (append mode) on `session/new`, when the agent advertises it.
 */
export type SystemPromptMode = "off" | "prompt" | "session";

/** One ACP agent: a command that speaks ACP over stdio. */
export interface AcpAgentConfig {
	/** Model id: selected with `/model acp/<id>`. */
	id: string;
	/** Display name in the model picker and error messages. */
	name: string;
	command: string;
	args: string[];
	env: Record<string, string>;
	/** `authenticate` method to use when the agent reports `auth_required`. */
	authMethod: string | undefined;
	piTools: PiToolsMode;
	mcpTransport: McpTransport;
	systemPrompt: SystemPromptMode;
	/** Extra `_meta` for `session/new`, for agent-specific session options. */
	sessionMeta: Record<string, unknown> | undefined;
	/** `_meta` for `session/new` that turns off the agent's own tools; merged in only with `piTools: "pi"`. */
	ownToolsOffMeta: Record<string, unknown> | undefined;
}

export interface AcpProviderConfig {
	agents: AcpAgentConfig[];
	autoApproveTools: boolean;
	/** Bounds `initialize`, `session/new`, and session setting changes. */
	handshakeTimeoutMs: number;
	/** Wire diagnostics on stderr (`ACP_DEBUG`). */
	debug: boolean;
	/** Show agent tool calls as native Pi tool rows (`ACP_NATIVE_TOOLS`); inline text otherwise. */
	nativeTools: boolean;
}

export type AcpAgentSettings = Partial<Omit<AcpAgentConfig, "id" | "command">> & { command: string };

/** The `acp` settings object. */
export type AcpSettings = Partial<{
	agents: Record<string, AcpAgentSettings>;
	autoApproveTools: boolean;
	handshakeTimeoutMs: number;
	nativeTools: boolean;
}>;

const DEFAULTS = {
	autoApproveTools: true,
	handshakeTimeoutMs: 30_000,
	nativeTools: true,
} as const;

/** The MCP server name Pi's tools are served under. */
export const PI_MCP_SERVER = "pi";

type AgentPreset = Partial<
	Pick<AcpAgentConfig, "piTools" | "mcpTransport" | "systemPrompt" | "sessionMeta" | "ownToolsOffMeta" | "env">
>;

/** What every ACP agent supports (the level of Antigravity and Mistral Vibe). */
const BASELINE: Required<Pick<AgentPreset, "piTools" | "mcpTransport" | "systemPrompt">> = {
	piTools: "both",
	mcpTransport: "stdio",
	systemPrompt: "prompt",
};

/**
 * Presets for agents that advertise more than the baseline, keyed by command
 * name. The provider still checks the agent's advertised capabilities and
 * falls back to the baseline when one is missing.
 */
export const AGENT_PRESETS: Readonly<Record<string, AgentPreset>> = {
	// Claude Code runs on Pi's tools alone: `tools: []` removes its built-ins,
	// `strictMcpConfig` ignores MCP servers from its own config, the cloud
	// connector loader is switched off by env, and Pi's server is pre-approved.
	// DISABLE_AUTO_COMPACT leaves compaction to Pi: when both sides compact, the
	// prompt cache is flushed twice and the two thresholds race. It also stops
	// claude-agent-acp's idle compaction before the 1-hour cache expires.
	"claude-agent-acp": {
		piTools: "pi",
		mcpTransport: "http",
		systemPrompt: "session",
		sessionMeta: { claudeCode: { options: { allowedTools: [`mcp__${PI_MCP_SERVER}`] } } },
		ownToolsOffMeta: { claudeCode: { options: { tools: [], strictMcpConfig: true } } },
		env: { ENABLE_CLAUDEAI_MCP_SERVERS: "0", DISABLE_AUTO_COMPACT: "1" },
	},
	// Codex has no documented ACP switch for its built-in tools, so it keeps them.
	"codex-acp": {
		piTools: "both",
		mcpTransport: "http",
		systemPrompt: "session",
	},
};

/** The preset for `command`, matched by its file name without a Windows extension. */
export function presetFor(command: string): AgentPreset {
	const name = basename(command).replace(/\.(exe|cmd|bat)$/i, "");
	return AGENT_PRESETS[name] ?? {};
}

const isStringArray = (value: unknown): value is string[] =>
	Array.isArray(value) && value.every((item) => typeof item === "string");

function parseStringRecord(value: unknown): Record<string, string> | undefined {
	if (!isRecord(value)) return undefined;
	const entries = Object.entries(value);
	const strings = entries.filter((entry): entry is [string, string] => typeof entry[1] === "string");
	return strings.length === entries.length ? Object.fromEntries(strings) : undefined;
}

function oneOf<T extends string>(allowed: readonly T[], value: unknown): T | undefined {
	return allowed.includes(value as T) ? (value as T) : undefined;
}

function parseAgent(value: unknown): AcpAgentSettings | undefined {
	if (!isRecord(value) || typeof value.command !== "string" || !value.command) return undefined;
	const agent: AcpAgentSettings = { command: value.command };
	if (typeof value.name === "string" && value.name) agent.name = value.name;
	if (isStringArray(value.args)) agent.args = value.args;
	const env = parseStringRecord(value.env);
	if (env) agent.env = env;
	if (typeof value.authMethod === "string" && value.authMethod) agent.authMethod = value.authMethod;
	const piTools = oneOf(["off", "both", "pi"] as const, value.piTools);
	if (piTools) agent.piTools = piTools;
	const mcpTransport = oneOf(["stdio", "http"] as const, value.mcpTransport);
	if (mcpTransport) agent.mcpTransport = mcpTransport;
	const systemPrompt = oneOf(["off", "prompt", "session"] as const, value.systemPrompt);
	if (systemPrompt) agent.systemPrompt = systemPrompt;
	if (isRecord(value.sessionMeta)) agent.sessionMeta = value.sessionMeta;
	if (isRecord(value.ownToolsOffMeta)) agent.ownToolsOffMeta = value.ownToolsOffMeta;
	return agent;
}

/** Keeps only well-typed keys; invalid agents and values are dropped. */
export function parseSettings(value: unknown): AcpSettings {
	if (!isRecord(value)) return {};
	const settings: AcpSettings = {};
	if (isRecord(value.agents)) {
		const agents = Object.entries(value.agents).flatMap(([id, raw]): [string, AcpAgentSettings][] => {
			const agent = parseAgent(raw);
			return id && agent ? [[id, agent]] : [];
		});
		settings.agents = Object.fromEntries(agents);
	}
	if (typeof value.autoApproveTools === "boolean") settings.autoApproveTools = value.autoApproveTools;
	if (typeof value.nativeTools === "boolean") settings.nativeTools = value.nativeTools;
	const timeout = value.handshakeTimeoutMs;
	if (typeof timeout === "number" && Number.isFinite(timeout) && timeout > 0) settings.handshakeTimeoutMs = timeout;
	return settings;
}

/** Reads `acp` from a settings.json file; a missing or malformed file yields {}. */
export function loadSettings(path: string): AcpSettings {
	let json: unknown;
	try {
		json = JSON.parse(readFileSync(path, "utf8"));
	} catch {
		return {};
	}
	return parseSettings(isRecord(json) ? json.acp : undefined);
}

function parseBooleanFlag(value: string | undefined): boolean | undefined {
	switch (value?.toLowerCase()) {
		case "1":
		case "true":
			return true;
		case "0":
		case "false":
			return false;
		default:
			return undefined;
	}
}

/** Merges plain objects key by key; any other value in `override` replaces the base one. */
export function deepMerge(base: Record<string, unknown>, override: Record<string, unknown>): Record<string, unknown> {
	const merged: Record<string, unknown> = { ...base };
	for (const [key, value] of Object.entries(override)) {
		const current = merged[key];
		merged[key] = isRecord(current) && isRecord(value) ? deepMerge(current, value) : value;
	}
	return merged;
}

function resolveAgent(id: string, agent: AcpAgentSettings): AcpAgentConfig {
	const preset = presetFor(agent.command);
	const merge = (base: Record<string, unknown> | undefined, override: Record<string, unknown> | undefined) =>
		base && override ? deepMerge(base, override) : (override ?? base);
	return {
		id,
		name: agent.name ?? id,
		command: agent.command,
		args: agent.args ?? [],
		env: { ...preset.env, ...agent.env },
		authMethod: agent.authMethod,
		piTools: agent.piTools ?? preset.piTools ?? BASELINE.piTools,
		mcpTransport: agent.mcpTransport ?? preset.mcpTransport ?? BASELINE.mcpTransport,
		systemPrompt: agent.systemPrompt ?? preset.systemPrompt ?? BASELINE.systemPrompt,
		sessionMeta: merge(preset.sessionMeta, agent.sessionMeta),
		ownToolsOffMeta: merge(preset.ownToolsOffMeta, agent.ownToolsOffMeta),
	};
}

export function resolveConfig({
	settings,
	env,
}: {
	settings: AcpSettings;
	env: Record<string, string | undefined>;
}): AcpProviderConfig {
	return {
		agents: Object.entries(settings.agents ?? {}).map(([id, agent]) => resolveAgent(id, agent)),
		autoApproveTools:
			parseBooleanFlag(env.ACP_AUTO_APPROVE) ?? settings.autoApproveTools ?? DEFAULTS.autoApproveTools,
		handshakeTimeoutMs: settings.handshakeTimeoutMs ?? DEFAULTS.handshakeTimeoutMs,
		debug: parseBooleanFlag(env.ACP_DEBUG) ?? false,
		nativeTools: parseBooleanFlag(env.ACP_NATIVE_TOOLS) ?? settings.nativeTools ?? DEFAULTS.nativeTools,
	};
}
