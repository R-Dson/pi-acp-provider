/**
 * ACP (Agent Client Protocol) v1 wire types and boundary parsers.
 *
 * Everything that crosses the stdio pipe is `unknown`; these parsers narrow it
 * exactly once, at the boundary, into the small domain types the provider
 * uses. Unknown fields and update kinds are ignored for forward compatibility.
 * Reference: https://agentclientprotocol.com/protocol/schema
 */

export const ACP_PROTOCOL_VERSION = 1;

/** ACP `auth_required` error code, returned by `session/new` or `session/prompt`. */
export const ACP_AUTH_REQUIRED = -32000;

export const RPC_METHOD_NOT_FOUND = -32601;
export const RPC_INVALID_PARAMS = -32602;
export const RPC_INTERNAL_ERROR = -32603;

export function isRecord(value: unknown): value is Record<string, unknown> {
	return typeof value === "object" && value !== null && !Array.isArray(value);
}

const optionalString = (value: unknown): string | undefined => (typeof value === "string" ? value : undefined);

/** `value` if it is one of `allowed`, else undefined. */
function oneOf<T extends string>(allowed: readonly T[], value: unknown): T | undefined {
	return allowed.includes(value as T) ? (value as T) : undefined;
}

// --- JSON-RPC envelopes ------------------------------------------------------

/** JSON-RPC ids sent by the agent may be numbers or strings; ours are numbers. */
export type RpcId = number | string;

export interface RpcError {
	code: number;
	message: string;
}

/** Incoming JSON-RPC message, classified by shape. */
export type RpcIncoming =
	| { kind: "response"; id: RpcId; outcome: { ok: true; result: unknown } | { ok: false; error: RpcError } }
	| { kind: "request"; id: RpcId; method: string; params: unknown }
	| { kind: "notification"; method: string; params: unknown };

const isRpcId = (value: unknown): value is RpcId => typeof value === "number" || typeof value === "string";

export function parseRpcMessage(value: unknown): RpcIncoming | null {
	if (!isRecord(value)) return null;
	const { id, method } = value;
	if (typeof method === "string") {
		return isRpcId(id)
			? { kind: "request", id, method, params: value.params }
			: { kind: "notification", method, params: value.params };
	}
	if (!isRpcId(id)) return null;
	const error = value.error;
	if (isRecord(error)) {
		return {
			kind: "response",
			id,
			outcome: {
				ok: false,
				error: {
					code: typeof error.code === "number" ? error.code : RPC_INTERNAL_ERROR,
					message: optionalString(error.message) ?? "Unknown agent error",
				},
			},
		};
	}
	return "result" in value ? { kind: "response", id, outcome: { ok: true, result: value.result } } : null;
}

// --- session/update ----------------------------------------------------------

const TOOL_CALL_STATUSES = ["pending", "in_progress", "completed", "failed"] as const;
export type ToolCallStatus = (typeof TOOL_CALL_STATUSES)[number];

/** The `session/update` variants the provider renders; everything else is dropped. */
export type AcpSessionUpdate =
	| { type: "agent_message_chunk"; text: string }
	| { type: "agent_thought_chunk"; text: string }
	| {
			/** `tool_call` announces a call; `tool_call_update` reports on it. Either may carry a final status. */
			type: "tool_call" | "tool_call_update";
			toolCallId: string;
			title: string | undefined;
			status: ToolCallStatus | undefined;
			summary: string;
	  };

export interface AcpCommand {
	name: string;
	description: string;
	/** Describes the command's arguments, when it takes any. */
	hint: string | undefined;
}

/** The `update` object of `session/update` params (`{ sessionId, update: { sessionUpdate, ... } }`). */
const updateOf = (params: unknown): Record<string, unknown> | undefined =>
	isRecord(params) && isRecord(params.update) ? params.update : undefined;

/** `available_commands_update`'s full command list, or null for other updates. */
export function parseAvailableCommands(params: unknown): AcpCommand[] | null {
	const update = updateOf(params);
	if (update?.sessionUpdate !== "available_commands_update" || !Array.isArray(update.availableCommands)) return null;
	return update.availableCommands.flatMap((command): AcpCommand[] =>
		isRecord(command) && typeof command.name === "string"
			? [
					{
						name: command.name,
						description: optionalString(command.description) ?? "",
						hint: isRecord(command.input) ? optionalString(command.input.hint) : undefined,
					},
				]
			: [],
	);
}

/** Text of a `ContentBlock`; non-text blocks (images, resources) render as nothing. */
function contentBlockText(block: unknown): string {
	return isRecord(block) && block.type === "text" ? (optionalString(block.text) ?? "") : "";
}

/** One-line summary of `ToolCallContent[]`: text output or the edited file paths. */
function toolContentSummary(content: unknown): string {
	if (!Array.isArray(content)) return "";
	const parts: string[] = [];
	for (const item of content) {
		if (!isRecord(item)) continue;
		if (item.type === "content") parts.push(contentBlockText(item.content));
		else if (item.type === "diff" && typeof item.path === "string") parts.push(`edited ${item.path}`);
	}
	return parts.filter(Boolean).join("; ");
}

export function parseSessionUpdate(params: unknown): AcpSessionUpdate | null {
	const update = updateOf(params);
	switch (update?.sessionUpdate) {
		case "agent_message_chunk":
		case "agent_thought_chunk":
			return { type: update.sessionUpdate, text: contentBlockText(update.content) };
		case "tool_call":
		case "tool_call_update":
			if (typeof update.toolCallId !== "string") return null;
			return {
				type: update.sessionUpdate,
				toolCallId: update.toolCallId,
				title: optionalString(update.title),
				status: oneOf(TOOL_CALL_STATUSES, update.status),
				summary: toolContentSummary(update.content),
			};
		default:
			return null;
	}
}

// --- session/request_permission ---------------------------------------------

const PERMISSION_KINDS = ["allow_once", "allow_always", "reject_once", "reject_always"] as const;
export type PermissionOptionKind = (typeof PERMISSION_KINDS)[number];

export interface AcpPermissionOption {
	optionId: string;
	name: string;
	kind: PermissionOptionKind;
}

export interface AcpPermissionRequest {
	/** Title of the tool call asking for permission, for display. */
	title: string;
	options: AcpPermissionOption[];
}

export type AcpPermissionOutcome = { outcome: "selected"; optionId: string } | { outcome: "cancelled" };

export function parsePermissionRequest(params: unknown): AcpPermissionRequest | null {
	if (!isRecord(params) || !Array.isArray(params.options)) return null;
	const options: AcpPermissionOption[] = [];
	for (const option of params.options) {
		if (!isRecord(option) || typeof option.optionId !== "string") continue;
		const kind = oneOf(PERMISSION_KINDS, option.kind);
		if (!kind) continue;
		options.push({ optionId: option.optionId, name: optionalString(option.name) ?? option.optionId, kind });
	}
	const title = (isRecord(params.toolCall) && optionalString(params.toolCall.title)) || "tool";
	return { title, options };
}

/** Selects the first offered option of the given kinds, in preference order. */
function selectOption(request: AcpPermissionRequest, kinds: PermissionOptionKind[]): AcpPermissionOutcome {
	for (const kind of kinds) {
		const option = request.options.find((candidate) => candidate.kind === kind);
		if (option) return { outcome: "selected", optionId: option.optionId };
	}
	return { outcome: "cancelled" };
}

/** Allows once (never "always", so the agent's own permission memory stays untouched). */
export const autoApprove = (request: AcpPermissionRequest): AcpPermissionOutcome =>
	selectOption(request, ["allow_once", "allow_always"]);

/** Rejects once; used when no UI can ask the user. */
export const autoReject = (request: AcpPermissionRequest): AcpPermissionOutcome =>
	selectOption(request, ["reject_once", "reject_always"]);

// --- Responses ----------------------------------------------------------------

export interface AcpAuthMethod {
	id: string;
	name: string;
	description: string | undefined;
}

export interface AcpAgentInfo {
	/** Negotiated protocol version from the `initialize` response. */
	protocolVersion: number;
	name: string;
	version: string;
	/** Methods the client may pass to `authenticate` when the agent reports `auth_required`. */
	authMethods: AcpAuthMethod[];
}

function parseAuthMethods(value: unknown): AcpAuthMethod[] {
	if (!Array.isArray(value)) return [];
	return value.flatMap((method): AcpAuthMethod[] =>
		isRecord(method) && typeof method.id === "string"
			? [{ id: method.id, name: optionalString(method.name) ?? method.id, description: optionalString(method.description) }]
			: [],
	);
}

export function parseInitializeResult(result: unknown): AcpAgentInfo {
	const record = isRecord(result) ? result : {};
	const info = isRecord(record.agentInfo) ? record.agentInfo : {};
	return {
		protocolVersion: typeof record.protocolVersion === "number" ? record.protocolVersion : ACP_PROTOCOL_VERSION,
		name: optionalString(info.name) ?? "unknown-agent",
		version: optionalString(info.version) ?? "0.0.0",
		authMethods: parseAuthMethods(record.authMethods),
	};
}

// --- Session configuration (models, reasoning level) -------------------------

export interface AcpChoice {
	value: string;
	name: string;
	description: string | undefined;
}

/** A selectable session setting and how to change it. */
export interface AcpSelect {
	/**
	 * `configId` for `session/set_config_option`; null for the legacy
	 * `models` API, which uses `session/set_model`.
	 */
	configId: string | null;
	currentValue: string | undefined;
	choices: AcpChoice[];
}

/** The session settings the provider maps onto Pi; both are absent when the agent offers none. */
export interface AcpSessionConfig {
	model: AcpSelect | undefined;
	/** `thought_level` config option: the agent's reasoning effort. */
	thoughtLevel: AcpSelect | undefined;
}

const emptySessionConfig = (): AcpSessionConfig => ({ model: undefined, thoughtLevel: undefined });

function parseChoice(record: Record<string, unknown>, value: string): AcpChoice {
	return { value, name: optionalString(record.name) ?? value, description: optionalString(record.description) };
}

/** Flattens a select option's `options`, which may be flat or grouped. */
function parseChoices(value: unknown): AcpChoice[] {
	if (!Array.isArray(value)) return [];
	return value.flatMap((item): AcpChoice[] => {
		if (!isRecord(item)) return [];
		if (Array.isArray(item.options)) return parseChoices(item.options);
		if (typeof item.value !== "string") return [];
		return [parseChoice(item, item.value)];
	});
}

/** The `select` config option of `category`, from a `configOptions` array. */
function findSelect(configOptions: unknown, category: string): AcpSelect | undefined {
	if (!Array.isArray(configOptions)) return undefined;
	const option: unknown = configOptions.find(
		(candidate) => isRecord(candidate) && candidate.category === category && candidate.type === "select",
	);
	if (!isRecord(option) || typeof option.id !== "string") return undefined;
	return { configId: option.id, currentValue: optionalString(option.currentValue), choices: parseChoices(option.options) };
}

/** Legacy (pre-config-options) `models: { availableModels, currentModelId }`. */
function parseLegacyModels(value: unknown): AcpSelect | undefined {
	if (!isRecord(value) || !Array.isArray(value.availableModels)) return undefined;
	const choices = value.availableModels.flatMap((model): AcpChoice[] =>
		isRecord(model) && typeof model.modelId === "string"
			? [parseChoice(model, model.modelId)]
			: [],
	);
	return { configId: null, currentValue: optionalString(value.currentModelId), choices };
}

/**
 * Reads models and reasoning level from a `session/new` or
 * `session/set_config_option` result (or a `config_option_update`).
 * `configOptions` wins over the legacy `models` field when both are present.
 */
export function parseSessionConfig(result: unknown, previous: AcpSessionConfig = emptySessionConfig()): AcpSessionConfig {
	if (!isRecord(result)) return previous;
	const hasConfigOptions = Array.isArray(result.configOptions);
	// A legacy model list only changes through `session/set_model`, so a
	// configOptions snapshot without a model option leaves it in place.
	const keptModel = !hasConfigOptions || previous.model?.configId === null ? previous.model : undefined;
	return {
		model: findSelect(result.configOptions, "model") ?? parseLegacyModels(result.models) ?? keptModel,
		thoughtLevel: hasConfigOptions ? findSelect(result.configOptions, "thought_level") : previous.thoughtLevel,
	};
}

/** The session settings after a `config_option_update`, or null for other updates. */
export function parseConfigOptionUpdate(params: unknown, previous: AcpSessionConfig): AcpSessionConfig | null {
	const update = updateOf(params);
	return update?.sessionUpdate === "config_option_update" ? parseSessionConfig(update, previous) : null;
}

export function parseNewSessionResult(result: unknown): { sessionId: string; config: AcpSessionConfig } {
	if (!isRecord(result) || typeof result.sessionId !== "string" || result.sessionId.length === 0) {
		throw new Error("session/new did not return a sessionId");
	}
	return { sessionId: result.sessionId, config: parseSessionConfig(result) };
}

const STOP_REASONS = ["end_turn", "max_tokens", "max_turn_requests", "refusal", "cancelled"] as const;
export type AcpStopReason = (typeof STOP_REASONS)[number];

/** Unknown or missing reasons count as `end_turn`. */
export function parseStopReason(result: unknown): AcpStopReason {
	return oneOf(STOP_REASONS, isRecord(result) ? result.stopReason : undefined) ?? "end_turn";
}
