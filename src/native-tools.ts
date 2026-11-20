/**
 * Native Pi display for tool calls that an ACP agent runs itself.
 *
 * Pi only draws tool rows for tool calls in an assistant message. The provider
 * therefore ends each Pi message at an ACP `tool_call` with a `toolCall` block
 * for {@link ACP_TOOL_NAME}. Pi "executes" that mirror tool, which waits for the
 * agent's final report of the call and returns it, and then calls the provider
 * again, which continues the same ACP turn. The tool hides its own declaration
 * (`prepareLoadout`), so no model is ever offered it.
 */

import { generateDiffString, renderDiff, type ToolDefinition } from "@earendil-works/pi-coding-agent";
import { Text } from "@earendil-works/pi-tui";
import { Type } from "typebox";
import type { AcpToolContent, AcpToolDiff, AcpToolLocation, ToolCallStatus } from "./acp.ts";

export const ACP_TOOL_NAME = "acp_tool";

/** Arguments of a mirrored call, as stored in the transcript. */
export interface AcpToolArgs {
	agent: string;
	title: string;
	kind?: string;
	locations?: AcpToolLocation[];
	input?: unknown;
}

export type AcpToolDetails = {
	status: ToolCallStatus | "unknown";
	diffs: { path: string; diff: string; created: boolean }[];
	output: string;
	terminals: string[];
};

/** What the agent last reported about one call. */
export interface AcpToolState {
	status: ToolCallStatus | undefined;
	content: AcpToolContent;
	rawOutput: unknown;
}

/** Resolves a mirrored call: its final state once the agent reports one, streaming partial states before. */
export type AwaitTool = (
	toolCallId: string,
	signal: AbortSignal | undefined,
	onPartial: (state: AcpToolState) => void,
) => Promise<AcpToolState>;

const MAX_INPUT_CHARS = 4000;
const MAX_OUTPUT_CHARS = 64_000;
const COLLAPSED_LINES = 12;

/** `rawInput` for the transcript, dropped when too large to be worth keeping. */
export function compactInput(rawInput: unknown): unknown {
	if (rawInput === undefined) return undefined;
	try {
		const json = JSON.stringify(rawInput);
		return json !== undefined && json.length <= MAX_INPUT_CHARS ? rawInput : undefined;
	} catch {
		return undefined;
	}
}

function rawOutputText(rawOutput: unknown): string {
	if (rawOutput === undefined || rawOutput === null) return "";
	if (typeof rawOutput === "string") return rawOutput;
	if (typeof rawOutput === "object" && "output" in rawOutput && typeof rawOutput.output === "string") return rawOutput.output;
	return "";
}

function cap(text: string): string {
	return text.length > MAX_OUTPUT_CHARS ? `${text.slice(0, MAX_OUTPUT_CHARS)}\n… (output truncated)` : text;
}

function toDetailDiff(diff: AcpToolDiff): AcpToolDetails["diffs"][number] {
	return { path: diff.path, diff: generateDiffString(diff.oldText ?? "", diff.newText).diff, created: diff.oldText === null };
}

/** The tool result Pi stores: text for any model that later reads the transcript, details for the renderers. */
export function toToolResult(state: AcpToolState, partial = false) {
	const output = cap(state.content.text.join("\n") || rawOutputText(state.rawOutput));
	const details: AcpToolDetails = {
		status: state.status ?? "unknown",
		diffs: state.content.diffs.map(toDetailDiff),
		output,
		terminals: state.content.terminals,
	};
	const lines = [
		...state.content.diffs.map((diff) => `${diff.oldText === null ? "created" : "edited"} ${diff.path}`),
		...(output ? [output] : []),
	];
	const failed = !partial && state.status === "failed";
	const text = lines.join("\n") || (failed ? "The agent reported this tool call as failed." : "Done.");
	return { content: [{ type: "text" as const, text }], details, ...(failed ? { isError: true } : {}) };
}

type Theme = Parameters<NonNullable<ToolDefinition["renderCall"]>>[1];

function clip(text: string, expanded: boolean, theme: Theme): string {
	const lines = text.split("\n");
	if (expanded || lines.length <= COLLAPSED_LINES) return text;
	const hidden = lines.length - COLLAPSED_LINES;
	return `${lines.slice(0, COLLAPSED_LINES).join("\n")}\n${theme.fg("muted", `… ${hidden} more lines (expand to see all)`)}`;
}

export function createAcpToolDefinition(awaitTool: AwaitTool) {
	const definition: ToolDefinition = {
		name: ACP_TOOL_NAME,
		label: "ACP agent tool",
		description: "Shows a tool call that an ACP agent ran itself. Never call this tool.",
		parameters: Type.Object({}, { additionalProperties: true }),
		prepareLoadout: () => ({ hiddenDeclarations: [ACP_TOOL_NAME] }),
		async execute(toolCallId, _params, signal, onUpdate) {
			const final = await awaitTool(toolCallId, signal, (state) => onUpdate?.(toToolResult(state, true)));
			return toToolResult(final);
		},
		renderCall(rawArgs, theme, context) {
			const args = rawArgs as AcpToolArgs;
			const kind = args.kind && args.kind !== "other" ? theme.fg("muted", `${args.kind} `) : "";
			let text = `${kind}${theme.fg("toolTitle", theme.bold(args.title || "tool"))}`;
			const paths = (args.locations ?? []).map((location) => (location.line ? `${location.path}:${location.line}` : location.path));
			const shownPaths = paths.filter((path) => !args.title?.includes(path));
			if (shownPaths.length > 0) text += ` ${theme.fg("muted", shownPaths.join(", "))}`;
			if (context.expanded && args.input !== undefined) {
				text += `\n${theme.fg("muted", JSON.stringify(args.input, null, 2))}`;
			}
			return new Text(text, 0, 0);
		},
		renderResult(result, { expanded, isPartial }, theme) {
			const details = result.details as AcpToolDetails | undefined;
			if (!details) return new Text("", 0, 0);
			const parts: string[] = [];
			for (const diff of details.diffs) {
				parts.push(theme.fg("muted", `${diff.created ? "created" : "edited"} ${diff.path}`));
				if (diff.diff) parts.push(clip(renderDiff(diff.diff), expanded, theme));
			}
			if (details.output) parts.push(theme.fg("toolOutput", clip(details.output.trimEnd(), expanded, theme)));
			if (!isPartial && details.status === "failed" && parts.length === 0) parts.push(theme.fg("error", "failed"));
			return new Text(parts.join("\n"), 0, 0);
		},
	};
	return definition;
}
