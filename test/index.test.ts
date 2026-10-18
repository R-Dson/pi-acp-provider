import test from "node:test";
import assert from "node:assert/strict";
import { mkdtempSync, mkdirSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";
import acpExtension from "../src/index.ts";

type EventHandler = (event: unknown, ctx: unknown) => void;

/** Minimal structural double of the ExtensionAPI surface the extension uses. */
function makeFakePi() {
	const providers: unknown[] = [];
	const handlers = new Map<string, EventHandler>();
	const pi = {
		registerProvider: (provider: unknown) => providers.push(provider),
		on: (event: string, handler: EventHandler) => {
			handlers.set(event, handler);
			return () => {};
		},
	};
	return { pi: pi as unknown as ExtensionAPI, providers, handlers };
}

/** Runs `fn` with HOME pointing at a temp dir holding the given settings.json. */
function withSettings(settings: unknown, fn: () => void | Promise<void>) {
	const home = mkdtempSync(join(tmpdir(), "pi-acp-"));
	mkdirSync(join(home, ".pi", "agent"), { recursive: true });
	writeFileSync(join(home, ".pi", "agent", "settings.json"), JSON.stringify(settings));
	const previous = process.env.HOME;
	process.env.HOME = home;
	return Promise.resolve(fn()).finally(() => {
		process.env.HOME = previous;
	});
}

test("agents from settings.json become acp models", () =>
	withSettings(
		{ acp: { agents: { copilot: { command: "copilot", args: ["--acp"] }, gemini: { name: "Gemini", command: "gemini" } } } },
		async () => {
			const { pi, providers, handlers } = makeFakePi();
			acpExtension(pi);
			assert.equal(providers.length, 1);
			const provider = providers[0] as {
				id: string;
				getModels: () => { id: string; name: string }[];
				auth: { apiKey?: { resolve: (input: unknown) => Promise<unknown> } };
			};
			assert.equal(provider.id, "acp");
			assert.deepEqual(
				provider.getModels().map((model) => [model.id, model.name]),
				[
					["copilot", "copilot"],
					["gemini", "Gemini"],
				],
			);
			assert.ok(await provider.auth.apiKey?.resolve({}), "ambient auth must report the provider as configured");

			handlers.get("session_start")?.({ type: "session_start", reason: "startup" }, { cwd: "/tmp", hasUI: false });
			handlers.get("session_shutdown")?.({ type: "session_shutdown", reason: "quit" }, undefined);
		},
	));

test("no configured agents warns once at startup", () =>
	withSettings({}, () => {
		const { pi, providers, handlers } = makeFakePi();
		acpExtension(pi);
		assert.deepEqual((providers[0] as { getModels: () => unknown[] }).getModels(), []);
		const notices: string[] = [];
		const ctx = { cwd: "/tmp", hasUI: true, ui: { notify: (message: string) => notices.push(message) } };
		handlers.get("session_start")?.({ type: "session_start", reason: "startup" }, ctx);
		handlers.get("session_start")?.({ type: "session_start", reason: "new" }, ctx);
		assert.equal(notices.length, 1);
		assert.match(notices[0] ?? "", /no agents configured/);
		handlers.get("session_shutdown")?.({ type: "session_shutdown", reason: "quit" }, undefined);
	}));
