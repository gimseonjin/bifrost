import i18n from "@/lib/i18n";
import type { VectorStoreInfo, VectorStoreStatus } from "@/lib/types/config";
import type { WarpBackfillJob, WarpConfig, WarpConfigInput } from "@/lib/types/warp";
/**
 * Validation helpers for the Warp settings form.
 *
 * These live outside the component because the form itself has no render
 * harness in this repo, and a rule that cannot be tested is a rule that quietly
 * stops holding.
 */

/**
 * Rejects a numeric field that was cleared rather than filled in.
 *
 * `valueAsNumber: true` hands React Hook Form `NaN` for an empty input, and RHF
 * treats the empty DOM value as "no value" and skips `min`/`max` entirely. With
 * nothing else checking, `NaN` reaches the PUT body, where `JSON.stringify`
 * writes it as `null` - so clearing the box silently sends null for a field the
 * server reads as a number.
 */
export function requireFiniteNumber(value: unknown, message: string): true | string {
	return isFiniteNumber(value) ? true : message;
}

/**
 * Narrows to a real, comparable number.
 *
 * The form parses its numeric inputs with `Number(...)`, and every comparison
 * against `NaN` is false - so a range check alone reports a non-numeric value
 * as valid. This is the guard that has to run before the range check, not after.
 */
export function isFiniteNumber(value: unknown): value is number {
	return typeof value === "number" && Number.isFinite(value);
}
/**
 * Days to keep saved chats.
 *
 * Zero is a supported value, not a missing one: the server reads 0 as "use the
 * default" (`WarpDefaultHistoryRetentionDays`), which is exactly how a config
 * that never set the field behaves. A `min: 1` rule made that documented value
 * unreachable, so once an operator had typed a number they could not get back
 * to default retention from the form at all.
 *
 * No upper bound: the per-owner conversation cap already limits the table, so
 * how long a transcript stays readable is a policy choice with no ceiling.
 */
export function validateWarpRetentionDays(value: unknown): true | string {
	if (typeof value !== "number" || !Number.isFinite(value)) return i18n.t("warp.validation.retentionRequired", { ns: "config" });
	if (!Number.isInteger(value)) return i18n.t("warp.validation.retentionWhole", { ns: "config" });
	if (value < 0) return i18n.t("warp.validation.retentionMin", { ns: "config" });
	return true;
}

/**
 * Tool-call rounds per answer, mirroring the server's 2..20 range.
 *
 * One is refused because the agent loop reserves its last step for answering:
 * a budget of one is spent on a tool call and the run ends with no answer, so
 * the server rejects it (`WarpMinMaxIterations`). Checking `< 1` here let the
 * form submit 1 and surface that as a failed save instead of an inline error.
 */
export function validateWarpMaxIterations(value: unknown): true | string {
	if (!isFiniteNumber(value) || !Number.isInteger(value) || value < 2 || value > 20)
		return i18n.t("warp.iterationsError", { ns: "config" });
	return true;
}

/**
 * The saved embedding space as one comparable string, so "did the space
 * change" is a single equality on the fields that define it. The chosen vector
 * store is one of them: a store switched to starts empty, so a run finished in
 * the old store says nothing about the new one. Uses NUL as the separator
 * because it cannot appear in a config value.
 */
export function warpSavedSpaceKey(
	config:
		| Partial<
				Pick<WarpConfig, "embedding_provider" | "embedding_model" | "embedding_dimension" | "log_vector_store_namespace" | "vector_store">
		  >
		| undefined,
): string {
	return [
		config?.embedding_provider ?? "",
		config?.embedding_model ?? "",
		config?.embedding_dimension ?? 0,
		config?.log_vector_store_namespace ?? "",
		config?.vector_store ?? "",
	].join("\u0000");
}

/** A finished backfill kept on the client, tagged with the space it ran under. */
export interface RetainedWarpBackfill {
	job: WarpBackfillJob;
	spaceKey: string;
}

/**
 * Decides whether a job that just reached a terminal status is worth keeping.
 *
 * A job belongs to the space that was configured while it ran; the server
 * refuses to change the space while a job is in flight, so any saved space
 * observed during the run is the job's space. The status endpoint hides a
 * finished job from the id-less read once the space changes, but the id-pinned
 * poll the view uses during a run does not, so a run that finished under the
 * old space can still arrive as "completed" after the new space was saved.
 * Keeping it would show a full bar for rows the new space never indexed, so it
 * is dropped unless the job's space is still the saved one. A job whose space
 * was never observed (null) is assumed to belong to the current space: the
 * only way to pin an id is to start or discover a running job, and both
 * happen under the configured space.
 */
export function retainFinishedWarpBackfill(
	job: WarpBackfillJob,
	jobSpaceKey: string | null,
	savedSpaceKey: string,
): RetainedWarpBackfill | null {
	const spaceKey = jobSpaceKey ?? savedSpaceKey;
	return spaceKey === savedSpaceKey ? { job, spaceKey } : null;
}

/**
 * The retained job, if it still describes the saved space. Checked at every
 * read rather than cleared by an effect on change: an effect fires once, and a
 * terminal status that lands after it has fired would be retained untouched.
 */
export function retainedWarpBackfillForSpace(retained: RetainedWarpBackfill | null, savedSpaceKey: string): WarpBackfillJob | null {
	return retained && retained.spaceKey === savedSpaceKey ? retained.job : null;
}
type VectorStoreList = readonly VectorStoreInfo[] | undefined;

/** The vector store picker is shown only when there is a choice to make: more than one configured store. */
export function showWarpVectorStorePicker(stores: VectorStoreList, savedChoice = ""): boolean {
	if ((stores?.length ?? 0) > 1) return true;
	// One store, but the saved choice does not reach it (a store removed from config.json, or
	// "" with no primary store): the picker is the way to choose the one that is left.
	const sole = stores?.[0];
	return !!sole && sole.status === "connected" && resolveWarpVectorStoreName(savedChoice, stores) !== sole.name;
}

/**
 * The store a stored choice names. "" is the primary vector_store, so it
 * resolves to the primary's name, or "" when no primary is configured.
 */
export function resolveWarpVectorStoreName(choice: string, stores: VectorStoreList): string {
	if (choice) return choice;
	return stores?.find((store) => store.primary)?.name ?? "";
}

/** The choice to store for a store picked by name: the primary is stored as "", any other store by its name. */
export function warpVectorStoreChoice(name: string, stores: VectorStoreList): string {
	return stores?.some((store) => store.primary && store.name === name) ? "" : name;
}

/**
 * Whether the form picks a different store from the saved one. Compared by the
 * store each choice resolves to, so a saved choice that names the primary
 * explicitly and the form's "" are the same store, not a pending switch.
 */
export function warpVectorStoreSwitchPending(choice: string, savedChoice: string, stores: VectorStoreList): boolean {
	return resolveWarpVectorStoreName(choice, stores) !== resolveWarpVectorStoreName(savedChoice, stores);
}

/**
 * The vector_store part of the save body. Sent only for a switch: an omitted
 * field keeps the stored choice, so a save that did not touch the picker (or
 * ran before the store list loaded) can never move Warp to another store.
 */
export function warpVectorStorePayload(
	choice: string,
	savedChoice: string,
	stores: VectorStoreList,
): Pick<WarpConfigInput, "vector_store"> {
	if (!warpVectorStoreSwitchPending(choice, savedChoice, stores)) return {};
	return { vector_store: warpVectorStoreChoice(resolveWarpVectorStoreName(choice, stores), stores) };
}

/** "missing" is a saved choice no longer configured in config.json. */
export type WarpVectorStoreOptionStatus = VectorStoreStatus | "missing";

export interface WarpVectorStoreOption {
	name: string;
	type: string;
	primary: boolean;
	status: WarpVectorStoreOptionStatus;
	error: string;
	selectable: boolean;
}

/**
 * The picker's options: every configured store, selectable only while
 * connected, since the server refuses a switch to a store it cannot reach. A
 * saved choice that is no longer configured is kept as an unselectable option,
 * so the picker shows what is saved instead of going blank.
 */
export function warpVectorStoreOptions(stores: VectorStoreList, savedChoice: string): WarpVectorStoreOption[] {
	const options: WarpVectorStoreOption[] = (stores ?? []).map((store) => ({
		name: store.name,
		type: store.type,
		primary: store.primary,
		status: store.status,
		error: store.error,
		selectable: store.status === "connected",
	}));
	if (savedChoice && !options.some((option) => option.name === savedChoice)) {
		options.push({ name: savedChoice, type: "", primary: false, status: "missing", error: "", selectable: false });
	}
	return options;
}

/**
 * Whether Warp will have a store it can use once the form is saved: what the
 * server reports while the saved store stays, or whether the picked store is
 * connected while a switch is pending.
 */
export function warpVectorStoreUsable(connected: boolean, choice: string, savedChoice: string, stores: VectorStoreList): boolean {
	if (!warpVectorStoreSwitchPending(choice, savedChoice, stores)) return connected;
	const name = resolveWarpVectorStoreName(choice, stores);
	return stores?.some((store) => store.name === name && store.status === "connected") ?? false;
}

export type WarpVectorStoreBadge =
	| { connected: true; name: string; type: string }
	| { connected: true; name: null; type: null }
	| { connected: false; error: string | null };

/**
 * What the status badge says. A config without connected_vector_store (a
 * server older than the store choice) still reports vector_store_connected,
 * so that is the fallback, shown without a name.
 */
export function warpVectorStoreBadge(
	config: Partial<Pick<WarpConfig, "vector_store_connected" | "connected_vector_store" | "vector_store_error">> | undefined,
): WarpVectorStoreBadge {
	const store = config?.connected_vector_store;
	if (store) return { connected: true, name: store.name, type: store.type };
	if (config?.vector_store_connected && store === undefined) return { connected: true, name: null, type: null };
	return { connected: false, error: config?.vector_store_error || null };
}

/**
 * The message of a 503 save refusal, whose body is `{reason, message}` rather
 * than the `{error: {message}}` shape getErrorMessage reads (e.g. saving an
 * enabled Warp while the chosen store is unavailable). Null for any other error.
 */
export function warpUnavailableMessage(error: unknown): string | null {
	if (!error || typeof error !== "object" || !("data" in error)) return null;
	const data = (error as { data?: unknown }).data;
	if (!data || typeof data !== "object" || !("reason" in data) || !("message" in data)) return null;
	const message = (data as { message?: unknown }).message;
	return typeof message === "string" && message ? message : null;
}