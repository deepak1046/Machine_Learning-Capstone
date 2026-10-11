import type { Classification, Explanation, Families, ModelInfo, ResultsData, StreamTick } from "./types";

export const START_COMMAND = "python -m uvicorn app.api.main:app --port 8000";

/** The request never reached the backend (it is not running, or the network is down). */
export class ApiUnreachable extends Error {
  constructor() {
    super(`Cannot reach the backend. Start it from the project root with: ${START_COMMAND}`);
    this.name = "ApiUnreachable";
  }
}

async function request<T>(url: string, init?: RequestInit): Promise<T> {
  let res: Response;
  try {
    res = await fetch(url, init);
  } catch {
    throw new ApiUnreachable();
  }
  // a dev proxy or gateway answers 502, 503 or 504 when the backend behind it is down
  if (res.status === 502 || res.status === 503 || res.status === 504) throw new ApiUnreachable();
  if (!res.ok) {
    let detail = `${res.status} ${res.statusText}`;
    try {
      const body = await res.json();
      if (typeof body.detail === "string") detail = body.detail;
      else if (Array.isArray(body.detail)) detail = body.detail.map((d: { msg: string }) => d.msg).join("; ");
    } catch {
      /* keep the status text */
    }
    throw new Error(detail);
  }
  return res.json() as Promise<T>;
}

const json = (body: unknown): RequestInit => ({
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify(body),
});

export const getModels = () => request<ModelInfo[]>("/api/models");
export const getFamilies = () => request<Families>("/api/families");
export const getResults = () => request<ResultsData>("/api/results");

export const classifySample = (body: { model: string; n: number; seed: number; classes: string[] | null }) =>
  request<Classification>("/api/classify/sample", json(body));

export function classifyUpload(model: string, file: File) {
  const form = new FormData();
  form.append("model", model);
  form.append("file", file);
  return request<Classification>("/api/classify/upload", { method: "POST", body: form });
}

export const explain = (model: string, x: (number | null)[]) =>
  request<Explanation>("/api/explain", json({ model, x }));

export const streamTick = (body: {
  model: string;
  n: number;
  seed: number;
  attack_share: number;
  vectors: string[] | null;
}) => request<StreamTick>("/api/stream/tick", json(body));
