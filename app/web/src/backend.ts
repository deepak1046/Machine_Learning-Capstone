import { useEffect, useState } from "react";
import { ApiUnreachable, getFamilies, getModels } from "./api";
import type { Families, ModelInfo } from "./types";

export type Connection = "connecting" | "online" | "offline";

const LOADING_MS = 2500; // the neural network loads in the background after the API starts
const IDLE_MS = 5000; // connection check once everything is loaded
const RETRY_MS = 4000; // while the backend cannot be reached
const MAX_LOADING_CHECKS = 60;

const sameModels = (a: ModelInfo[], b: ModelInfo[]) => JSON.stringify(a) === JSON.stringify(b);

/** Loads the model list and class families, then keeps checking that the backend is still reachable. */
export function useBackend() {
  const [models, setModels] = useState<ModelInfo[] | null>(null);
  const [families, setFamilies] = useState<Families>({});
  const [connection, setConnection] = useState<Connection>("connecting");
  const [error, setError] = useState<string | null>(null);
  const [unreachable, setUnreachable] = useState(false);

  useEffect(() => {
    let cancelled = false;
    let timer = 0;
    let loadingChecks = 0;
    let haveFamilies = false;

    const check = async () => {
      let delay = IDLE_MS;
      try {
        const list = await getModels();
        if (cancelled) return;
        setModels((prev) => (prev && sameModels(prev, list) ? prev : list));
        setConnection("online");
        setError(null);
        setUnreachable(false);
        if (!haveFamilies) {
          const groups = await getFamilies();
          if (cancelled) return;
          setFamilies(groups);
          haveFamilies = true;
        }
        if (list.some((m) => !m.loaded) && ++loadingChecks < MAX_LOADING_CHECKS) delay = LOADING_MS;
      } catch (e) {
        if (cancelled) return;
        setConnection("offline");
        setError(e instanceof Error ? e.message : String(e));
        setUnreachable(e instanceof ApiUnreachable);
        delay = RETRY_MS;
      }
      timer = window.setTimeout(check, delay);
    };

    void check();
    return () => {
      cancelled = true;
      window.clearTimeout(timer);
    };
  }, []);

  return { models, families, connection, error, unreachable };
}
