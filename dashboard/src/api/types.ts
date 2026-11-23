// TypeScript mirrors of the Pydantic/JSON shapes returned by /admin/*.
// Keep these in sync with docs/api-contract.md — that doc is the source of truth.

export interface Stats {
  window: string;
  requests: number;
  cache: {
    hit_rate: number;
    exact: number;
    semantic: number;
    misses: number;
    bypassed: number;
  };
  tokens: { total: number; served_from_cache: number };
  saved: {
    tokens: number;
    basis: string;
    note?: string;
    modeled_usd?: number;
    inference_seconds?: number;
  };
  latency: {
    p50: number;
    p95: number;
    p99: number;
    hit_p95: number;
    miss_p95: number;
  };
  rate_limit: {
    rejected: number;
    rejections_by_key: { api_key_id: number; count: number }[];
  };
  by_model: {
    model: string;
    requests: number;
    tokens: number;
    cache_hit_rate: number;
    p95: number;
  }[];
}

export interface TimeseriesPoint {
  ts: string;
  requests: number;
  hits: number;
  hit_rate: number;
  tokens: number;
  avg_latency: number;
  p95: number;
}

export interface Timeseries {
  window: string;
  bucket_minutes: number;
  series: TimeseriesPoint[];
}

export interface CacheEntry {
  prompt_hash: string;
  preview: string | null;
  model: string;
  hits: number;
  best_similarity: number | null;
}

export type Window = "1h" | "24h" | "7d";
