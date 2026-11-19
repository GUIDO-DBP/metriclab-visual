export type Page = 'home' | 'classify' | 'evaluation' | 'explorer' | 'dataset' | 'metric'
export type ModelKey = 'gini' | 'entropy' | 'forest' | 'xgboost' | 'lightgbm'
export const HSV_ID = 'hsv128-edge16-v1'
export const RESNET_ID = 'torchvision-resnet18-imagenet1k-v1-avgpool512-fp32'
export type DescriptorInfo = { id: string; name: string; dimensions: number; available: boolean; reason: string; device?: string }
export type GalleryImage = {id: number; category: string; url: string; split?: string; distance?: number}
export type Criterion = 'gini' | 'entropy'
export type ModelStatus = { key: string; name: string; family: string; primary: boolean; available: boolean; reason: string }
export type Status = {
  ready: boolean; classification_ready: boolean; metric_ready: boolean; dataset: string; kind: string;
  indexed: number; queries: number; categories: string[]; dimensions: number; descriptor: string; models: ModelStatus[];
  error?: string; model_error?: string; metric_error?: string; run_id?: string;
}
export type Sample = { id: number; category: string; url: string }
export type DatasetInfo = Status & {
  samples: Sample[]; counts: Record<string, number>; class_counts: Record<string, Record<string, number>>;
  max_per_class: number; skipped: number; total_processed: number; seed: number; query_fraction: number;
  fingerprint: string; image_size: number; imbalance: { min_train: number; max_train: number; ratio: number };
}
export type NodeClass = { category: string; proportion: number; weight: number }
export type DecisionNode = {
  id: number; node?: number; left: number; right: number; feature: number; feature_name: string;
  threshold: number | null; impurity: number; samples: number; weighted_samples: number; leaf: boolean;
  classes: NodeClass[]; display_depth?: number; value?: number | null; direction?: 'left' | 'right' | null;
}
export type ClassifierTree = {
  key: string; criterion: string; tree_index: number; tree_count: number; total_nodes: number;
  total_depth: number; root: number; nodes: DecisionNode[];
}
export type ModelInfo = {
  key: string; name: string; n_classes: number; dimensions: number; tree_count?: number;
  explorable_trees?: number; boosting_rounds?: number; criterion?: string;
  metadata: { params?: Record<string, unknown>; versions?: Record<string, string>; run_id?: string; reused?: boolean };
}
export type Importance = {
  key: string; measure: string; note: string; groups: Record<string,number>;
  features: { index: number; name: string; value: number; raw_value: number; group: string }[];
}
export type Prediction = {
  key: ModelKey; name: string; category: string; inference_ms: number; family: string;
  probabilities: { category: string; value: number }[]; probability_label: string; decisions?: DecisionNode[];
  examples: GalleryImage[]; descriptor_id: string;
}
export type Classification = {
  predictions: Prediction[]; descriptor: number[]; unavailable: ModelStatus[]; agreement: Record<string, number>;
  unanimous: boolean; vector_sha256: string; dataset: string; run_id: string;
  descriptor_id: string; dimensions: number; extraction: {device: string; cache_hit: boolean; extraction_ms: number}; total_ms: number;
}
export type Metric = {
  key: string; model: string; accuracy: number; balanced_accuracy: number; precision_macro: number;
  recall_macro: number; f1_macro: number; train_s: number | null; inference_ms?: number;
  inference_samples?: number; reused: boolean; confusion: number[][];
}
export type Experiments = {
  dataset?: string;
  metrics: Metric[]; secondary_metrics?: Metric[]; classes?: string[]; protocol?: string; timing_protocol?: string;
  message?: string; n_test?: number; n_train?: number; n_classes?: number; fingerprint?: string; run_id?: string;
  versions?: Record<string, string>;
  evaluation_scope?: string; descriptor_id?: string;
  cv_metrics?: {key: ModelKey; mean: Record<string,number>; std: Record<string,number>}[];
  internal_confirmation?: Record<string,{accuracy:number;balanced_accuracy:number;f1_macro:number}>;
  tuning_selection?: {candidate:number;cv_mean:{accuracy:number;balanced_accuracy:number;f1_macro:number}};
}
export type Neighbor = { id: number; distance: number; category: string; url: string }
export type Run = {
  algorithm: number; name: string; query_ms: number; distances: number; avoided_pct: number;
  build_ms: number; build_distances: number; memory_bytes: number; oracle_match: boolean; neighbors: Neighbor[];
}
export type TraceEvent = {
  step: number; kind: string; node: number; object: number; dist: number; bound: number;
  radius: number; tau: number | null; depth: number; reason: string;
}
export type SearchResult = { runs: Run[]; trace: TraceEvent[]; descriptor: number[] }
export type MetricNode = { id: number; left: number; right: number; pivot: number; radius: number; leaf: boolean }
