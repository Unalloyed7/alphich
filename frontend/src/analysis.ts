/** Public analysis contracts from backend/app/intelligence/routes.py. */
export interface LlmStatus {
  configured: boolean;
  prices_confirmed: boolean;
  daily_limit_rub: string;
  spent_rub: string;
  reserved_rub: string;
  budget_day_utc: string;
  extraction_model: string | null;
  analysis_model: string | null;
  bank_index_present: boolean;
}

export interface Claim {
  id: string;
  kind: 'news_fact' | 'company_fact' | 'bank_view' | 'model_inference';
  text: string;
  evidence_ids: string[];
  conditions: string[];
}

export interface Insight {
  id: string;
  company_id: string;
  role: string;
  event_id: string;
  status: string;
  headline: string;
  as_of: string;
  claims: Claim[];
  relevance_reasons: Array<{ reason?: string; feature_id?: string; [key: string]: unknown }>;
  bank_view: { status: string; claim_ids: string[]; reason?: string | null };
  risks: string[];
  opportunities: string[];
  next_steps: Array<{ text: string; basis_claim_ids: string[]; conditions: string[] }>;
  missing_data: string[];
  verification: { status: string; issues?: string[] };
  limitations?: string[];
  run_id: string;
  extraction_coverage?: unknown;
}

export interface Answer extends Insight {
  insight_id: string;
  question: string;
}

export interface FeedItem {
  event_id: string;
  event_version: number;
  priority?: number;
  relevance_tier?: string;
  status?: string;
  reasons?: Array<{ reason?: string; label?: string; explanation?: string; [key: string]: unknown }>;
  relevance_reasons?: Array<{ reason?: string; label?: string; [key: string]: unknown }>;
  insight?: { id: string; status: string; headline: string; as_of: string; bank_view?: { status: string } };
}

export interface FeedResult {
  items: FeedItem[];
  limitations?: string[];
}

export interface Evidence {
  id?: string;
  kind?: string;
  text?: string;
  title?: string;
  document_id?: string;
  page?: number;
  published_at?: string;
  url?: string;
  [key: string]: unknown;
}

export interface AuditRun {
  id: string;
  status: string;
  steps: Array<{ action?: string; [key: string]: unknown }>;
  verification?: { status?: string; issues?: string[] };
  llm_calls: Array<{ model: string; status: string; cost_rub: string | null; input_tokens: number | null; output_tokens: number | null }>;
}
