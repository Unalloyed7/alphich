/** Human-readable representations of server-owned analysis fields. No inferred facts. */
import type { Claim } from './analysis';

export interface RelevanceReason {
  explanation?: unknown;
  reason?: unknown;
  label?: unknown;
  code?: unknown;
  [key: string]: unknown;
}

export function relevanceDescription(value: RelevanceReason): string | null {
  // The personalization selector returns `explanation`, not `reason`.
  for (const candidate of [value.explanation, value.reason, value.label]) {
    if (typeof candidate === 'string' && candidate.trim()) return candidate.trim();
  }
  if (typeof value.code === 'string' && value.code.trim()) {
    return `Основание отбора: ${value.code.trim()}`;
  }
  return null;
}

export function claimReferenceTexts(ids: string[], claims: Claim[]): string[] {
  const byId = new Map(claims.map(claim => [claim.id, claim.text]));
  return ids.map(id => byId.get(id) ?? `Тезис ${id} не найден в карточке`);
}
