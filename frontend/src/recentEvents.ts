/** Put newly arrived RSS/manual events first, preserving older loaded pages. */
export function mergeRecentEvents<T extends { id: string }>(loaded: T[], latest: T[]): T[] {
  const seen = new Set<string>();
  const merged: T[] = [];
  for (const event of [...latest, ...loaded]) {
    if (seen.has(event.id)) continue;
    seen.add(event.id);
    merged.push(event);
  }
  return merged;
}
