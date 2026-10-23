/** Labels change presentation only; analysis and grades remain server-owned. */
export function bankLabel(status: string): string {
  return ({ found: 'Есть позиция Альфа-Банка', historical_only: 'Только историческая позиция банка', conflicting: 'Позиции банка противоречат друг другу', not_found: 'Позиция банка не найдена', unavailable: 'Позиция банка недоступна' } as Record<string, string>)[status] ?? 'Подтверждённой позиции банка нет';
}
export function analysisLabel(status?: string): string {
  return ({ ready: 'Анализ готов', completed: 'Завершено', succeeded: 'Завершено', verified: 'Проверено системой', passed: 'Проверка пройдена', valid: 'Проверка пройдена', insufficient_evidence: 'Недостаточно подтверждений', failed: 'Ошибка обработки', partial: 'Частичный результат', limited: 'Есть ограничения', running: 'В работе', queued: 'В очереди', rejected: 'Не прошёл проверку' } as Record<string, string>)[status ?? ''] ?? (status ? `Статус: ${status}` : 'Результат не указан');
}
export function limitationLabel(value: string): string {
  return ({ bounded_candidate_pool: 'Отбор выполняется среди ограниченного набора доступных публикаций.', machine_relevance_not_human_review: 'Значимость оценивает система. Экспертная проверка человеком не проводилась.', news_dense_index_unavailable_or_pending: 'Семантический индекс пока недоступен: используется поиск по ключевым словам.' } as Record<string, string>)[value] ?? value;
}
