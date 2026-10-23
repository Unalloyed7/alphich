import { useEffect, useState, type FormEvent } from 'react';
import { analysisApi, ApiError } from './api';
import type { FeedbackValue, InsightFeedback } from './analysis';
import { Icon } from './Icon';
const choices: { value: FeedbackValue; title: string; hint: string }[] = [
  { value: 'useful', title: 'Полезно', hint: 'Помогает разобраться во влиянии события' },
  { value: 'not_relevant', title: 'Не для моей компании', hint: 'Связь с бизнесом не подходит' },
  { value: 'needs_review', title: 'Есть неточность', hint: 'Факты или вывод нужно проверить' },
];
export function FeedbackPanel({ insightId, token }: { insightId: string; token: string }) {
  const [saved, setSaved] = useState<InsightFeedback | null>(null);
  const [value, setValue] = useState<FeedbackValue | null>(null);
  const [comment, setComment] = useState('');
  const [loading, setLoading] = useState(true);
  const [available, setAvailable] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState<string | null>(null);
  useEffect(() => {
    let active = true;
    void analysisApi.feedback(insightId, token).then(result => {
      if (active) { setSaved(result.feedback); setValue(result.feedback?.value ?? null); setComment(result.feedback?.comment ?? ''); }
    }).catch(error => {
      if (!active) return;
      if (error instanceof ApiError && error.status === 404) { setAvailable(false); setError('Оценка анализа пока недоступна на этом сервере.'); }
      else setError('Не удалось загрузить вашу оценку. Можно повторить сохранение.');
    }).finally(() => { if (active) setLoading(false); });
    return () => { active = false; };
  }, [insightId, token]);
  const submit = async (event: FormEvent) => {
    event.preventDefault(); if (!value) return;
    setBusy(true); setError(null);
    try { setSaved((await analysisApi.saveFeedback(insightId, value, comment.trim(), token)).feedback); }
    catch (error) { setError(error instanceof Error ? error.message : 'Не удалось сохранить оценку. Попробуйте ещё раз.'); }
    finally { setBusy(false); }
  };
  const changed = saved?.value !== value || (saved?.comment ?? '') !== comment.trim();
  return <section className="feedback-panel" aria-label="Обратная связь по аналитике"><div className="feedback-heading"><Icon name="thumbs" size={19}/><div><h4>Оценка анализа</h4><p>Укажите, относится ли анализ к компании и есть ли в нём неточности.</p></div></div>{loading ? <p role="status" className="insight-muted">Загружаем вашу оценку…</p> : <form onSubmit={event => void submit(event)}><fieldset disabled={!available || busy}><legend className="sr-only">Ваша оценка анализа</legend><div className="feedback-choices">{choices.map(choice => <label key={choice.value} className={value === choice.value ? 'active' : ''}><input type="radio" name={`feedback-${insightId}`} value={choice.value} checked={value === choice.value} onChange={() => setValue(choice.value)}/><span>{choice.title}</span></label>)}</div>{value && <><p className="feedback-hint">{choices.find(choice => choice.value === value)?.hint}</p><label className="feedback-comment-label" htmlFor={`feedback-comment-${insightId}`}>Комментарий <span>необязательно</span></label><textarea id={`feedback-comment-${insightId}`} rows={2} maxLength={500} value={comment} onChange={event => setComment(event.target.value)} placeholder="Например, у компании другие условия договора…"/><div className="feedback-footer"><small>{comment.length} / 500</small><button type="submit" className="secondary-button" disabled={!changed}>{busy ? 'Сохраняем…' : 'Сохранить оценку'}</button></div></>}</fieldset></form>}{saved && !changed && <p className="feedback-saved" role="status"><Icon name="check" size={15}/> Оценка сохранена для проверки качества.</p>}{error && <p className="insight-error" role="alert">{error}</p>}</section>;
}
