import { test } from 'node:test';
import { strict as assert } from 'node:assert';
import { mergeRecentEvents } from '../../frontend/src/recentEvents.ts';

test('new RSS events appear first without a full reload', () => {
  assert.deepEqual(mergeRecentEvents([{ id: 'manual' }], [{ id: 'rss' }, { id: 'manual' }]).map(x => x.id), ['rss', 'manual']);
});
test('previously paginated events are preserved', () => {
  assert.deepEqual(mergeRecentEvents([{ id: 'older' }, { id: 'oldest' }], [{ id: 'newer' }]).map(x => x.id), ['newer', 'older', 'oldest']);
});
test('updated versions replace stale copies', () => {
  assert.deepEqual(mergeRecentEvents([{ id: 'event-1', version: 1 }], [{ id: 'event-1', version: 2 }]), [{ id: 'event-1', version: 2 }]);
});
test('no duplicates on consecutive polls', () => {
  const first = mergeRecentEvents([{ id: 'old' }], [{ id: 'new' }, { id: 'old' }]);
  assert.deepEqual(mergeRecentEvents(first, [{ id: 'new' }, { id: 'old' }]), first);
});
