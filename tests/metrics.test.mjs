import test from 'node:test';
import assert from 'node:assert/strict';
import {parseCounters} from '../scripts/metrics.mjs';

const fresh = `# TYPE sglang:cached_tokens_total counter
# TYPE sglang:hicache_backup_bytes_total counter
# TYPE sglang:load_back_bytes_total counter
sglang:num_running_reqs{model_name="qwen"} 0
sglang:num_queue_reqs{model_name="qwen"} 0
`;

test('fresh host counters are zero, not missing-instrumentation failures', () => {
  assert.deepEqual(parseCounters(fresh), {hostHits: 0, backupBytes: 0, loadBytes: 0, running: 0, queued: 0});
});
test('GPU hits do not count as host reuse; scientific byte notation works', () => {
  const result = parseCounters(fresh + `sglang:cached_tokens_total{cache_source="device"} 90000
sglang:cached_tokens_total{cache_source="host"} 27706
sglang:hicache_backup_bytes_total{pool="kv"} 1.5e9
sglang:load_back_bytes_total{pool="kv"} 1.348472832e9
`);
  assert.equal(result.hostHits, 27706);
  assert.equal(result.backupBytes, 1500000000);
  assert.equal(result.loadBytes, 1348472832);
});
test('missing metrics fail closed rather than silently claiming idleness', () => {
  assert.throws(() => parseCounters(''));
  assert.throws(() => parseCounters(fresh.replace(/sglang:num_queue_reqs.*\n/, '')));
});
