// Single-GPU profile: sum labels only within this engine's metrics endpoint.
export function parseCounters(text) {
  const result = {hostHits: 0, backupBytes: 0, loadBytes: 0, running: 0, queued: 0};
  const names = {'sglang:hicache_backup_bytes_total': 'backupBytes',
    'sglang:load_back_bytes_total': 'loadBytes', 'sglang:num_running_reqs': 'running',
    'sglang:num_queue_reqs': 'queued'};
  const seen = new Set();
  for (const line of text.split('\n')) {
    const match = line.match(/^(sglang:\w+)(?:\{(.*)\})?\s+([\d.eE+-]+)$/);
    if (!match) continue;
    let key = names[match[1]];
    if (match[1] === 'sglang:cached_tokens_total' && /cache_source="host"/.test(match[2] || ''))
      key = 'hostHits';
    if (key) { result[key] += Number(match[3]); seen.add(key); }
  }
  // Prometheus labeled counters have no samples before their first increment.
  // Require instrumentation families, but interpret those not-yet-created samples as zero.
  for (const name of ['sglang:cached_tokens', 'sglang:hicache_backup_bytes', 'sglang:load_back_bytes'])
    if (!text.includes(name)) throw Error('Required cache instrumentation is missing');
  if (!seen.has('running') || !seen.has('queued')) throw Error('Scheduler gauges are missing');
  if (!Object.values(result).every(value => Number.isFinite(value) && value >= 0))
    throw Error('Invalid metric counter');
  return result;
}
