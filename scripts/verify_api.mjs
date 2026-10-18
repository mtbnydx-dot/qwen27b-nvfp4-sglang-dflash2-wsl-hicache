// Synthetic probes only. --host-cache deliberately churns L1; use a maintenance window.
import {randomUUID} from 'node:crypto';
import {profile} from './launch.mjs';
import {parseCounters} from './metrics.mjs';

const flags = process.argv.slice(2);
if (flags.some(x => !['--host-cache', '--allow-load'].includes(x))) throw Error('Unknown flag');
if (flags.includes('--host-cache') && !flags.includes('--allow-load'))
  throw Error('--host-cache needs --allow-load; it sends up to 21 long requests');
const base = new URL(process.env.QWEN_BASE_URL || 'http://127.0.0.1:8000');
if (!['http:', 'https:'].includes(base.protocol) || base.username || base.password ||
    base.search || base.hash || base.pathname !== '/')
  throw Error('QWEN_BASE_URL must be an origin without credentials/path/query');
if (base.protocol !== 'https:' && !['localhost', '127.0.0.1', '[::1]'].includes(base.hostname))
  throw Error('Use HTTPS for a non-loopback endpoint');
const headers = {'Content-Type': 'application/json',
  ...(process.env.QWEN_API_KEY ? {Authorization: 'Bearer ' + process.env.QWEN_API_KEY} : {})};
const pause = ms => new Promise(resolve => setTimeout(resolve, ms));

async function request(route, body, timeout = 180000) {
  const response = await fetch(new URL(route, base), {
    method: body === undefined ? 'GET' : 'POST', headers,
    body: body === undefined ? undefined : JSON.stringify(body),
    redirect: 'error', signal: AbortSignal.timeout(timeout),
  });
  if (!response.ok) throw Error(`HTTP ${response.status} on ${route}`);
  return response;
}

async function counters() {
  const text = await (await request('/metrics', undefined, 10000)).text();
  return parseCounters(text);
}

async function chat(prompt, maxTokens = 64) {
  const started = performance.now();
  const data = await (await request('/v1/chat/completions', {
    model: profile.servedModel, messages: [{role: 'user', content: prompt}],
    temperature: 0, max_tokens: maxTokens, chat_template_kwargs: {enable_thinking: false},
  })).json();
  if (data.model !== profile.servedModel) throw Error('Unexpected response model');
  return {data, seconds: +((performance.now() - started) / 1000).toFixed(3)};
}

async function main() {
  const models = await (await request('/v1/models')).json();
  if (!models.data?.some(x => x.id === profile.servedModel)) throw Error('Expected model not advertised');
  const info = await (await request('/server_info')).json();
  const args = info.server_args || info;
  for (const [name, expected] of Object.entries({context_length: 262144,
    max_running_requests: 6, speculative_num_draft_tokens: 3, speculative_algorithm: 'DFLASH',
    enable_hierarchical_cache: true, hicache_size: 32, mamba_max_states_per_path: 1}))
    if (args[name] !== expected) throw Error(`Runtime setting mismatch: ${name}`);
  const smoke = await chat('计算 13 乘以 17，只输出数字。');
  if (smoke.data.choices?.[0]?.message?.content?.trim() !== '221')
    throw Error('Synthetic arithmetic smoke failed');
  console.log(JSON.stringify({event: 'SMOKE_PASS', model: profile.servedModel, seconds: smoke.seconds,
    note: 'Short non-streamed smoke; not comprehensive quality or load validation'}));
  if (!flags.includes('--host-cache')) return;

  const before = await counters();
  if (before.running || before.queued) throw Error('Engine is busy; no cache churn started');
  // A fresh namespace prevents accidentally reusing a previous test seed.
  const run = randomUUID();
  async function document(i, label) {
    const code = 'CACHEOK' + (88000 + i);
    const prompt = `Run ${run} Document_ID_${i}. Verification code: ${code}.\n` +
      Array.from({length: 1800}, (_, j) => `background ${i} row ${j}: alpha beta gamma delta.\n`).join('') +
      '\nReturn only this document\'s verification code, without explanation.';
    const result = await chat(prompt, 32);
    if (result.data.choices?.[0]?.message?.content?.trim() !== code)
      throw Error('Replayed prefix answer mismatch');
    console.log(JSON.stringify({event: 'CACHE_REQUEST', label, seconds: result.seconds,
      promptTokens: result.data.usage?.prompt_tokens, outputTokens: result.data.usage?.completion_tokens}));
  }
  await document(0, 'cold');
  for (let i = 1; i <= 18; i++) {
    await document(i, 'filler-' + i);
    if (i < 12 || i % 6) continue;
    // Count reload and host hits across the seed replay, not unrelated filler traffic.
    const replayBefore = await counters();
    await document(0, 'seed-replay-after-' + i);
    await pause(1100);
    const after = await counters();
    const delta = {backupBytes: after.backupBytes - before.backupBytes,
      loadBytes: after.loadBytes - replayBefore.loadBytes,
      hostHitTokens: after.hostHits - replayBefore.hostHits};
    if (delta.backupBytes > 0 && delta.loadBytes > 0 && delta.hostHitTokens > 0) {
      console.log(JSON.stringify({event: 'HOST_ROUNDTRIP_PASS', ...delta}));
      return;
    }
  }
  throw Error('No observed seed host-cache reuse; allocation or faster replay alone is insufficient');
}

main().catch(error => { console.error(error.message); process.exitCode = 1; });
