// Standalone launcher: no private manager, shell interpolation or service replacement.
import fs from 'node:fs';
import path from 'node:path';
import {fileURLToPath} from 'node:url';
import {spawnSync} from 'node:child_process';

export const profile = JSON.parse(fs.readFileSync(
  new URL('../profiles/rtx-pro-5000-48gb.json', import.meta.url), 'utf8'));

export function serverArgs(p = profile) {
  return ['-m', 'sglang.launch_server',
    '--model-path', '/models/' + p.modelDirectory,
    '--served-model-name', p.servedModel,
    '--speculative-algorithm', 'DFLASH',
    '--speculative-draft-model-path', '/models/' + p.draftDirectory,
    '--speculative-num-draft-tokens', String(p.draftBlockSize),
    '--speculative-draft-model-quantization', p.draftQuantization,
    '--speculative-draft-attention-backend', p.draftAttentionBackend,
    '--attention-backend', p.attentionBackend,
    '--kv-cache-dtype', 'fp8_e4m3', '--mamba-ssm-dtype', 'bfloat16',
    '--mamba-radix-cache-strategy', 'extra_buffer_lazy',
    '--max-mamba-cache-size', String(p.maxMambaCacheSize),
    '--mamba-max-states-per-path', String(p.mambaMaxStatesPerPath),
    '--max-running-requests', String(p.maxRunningRequests),
    '--max-total-tokens', String(p.maxTotalTokens),
    '--mem-fraction-static', String(p.memFractionStatic),
    '--context-length', String(p.contextLength),
    '--chunked-prefill-size', String(p.chunkedPrefillSize),
    '--prefill-decode-interval', String(p.prefillDecodeInterval),
    '--enable-hierarchical-cache', '--hicache-size', String(p.hicacheSizeGB),
    '--hicache-io-backend', 'direct', '--hicache-mem-layout', 'page_first_direct',
    '--hicache-write-policy', 'write_back',
    '--cuda-graph-max-bs-decode', String(p.maxRunningRequests),
    '--cuda-graph-backend-prefill', 'disabled', '--mm-feature-transport', 'cpu',
    '--reasoning-parser', 'qwen3', '--tool-call-parser', 'qwen3_coder',
    '--default-chat-template-kwargs', '{"enable_thinking":true}',
    '--sampling-defaults', 'model', '--enable-metrics', '--enable-cache-report',
    '--watchdog-timeout', '1800', '--host', '0.0.0.0', '--port', '8000'];
}

export function dockerArgs({modelsDir, port = 8000, name = 'qwen27b-hicache'} = {}) {
  if (!modelsDir || !path.isAbsolute(modelsDir) || /[,\r\n]/.test(modelsDir))
    throw Error('QWEN_MODELS_DIR must be an absolute directory without commas/newlines');
  if (!Number.isInteger(port) || port < 1024 || port > 65535)
    throw Error('QWEN_PORT must be an integer from 1024 to 65535');
  if (!/^[a-zA-Z0-9][a-zA-Z0-9_.-]{0,80}$/.test(name))
    throw Error('Invalid QWEN_CONTAINER name');
  return ['run', '-d', '--pull', 'never', '--platform', profile.platform,
    '--name', name, '--restart', 'on-failure:3',
    '--label', 'ai.qwen.hicache.recipe=2026-10-18',
    '--gpus', 'device=0', '--shm-size', '16g',
    '--security-opt', 'no-new-privileges:true', '-p', `127.0.0.1:${port}:8000`,
    '--mount', `type=bind,source=${modelsDir},target=/models,readonly`,
    '--mount', 'type=volume,source=qwen27b-hicache-compile-cache,target=/root/.cache',
    '-e', 'HF_HUB_OFFLINE=1', '-e', 'TRANSFORMERS_OFFLINE=1',
    '-e', 'SGLANG_WSL_HICACHE_PINNED_ALLOCATOR=1',
    '-e', `SGLANG_VLM_CACHE_SIZE_MB=${profile.vlmCacheMB}`,
    ...(profile.mambaSkipDecodeLock ? ['-e', 'SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=1'] : []),
    '--health-cmd', 'curl -fsS http://127.0.0.1:8000/health || exit 1',
    '--health-interval', '30s', '--health-timeout', '5s', '--health-retries', '3',
    '--health-start-period', '20m', '--entrypoint', 'python3', profile.image,
    ...serverArgs()];
}

function docker(args, inherit = false) {
  const result = spawnSync(process.platform === 'win32' ? 'docker.exe' : 'docker', args,
    {encoding: 'utf8', shell: false, windowsHide: true,
      stdio: inherit ? 'inherit' : 'pipe', timeout: 120000});
  if (result.error) throw result.error;
  if (result.status !== 0) throw Error(`Docker failed (${result.status}). Inspect Docker locally.`);
  return result.stdout?.trim();
}

function main() {
  const flags = process.argv.slice(2);
  if (flags.some(flag => !['--start', '--help'].includes(flag))) throw Error('Unknown flag');
  if (flags.includes('--help')) {
    console.log('Set QWEN_MODELS_DIR (absolute), optionally QWEN_PORT / QWEN_CONTAINER.');
    console.log('node scripts/launch.mjs         # dry run, no Docker changes');
    console.log('node scripts/launch.mjs --start # start a NEW container; never stop/replace one');
    return;
  }
  const modelsDir = process.env.QWEN_MODELS_DIR;
  const name = process.env.QWEN_CONTAINER || 'qwen27b-hicache';
  const args = dockerArgs({modelsDir, name, port: Number(process.env.QWEN_PORT || 8000)});
  if (!flags.includes('--start')) {
    console.log(JSON.stringify({dryRun: true, executable: 'docker', args}, null, 2));
    return;
  }
  // Fail closed on a missing image, Docker error or existing container. No stop/rm/rename.
  const names = docker(['ps', '-a', '--format', '{{.Names}}']).split('\n');
  if (names.includes(name)) throw Error('Container already exists; no changes made');
  const image = JSON.parse(docker(['image', 'inspect', profile.image]))[0];
  if (image.Os !== 'linux' || image.Architecture !== 'amd64' ||
      image.Config?.Labels?.['ai.qwen.hicache.patch'] !== 'sgl-project/sglang#40571' ||
      image.Config?.Labels?.['ai.qwen.hicache.allocator'] !== 'wsl-owned-pinned-v1')
    throw Error('Build the patched Linux amd64 image first');
  for (const dir of [profile.modelDirectory, profile.draftDirectory]) {
    if (!fs.statSync(path.join(modelsDir, dir, 'config.json')).isFile())
      throw Error('Model config missing');
  }
  const info = JSON.parse(docker(['info', '--format', '{{json .}}']));
  if (info.OSType !== 'linux' || info.MemTotal < 96 * 2**30)
    throw Error('This profile requires a Linux Docker engine with at least 96 GiB RAM available to it');
  console.log('Starting only the new Qwen container. Readiness and GPU load tests are still required.');
  docker(args, true);
}

if (process.argv[1] && path.resolve(process.argv[1]) === fileURLToPath(import.meta.url)) {
  try { main(); } catch (error) { console.error(error.message); process.exitCode = 1; }
}
