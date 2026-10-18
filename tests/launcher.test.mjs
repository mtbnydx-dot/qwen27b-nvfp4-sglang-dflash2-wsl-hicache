import test from 'node:test';
import assert from 'node:assert/strict';
import path from 'node:path';
import {dockerArgs, profile, serverArgs} from '../scripts/launch.mjs';

const modelRoot = path.resolve('example-models');
const value = (args, key) => args[args.indexOf(key) + 1];

test('exact qualified serving flags, not a private manager preset', () => {
  const args = serverArgs();
  for (const [key, expected] of Object.entries({
    '--context-length': '262144', '--max-total-tokens': '262272',
    '--max-running-requests': '6', '--speculative-algorithm': 'DFLASH',
    '--speculative-num-draft-tokens': '3', '--speculative-draft-model-quantization': 'fp8',
    '--mamba-max-states-per-path': '1', '--hicache-size': '32',
    '--hicache-io-backend': 'direct', '--hicache-mem-layout': 'page_first_direct',
    '--hicache-write-policy': 'write_back', '--kv-cache-dtype': 'fp8_e4m3',
    '--mamba-ssm-dtype': 'bfloat16', '--attention-backend': 'flashinfer',
  })) assert.equal(value(args, key), expected, key);
  assert.deepEqual(JSON.parse(value(args, '--default-chat-template-kwargs')), {enable_thinking: true});
  assert.ok(args.includes('--enable-hierarchical-cache'));
  assert.ok(args.includes('--enable-cache-report'));
  assert.equal(profile.minimumFreeMiBForQualification, 2560);
});

test('only loopback, readonly weights, offline credentials-free runtime', () => {
  const args = dockerArgs({modelsDir: modelRoot});
  assert.equal(value(args, '-p'), '127.0.0.1:8000:8000');
  assert.equal(value(args, '--pull'), 'never');
  assert.ok(args.includes(`type=bind,source=${modelRoot},target=/models,readonly`));
  assert.ok(args.includes('SGLANG_WSL_HICACHE_PINNED_ALLOCATOR=1'));
  assert.ok(args.includes('HF_HUB_OFFLINE=1'));
  assert.ok(!args.includes('--privileged'));
  assert.ok(!args.includes('--network=host'));
  assert.ok(!args.some(arg => /HF_TOKEN|HUGGING_FACE_HUB_TOKEN/.test(arg)));
});

test('bounded arguments, spaces remain one argument, no shell injection', () => {
  const dir = path.join(modelRoot, 'model files');
  const args = dockerArgs({modelsDir: dir, port: 18000, name: 'qwen-trial'});
  assert.equal(value(args, '-p'), '127.0.0.1:18000:8000');
  assert.ok(args.includes(`type=bind,source=${dir},target=/models,readonly`));
  for (const port of [0, 80, 65536, 1.2, NaN])
    assert.throws(() => dockerArgs({modelsDir: dir, port}));
  for (const name of ['-bad', 'a b', 'name;command', ''])
    assert.throws(() => dockerArgs({modelsDir: dir, name}));
  for (const modelsDir of ['', 'relative-path', modelRoot + ',readonly=false'])
    assert.throws(() => dockerArgs({modelsDir}));
});
