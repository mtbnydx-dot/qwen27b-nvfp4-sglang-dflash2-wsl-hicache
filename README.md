# Qwen 27B · NVFP4 · SGLang · DFlash2 · WSL HiCache

**速度速览：单路 decode 73.95 token/s；6 路合计 381.44 token/s（含 prefill）。**

**Speed at a glance: 73.95 token/s single-request decode; 381.44 token/s aggregate across six concurrent requests (including prefill).**

以上为 **RTX PRO 5000 Blackwell 48GB** 上于 **2026-10-18** 测得的短请求合成负载结果，并非所有请求的速度保证；6 路合计速度不是单个请求速度。

Measured on an **RTX PRO 5000 Blackwell 48GB** on **2026-10-18** using short synthetic requests. These are not speed guarantees for every workload; six-request aggregate throughput is not the speed of a single request.

在 **Windows + Docker/WSL2 + RTX PRO 5000 Blackwell 48GB** 上运行 Qwen3.8-27B Uncensored：保留 **262,144 上下文、6 路并发、DFlash2 K3、默认思考、视觉和工具调用**，并让 CPU HiCache 真正完成写入与读回。

Run Qwen3.8-27B Uncensored on **Windows + Docker/WSL2 + an RTX PRO 5000 Blackwell 48GB**, retaining a **262,144-token context, six concurrent request slots, DFlash2 K3, thinking enabled by default, vision, and tool calling**, with CPU HiCache that actually writes prefixes back and reloads them.

这是一份固定版本、独立维护的部署方案，包含两项限定范围的兼容性补丁、独立启动器和可复现检查；不是任何厂商的官方发行版。

This is a version-pinned, independently maintained deployment recipe with two scoped compatibility patches, a standalone launcher, and reproducible checks. It is not an official vendor release.

## 这份方案解决什么 / What this recipe addresses

- WSL 上注册内存的 CPU 地址不一定等于 GPU 可访问地址；单独打上游 JIT 补丁仍覆盖不到本配置的 AOT direct 路径。

  On WSL, the CPU address of registered host memory is not necessarily the address the GPU can access. The upstream JIT patch alone does not cover the AOT direct-copy path used by this configuration.

- 增加**显式启用、按设备能力判断**的 PyTorch-owned pinned allocator，验证实际地址别名，并保持内存所有权正确。

  Add an **explicitly enabled, device-capability-gated** PyTorch-owned pinned allocator. It checks the actual device-visible address alias and preserves correct memory ownership.

- 使用现有上游参数 `--mamba-max-states-per-path 1`，避免测试中出现的“CPU 缓存一直写、始终不读”现象。

  Use the existing upstream option `--mamba-max-states-per-path 1` to address the observed behavior where the CPU cache kept receiving writes but never reloaded a prefix.

- 不修改模型权重或模型计算精度；NVFP4 权重、FP8 KV 和 FP8 draft 是原配置本身。

  The compatibility fixes do not modify model weights or change the model's numerical precision. NVFP4 weights, FP8 KV, and the FP8 draft model are already part of the underlying configuration.

具体原理、来源和限制见 [实现说明](docs/implementation.md)。

See the [implementation notes](docs/implementation.md) for the mechanisms, provenance, and limitations.

## 已验证的结果 / Verified results

原部署验证日期：**2026-10-18**。硬件为 RTX PRO 5000 Blackwell 48GB，Windows 约 256GiB 内存。以下是特定合成负载的测量，不是对所有请求的速度承诺。

The original deployment was validated on **2026-10-18**, using an RTX PRO 5000 Blackwell 48GB and approximately 256GiB of Windows system RAM. These measurements come from specific synthetic workloads, not a speed guarantee for every request.

| 项目 / Metric | 实测 / Measured result |
| --- | --- |
| 单路短请求 decode<br>Single-request decode, short workload | **73.95 token/s** |
| 6 路短请求合计，含 prefill<br>Six-request aggregate throughput, including prefill | **381.44 token/s**，观察到 6 个 active 请求<br>**381.44 token/s**, with six active requests observed |
| 长前缀恢复<br>Long-prefix recovery | 27,733-token 请求：冷请求 **10.562s**，CPU 命中恢复 **1.263s**<br>27,733-token request: **10.562s** cold, **1.263s** with a CPU-cache hit |
| 确认真实 CPU 命中<br>Confirmed host-cache reuse | 27,706 host-hit tokens；读回 1,348,472,832 bytes<br>27,706 host-hit tokens; 1,348,472,832 bytes reloaded |
| 单请求容量<br>Single-request capacity | **262,000 输入 + 32 输出 tokens**，无截短<br>**262,000 input + 32 output tokens**, without truncation |
| 实测最低空闲显存<br>Lowest observed free VRAM | **2,834MiB**；本次验收下限为 2,560MiB<br>**2,834MiB**; the acceptance threshold for this run was 2,560MiB |
| CPU 缓存配置<br>CPU-cache setting | `--hicache-size 32`，**十进制 GB，不是 GiB**<br>`--hicache-size 32`, in **decimal GB, not GiB** |

**8.36× 是该长前缀短回答请求的端到端收益，不是 decode 变快 8 倍。** 开启前后的短请求 decode 基本相同。6 路共享 GPU KV 池，不是每路各有 262K；CPU 缓存是非活跃前缀的 L2，不扩大正在推理的 GPU 上下文。

**The 8.36× improvement is end-to-end latency for that long-prefix, short-answer request—not an eightfold increase in decode speed.** Short-request decode was essentially unchanged. The six slots share one GPU KV pool; each does not get its own 262K-token pool. CPU cache is an L2 tier for inactive prefixes, not an expansion of the active GPU context.

32GB 也不是全部 RAM 开销：target、Mamba、draft 及 pinned allocator 预留都会占内存。原机相对无 HiCache 时多占约 **53GiB** 主机内存。完整口径见 [验证记录](docs/validation.md)。

The 32GB setting is not the total RAM cost: target-model, Mamba, and draft-model pools, plus pinned-allocator reservations, all consume memory. The original machine used approximately **53GiB** more host RAM than it did without HiCache. See the [validation report](docs/validation.md) for the full measurement methodology.

## 快速开始 / Quick start

### 1. 准备环境和模型 / Prepare the environment and models

需要 Windows 上可用的 NVIDIA GPU / Docker Desktop Linux + WSL2 后端，以及 Node.js 22 或更新版。该参数档仅验证于上述 48GB GPU；不是 DGX 双机方案。

You need a working NVIDIA GPU setup on Windows, Docker Desktop using its Linux/WSL2 backend, and Node.js 22 or later. This profile has only been validated on the 48GB GPU specified above; it is not a dual-DGX deployment recipe.

为 Docker/WSL 提供足够内存：启动器要求 Docker 可见总内存至少 96GiB，这只是保守的启动前置条件，不保证运行时空闲 RAM。先确认 GPU 未被其他模型占用，并为 Windows 留足内存；不要在模型运行时随意重启 WSL 或 Docker。

Provide sufficient memory to Docker/WSL. The launcher requires at least 96GiB of total RAM visible to Docker; this is a conservative startup prerequisite, not a guarantee of available RAM at runtime. Ensure that other models are not occupying the GPU, leave enough RAM for Windows, and do not restart WSL or Docker while a model is running.

自行下载完整模型，放到仓库**外**的同一个目录。运行时仅只读挂载，不下载模型，不挂载 HF 登录凭据，也不要求清除现有凭据。

Download both complete models separately and place them under one parent directory **outside** this repository. The runtime mounts them read-only; it does not download models, mount Hugging Face login credentials, or require you to remove existing credentials.

| 模型 / Model | 验证所用 revision / Validated revision |
| --- | --- |
| [orcarouter/Qwen3.8-27B-Uncensored-NVFP4](https://huggingface.co/orcarouter/Qwen3.8-27B-Uncensored-NVFP4) | `69d21348b2d6c11439fb69368f40414c2256e44e` |
| [incoai/Qwen3.8-27B-DFlash2](https://huggingface.co/incoai/Qwen3.8-27B-DFlash2) | `015e795645c74b1a0eeef3b570031fb62e769bc5` |

目录结构：

Directory layout:

```text
模型根目录 / model-root/
├── orcarouter-Qwen3.8-27B-Uncensored-NVFP4/
│   ├── config.json
│   └── ...完整权重、tokenizer、视觉配置等 / complete weights, tokenizer, vision configuration, etc.
└── incoai-Qwen3.8-27B-DFlash2/
    ├── config.json
    └── ...完整 draft 文件 / complete draft-model files
```

上面固定的是实测版本，**不声称是仓库最新 HEAD**。后续升级需重新验证。

These are the revisions used in the measurements; they are **not claimed to be the latest repository HEADs**. Any subsequent upgrade requires revalidation.

### 2. 构建独立镜像 / Build a separate image

```powershell
git clone https://github.com/mtbnydx-dot/qwen27b-nvfp4-sglang-dflash2-wsl-hicache.git
cd qwen27b-nvfp4-sglang-dflash2-wsl-hicache
docker build --platform linux/amd64 -t qwen27b-hicache:sglang-0.5.22-wsl-v1 .
```

Dockerfile 固定 SGLang 基础镜像 digest；补丁安装器核对两个目标源文件的 SHA-256，使用 `--fuzz=0`，不匹配就停止。构建上下文只允许补丁与安装器进入镜像。

The Dockerfile pins the SGLang base-image digest. The patch installer checks the SHA-256 hashes of both target source files, applies patches with `--fuzz=0`, and stops on a mismatch. Only the patches and their installer are allowed into the image from the build context.

**发布状态说明：** GPU 实测来自原部署的相同补丁与启动参数。此仓库新增的独立启动器通过本地单元测试，补丁通过固定源码应用检查；这份 Dockerfile 尚未在发布机器上完成独立镜像构建和整模型 GPU 复验。请按下面步骤验证，不能把打包检查当成另一台机器上的运行保证。

**Release status:** The GPU measurements come from the original deployment using the same patches and serving arguments. This repository's standalone launcher has passed local unit tests, and the patches have passed application checks against the pinned source. However, this Dockerfile has not yet undergone an independent image build and full-model GPU revalidation on the publishing machine. Follow the checks below; packaging checks are not a guarantee that the recipe will work on another machine.

### 3. 先验证 GPU 拷贝，再启动 / Verify GPU copies before starting the server

在 GPU 空闲时执行一次隔离检查，无模型/凭据挂载、无网络、无端口发布：

While the GPU is idle, run this isolated check. It mounts no models or credentials, has no network access, and publishes no ports:

```powershell
docker run --rm --gpus device=0 --network none --memory 16g --shm-size 1g --mount "type=bind,source=$($PWD.Path)/tests,target=/tests,readonly" --entrypoint python3 qwen27b-hicache:sglang-0.5.22-wsl-v1 /tests/verify_direct_gpu.py
```

应出现 `DIRECT_PATCH_SUITE_PASSED`。该用例针对 host-pointer capability 为 0 的受影响设备；其他平台会拒绝把“不适用”计作通过。它会分配测试内存，不要与线上负载同时运行。

Expect `DIRECT_PATCH_SUITE_PASSED`. This test targets affected devices whose host-pointer capability is 0; it rejects other platforms rather than counting “not applicable” as a pass. It allocates test memory, so do not run it alongside production workloads.

设置本机路径，先查看 dry-run 参数：

Set your local model path and inspect the dry-run arguments first:

```powershell
$env:QWEN_MODELS_DIR = 'D:\Models'
node scripts/launch.mjs
node scripts/launch.mjs --start
```

Linux/WSL shell 示例：`export QWEN_MODELS_DIR=/mnt/d/Models`，然后运行相同 Node 命令。容器名默认 `qwen27b-hicache`，可用 `QWEN_CONTAINER` 修改；本机端口默认 8000，可用 `QWEN_PORT` 修改。

For a Linux/WSL shell, use `export QWEN_MODELS_DIR=/mnt/d/Models`, then run the same Node commands. The default container name is `qwen27b-hicache`, configurable through `QWEN_CONTAINER`. The default local port is 8000, configurable through `QWEN_PORT`.

启动器**不会停掉、删除或替换现有容器**。同名容器存在即退出；端口冲突由 Docker 拒绝。首次 JIT 编译和模型加载需要时间：

The launcher **does not stop, delete, or replace existing containers**. It exits if a container with the same name already exists, and Docker rejects port conflicts. Initial JIT compilation and model loading take time:

```powershell
docker logs --tail 100 -f qwen27b-hicache
curl.exe http://127.0.0.1:8000/health
node scripts/verify_api.mjs
```

### 4. 验证 CPU 缓存确实读回 / Verify actual CPU-cache reloads

仅在测试实例或维护窗口执行。下面会主动挤出 GPU 前缀缓存，最多发送 21 个长请求，**不要对正在服务用户的端点运行**：

Run this only against a test instance or during a maintenance window. It deliberately evicts GPU prefix-cache entries and sends up to 21 long requests. **Do not run it against an endpoint actively serving users:**

```powershell
node scripts/verify_api.mjs --host-cache --allow-load
```

通过必须同时满足：正确回答、host backup bytes 增加、种子回放期间 host load bytes 增加、host-hit tokens 增加。只看到 RAM 上涨或第二次回答变快不算通过。

Passing requires all of the following: correct answers, increased host backup bytes, increased host load bytes during seed replay, and increased host-hit tokens. Higher RAM usage or a faster second answer alone does not count as a pass.

默认请求本机 8000 端口；可设置 `QWEN_BASE_URL` 为无路径的引擎 origin。若配置了鉴权，可通过 `QWEN_API_KEY` 环境变量传入，脚本不会打印它。不要把密钥写进仓库或命令行参数。

The probe defaults to port 8000 on the local machine. You can set `QWEN_BASE_URL` to the engine's origin without a path. If authentication is configured, supply the key through the `QWEN_API_KEY` environment variable; the script does not print it. Do not put keys in the repository or command-line arguments.

这个简化探针不是完整验收：还需监测 C6、显存余量、CUDA/OOM、输出质量、长上下文容量及自己的真实工作负载。默认思考保持开启；合成缓存探针仅为自己的请求关闭思考。

This simplified probe is not a complete acceptance suite. You must also check six-request concurrency (C6), free VRAM, CUDA errors and out-of-memory failures, output quality, long-context capacity, and your own real workloads. Thinking remains enabled by default; the synthetic cache probe disables it only for its own requests.

## 暴露接口与回滚 / Exposing the API and rolling back

Docker 仅发布 `127.0.0.1:8000`。容器内部监听 `0.0.0.0` 不等于对公网暴露。若要公网访问，请另配 HTTPS、鉴权、请求限额；不要直接开放 8000 或管理/metrics 接口。

Docker publishes only `127.0.0.1:8000`. Listening on `0.0.0.0` inside the container does not mean the service is exposed to the public internet. For public access, configure HTTPS, authentication, and request limits separately. Do not directly expose port 8000 or the management/metrics endpoints.

本仓库不含任何生产域名、VPN/转发规则、账户凭据或私有管理器。它没有自动切换生产流量，也没有自动回滚其他容器。若迁移现有服务，先保留原容器和参数；测试失败时仅停新容器 `docker stop qwen27b-hicache`，再按你自己的管理流程恢复原服务。不要删除权重和 HF 登录凭据。

This repository contains no production domains, VPN/forwarding rules, account credentials, or private management framework. It does not automatically switch production traffic or roll back other containers. When migrating an existing service, preserve its original container and configuration first. If testing fails, stop only the new container with `docker stop qwen27b-hicache`, then restore the original service through your own management procedure. Do not delete model weights or Hugging Face login credentials.

## 文件与本地检查 / Repository layout and local checks

```text
profiles/        固定硬件参数与模型 revisions
                 Pinned hardware profile and model revisions
patches/         上游 JIT guard + 本地 opt-in WSL allocator
                 Upstream JIT guard + local opt-in WSL allocator
scripts/         构建时补丁检查、独立启动器、API/host-cache 探针
                 Build-time patch checks, standalone launcher, API/host-cache probes
tests/           CPU 参数测试、隔离 AOT GPU round-trip 测试
                 CPU argument tests and isolated AOT GPU round-trip tests
docs/            原理、来源、实测口径与限制
                 Implementation details, provenance, measurement methodology, and limitations
```

```sh
node --test
node --check scripts/verify_api.mjs
```

代码采用 Apache-2.0，见 [LICENSE](LICENSE) 和 [NOTICE](NOTICE)。模型权重、基础镜像及其依赖遵循各自的许可与访问条件；本仓库不打包或重新分发它们。

The code is licensed under Apache-2.0; see [LICENSE](LICENSE) and [NOTICE](NOTICE). Model weights, the base image, and their dependencies remain subject to their respective licenses and access conditions. This repository does not bundle or redistribute them.
