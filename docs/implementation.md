# 实现、版本固定与来源

## 范围

本仓库是一套单卡 Windows/Docker/WSL2 的部署 recipe。它不修改模型权重，不生成去审查权重，不提供 DGX 集群调度或公网 VPN 配置。目标权重使用 OrcaRouter 已发布的 checkpoint；DFlash2 draft 使用 incoai 发布的 checkpoint。

固定 SGLang 0.5.22 源码版本 `baa9befa381511cd4d56f83980951e5a45fffc6d`，基础镜像：

```text
lmsysorg/sglang@sha256:e3a4f06001a29542fc4a8638536d9ffc5eb9c9942b08603c09b19a56eae4a1ae
```

模型和 draft revisions 见 `profiles/rtx-pro-5000-48gb.json`。不要把本地测试镜像的 image ID 当成其他人可以拉取的 registry digest；本仓库提供 Dockerfile，构建会生成你自己的 image ID。

## 1. 上游 JIT staged-copy guard

来源：[SGLang PR #40571](https://github.com/sgl-project/sglang/pull/40571)，取自 head `ca823fe4cfb4bba75360416929f74749721b9cb7`；相关报告：[issue #40232](https://github.com/sgl-project/sglang/issues/40232)。该 PR 在原部署检查时尚未合并，这里不是声称已进入正式 release。

修改 `kernels/jit/csrc/kvcacheio/staged_write_back.cuh`：查询 `cudaDevAttrCanUseHostPointerForRegisteredMem`。不能用原始 host 地址时，避开 batch-copy 快路径，交回原有逐页拷贝回退路径。

这段补丁归上游贡献者所有。本仓库只固定并复用它；**它单独不足以解决本配置的全部问题**。`direct + page_first_direct` 使用的 AOT 算子是另一条路径。

## 2. WSL owned-pinned allocator

本地补丁修改 `srt/mem_cache/pool_host/common.py`。只在以下条件同时满足时生效：

- `SGLANG_WSL_HICACHE_PINNED_ALLOCATOR=1`；
- 请求 pinned memory，非 HIP；
- allocator 类型恰好是普通 `HostTensorAllocator`；
- 当前 CUDA 设备的 host-pointer capability 为 0。

此时使用 `torch.empty(..., device="cpu", pin_memory=True)`，并检查 `cudaHostGetDevicePointer` 返回地址等于 tensor 地址。检查失败直接报错，不尝试以未知地址继续运行。

分配由 PyTorch 管理，因此 `_sglang_cuda_host_registered_ranges` 标记为空，防止 SGLang 清理时错误地 `cudaHostUnregister` PyTorch 拥有的内存。自定义、共享、分布式 allocator 不被替换；支持直接使用 host 地址的设备不进入该分支。环境变量未设置也不生效。

实机 AOT 测试中，默认 mmap + cudaHostRegister 的 MHA/Mamba 两种几何均复现 CUDA error 700，地址别名不同；owned-pinned 方案的地址别名相同，完整读写与清理测试通过。此结论仅针对已测平台，不保证所有虚拟化设备都支持相同别名；代码保留了 fail-closed 检查。

## 3. Mamba checkpoint 容量

`--mamba-max-states-per-path 1` 是已有上游参数，不是本仓库新增的功能。启发来自 [club-3090 实验记录](https://github.com/noonghunna/club-3090/blob/master/docs/UPSTREAM.md) 中关于 [SGLang #33713](https://github.com/sgl-project/sglang/issues/33713) 的记录。

本机默认无限制保留时，host KV 池持续写入，但加载字节和 host-hit tokens 始终为零；加上 path cap 1 后，长前缀回放出现了实际 host 命中和加载。这支持 checkpoint 保留策略是本机的实用修复，但不证明该上游 issue 的所有情况都由同一原因造成。

此参数限制每条路径上可淘汰的 Mamba 状态保留，不删除全部 KV，也不是把上下文降低为 1。tail、fork 和 locked 节点遵循上游保护逻辑。

## 固定参数摘要

| 部件 | 值 |
| --- | --- |
| target / draft | NVFP4 / FP8 |
| DFlash2 draft block | 3 |
| target / draft attention | FlashInfer / FlashInfer |
| KV / Mamba SSM | FP8 e4m3 / BF16 |
| 上下文 / GPU 总 token pool | 262144 / 262272 |
| 最大 running requests | 6 |
| `mem-fraction-static` | 0.84 |
| Mamba cache slots / states per path | 18 / 1 |
| chunked prefill / prefill-decode interval | 512 / 1 |
| HiCache target budget | 32 十进制 GB |
| HiCache IO / layout / write policy | direct / page_first_direct / write_back |
| CUDA graph decode 最大 batch | 6 |
| prefill graph | disabled |
| VLM feature cache | 32MB，transport 为 CPU |
| 思考 / tool parser | 默认开启 / qwen3_coder |

另保留 `SGLANG_OPT_MAMBA_SKIP_DECODE_LOCK=1`。参数是该硬件上整体测试后的组合，不表示其中每个开关都单独完成因果 A/B。

## 修改或升级时

不要直接把基础镜像替换成 `latest` 并删除源文件校验。上游若合并修复或改变 AOT/host allocator，需要先重新读源码、判断补丁是否仍需要，再做隔离 GPU 拷贝、功能、负载和真实 L2 命中测试。补丁哈希校验只覆盖两个小源文件，不扫描模型权重。

新 recipe 的 Dockerfile/启动器与原私有控制器分离，启动时不处理已有容器，也不会自动回滚线上模型。只使用未携带凭据的新编译缓存卷；不要拿包含 HF token 的旧宿主缓存来制作镜像。
