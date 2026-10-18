# 验证记录与测量口径

## 原部署验收：2026-10-18

平台：RTX PRO 5000 Blackwell 48GB；Windows + Docker/WSL2；Windows 可见总内存约 255.16GiB。SGLang 0.5.22，PyTorch 2.14.1+cu130，CUDA toolkit 13.0.88。所有输入为合成测试，不发布用户提示词、请求记录、生产域名或凭据。

### GPU 拷贝

- 上游 JIT guard 后，7 个独立进程测试通过，覆盖 MHA/MLA 和跨 64-page staging 边界的 65-page 传输。
- 原始 JIT 的第一个 128KiB / 4-page 用例也通过，因此它本身**不是**已复现的失败对照，不能据此宣称修复有效。
- 独立 AOT direct 用例中，默认 registered allocator 的 MHA 与 Mamba 两种几何均复现 illegal-address 700。
- owned-pinned allocator 在两种几何、4 / 65 pages 共 4 个用例通过；非默认 CUDA stream、乱序页映射、所有字节（包括不应被改动的页）与清理过程均检查。

仓库中的 `tests/verify_direct_gpu.py` 由最后一组用例适配，只测试已打补丁的镜像。它不包含故意触发 illegal address 的旧失败分支，也不会在运行时修改生产容器。

### 服务功能

原部署通过默认思考、中文、简单视觉、工具调用 JSON、SSE 完整性及模型 alias 检查。两个完整采样回答人工阅读；生成的 LRU 示例及 5 个单元测试在无网络、只读、非特权 CPU 容器执行，全部通过。

这是小范围 smoke/质量检查，不是能力排行榜、完整模型质量评测或长时间稳定性验证。

### 短请求性能

| 指标 | CPU cache 关闭时 K3 基线 | 最终 HiCache 配置 |
| --- | --- | --- |
| C1 纯 decode | 74.20 token/s | 73.95 token/s |
| C6 合计（包含 prefill） | 384.12 token/s | 381.44 token/s |

最终 C6 为 3072 输出 tokens / 8.054s，采样看到了 6 个同时 active 的请求；最低空闲显存 2837MiB。两轮 C6 混合 tool / 10,890-token prefill 也通过。

这些数据表明短 decode 基本持平，不能证明该缓存方案会让所有请求更快，也不能把 C6 合计速度当作单个请求速度。吞吐会随提示长度、并发、思考长度和 draft 接受率变化。

### 真正的 host-prefix 恢复

先发送 27,733-token seed，再发送 12 个不同长文档制造 GPU 缓存压力，期间不刷新 seed。

| 项目 | 结果 |
| --- | --- |
| seed 冷请求 | 10.562s |
| 压力后的 seed CPU 恢复 | 1.263s |
| host-hit tokens | 27,706 |
| 实际 host load bytes | 1,348,472,832 |
| 读回 Mamba checkpoint | 2 |
| 随后的 GPU-cached 再次请求 | 0.380s |

约 8.36× 是上述短答案合成请求的端到端对比，不是 decode 提升。输出验证码正确。GPU-cached 0.380s 和 CPU 恢复 1.263s 也不是同一级缓存，不能混为一个数字。

默认 path cap 的早期尝试虽然累计写回约 109GB，但加载字节和 host 命中为零，被明确判定失败。不能仅凭“HiCache allocated”或 CPU 内存占用认为功能正常。

### 最大容量与并发 host 回放

- 262,000 实际输入 + 32 输出 tokens，无截断，173.353s；最低空闲显存 2834MiB。重复 token 的容量测试不代表超长文档推理质量。
- 上述容量压力后，6 个客户端同时回放旧长前缀，全部回答正确，6.258s 完成；host-hit tokens 合计 166,266，host load bytes 8,091,052,032；最低空闲显存 2835MiB。
- 该短回放测试的 scheduler 采样未捕获 active decode（记录为 0）。它证明 6 个客户端同时提交并完成，不是又一次观察到 6 路同时 decode；后者由独立 C6 长输出测试确认。

### 内存与验收边界

配置的 `hicache-size=32` 是十进制 GB。启动日志中的 target KV、Mamba、draft host pool 合计约 39.4GB；PyTorch pinned allocator 的预留使实际 RAM 增量进一步增至约 53GiB。该测试中的 nominal host KV 容量约 715,370 tokens。

Windows 可用内存从无缓存时约 162.55GiB 降到验收末尾约 109.8GiB。验收末尾 GPU 空闲 2842MiB，观察窗口内没有 OOM 或重启。

本次明确采用 **2560MiB** 空闲显存下限。不能声称通过了此前的 3GiB / 4GiB 门槛。其他桌面负载、显示输出、驱动版本和并发组合都可能改变余量；若需要更大保护空间，应降低内存参数并重新验证。

CPU cache 是 inactive-prefix L2，不是把内存当成无限 GPU KV，也不意味着 6 路可同时各运行 262K 上下文。未完成多日真实流量 soak；不作无限稳定承诺。

## 公开打包验证

为避免泄露基础设施配置，未直接推送原私有管理器或日志。独立启动器和探针是发布阶段适配版本：

- 本地 CPU 单元测试覆盖参数、loopback/只读挂载、非法参数拒绝和首次命中前的 metrics 行为。
- 两个 patch 在固定 SGLang 源码副本上检查并应用，拒绝源文件哈希不匹配，Python 源码做语法检查。
- 独立 Dockerfile 未在发布机器完成实际构建，新包装未进行第二轮 GPU / API 实测。

因此请区分“核心参数和补丁已在原机验收”与“公开包装在你的环境验收”。首次部署仍需完整检查。
