FROM lmsysorg/sglang@sha256:e3a4f06001a29542fc4a8638536d9ffc5eb9c9942b08603c09b19a56eae4a1ae

# No weights, credentials, production logs or host caches enter the build context.
COPY patches/ /opt/qwen-hicache-patches/
COPY scripts/apply_patches.py /opt/qwen-hicache-patches/apply_patches.py
RUN command -v patch && python3 /opt/qwen-hicache-patches/apply_patches.py

LABEL org.opencontainers.image.title="Qwen 27B SGLang DFlash2 WSL HiCache" \
      org.opencontainers.image.source="https://github.com/mtbnydx-dot/qwen27b-nvfp4-sglang-dflash2-wsl-hicache" \
      ai.qwen.hicache.patch="sgl-project/sglang#40571" \
      ai.qwen.hicache.allocator="wsl-owned-pinned-v1" \
      ai.qwen.hicache.recipe="2026-10-18"
