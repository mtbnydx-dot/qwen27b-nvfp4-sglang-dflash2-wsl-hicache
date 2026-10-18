"""Verify the pre-patched WSL HiCache AOT path, with no model/cache mounts.

Run only in a disposable GPU container while its GPU is idle. This allocates
test tensors; it never patches the image or starts a model server.
"""
import json
import os
import subprocess
import sys
import time

if len(sys.argv) == 1:
    results = {}
    for count in (4, 65):
        for family in ("mha", "mamba"):
            p = subprocess.run([sys.executable, __file__, "registered", family, str(count)], timeout=180, env={**os.environ, "SGLANG_WSL_HICACHE_PINNED_ALLOCATOR": "1"})
            results[family + "-" + str(count)] = p.returncode
    ok = all(value == 0 for value in results.values())
    print(json.dumps({"event": "DIRECT_PATCH_SUITE_PASSED" if ok else "DIRECT_PATCH_SUITE_FAILED", "results": results}), flush=True)
    sys.exit(0 if ok else 1)

import torch
from cuda.bindings import runtime
from sgl_kernel.kvcacheio import (
    transfer_kv_all_layer_direct_lf_pf,
    transfer_kv_per_layer_direct_pf_lf,
)
from sglang.srt.mem_cache.pool_host.common import (
    HostTensorAllocator,
    alloc_with_host_register,
    alloc_with_pin_memory,
)

allocator, family = sys.argv[1:3]
count = int(sys.argv[3])
torch.cuda.set_per_process_memory_fraction(0.015, 0)
err, attribute = runtime.cudaDeviceGetAttribute(runtime.cudaDeviceAttr.cudaDevAttrCanUseHostPointerForRegisteredMem, 0)
assert int(err) == 0 and attribute == 0
page, layers, width, buffers = (32, 4, 128, 2) if family == "mha" else (1, 40, 16384, 1)
source_pages = list(range(2, count + 2))
host_pages = list(range(count, 0, -1))
dest_pages = list(range(count))
allocate = alloc_with_host_register if allocator == "registered" else alloc_with_pin_memory
host = [allocate((count + 8, layers, page, width), torch.uint8, "cpu", True, HostTensorAllocator()) for _ in range(buffers)]
for h in host:
    h.fill_(97)
err, device_alias = runtime.cudaHostGetDevicePointer(host[0].data_ptr(), 0)
assert int(err) == 0
assert int(device_alias) == host[0].data_ptr()
assert all(getattr(h, "_sglang_cuda_host_registered_ranges", None) == [] for h in host)
print(json.dumps({"event": "DIRECT_BEGIN", "allocator": "patched-owned-pinned", "family": family, "pages": count, "host_attribute": attribute, "same_host_alias": True}), flush=True)

source = []
for i in range(layers * buffers):
    values = ((torch.arange((count + 8) * page * width, dtype=torch.int32, device="cuda") * 17 + i * 31) % 251).to(torch.uint8)
    source.append(values.reshape((count + 8) * page, width))
expected_source = [s.cpu() for s in source]
stream = torch.cuda.Stream()
stream.wait_stream(torch.cuda.current_stream())
indices = lambda pages: torch.tensor([p * page + offset for p in pages for offset in range(page)], dtype=torch.int64)
src_indices, host_indices, dst_indices = map(indices, (source_pages, host_pages, dest_pages))
started = time.monotonic()
with torch.cuda.stream(stream):
    transfer_kv_all_layer_direct_lf_pf(source, host, src_indices, host_indices, page)
stream.synchronize()
for b, actual in enumerate(host):
    expected = torch.full_like(actual, 97)
    for sp, hp in zip(source_pages, host_pages):
        for layer in range(layers):
            expected[hp, layer].copy_(expected_source[b * layers + layer][sp * page:(sp + 1) * page])
    assert torch.equal(actual, expected), "D2H mismatch or untouched page corruption"

with torch.cuda.stream(stream):
    dest = [torch.full_like(s, 91) for s in source]
    for layer in range(layers):
        transfer_kv_per_layer_direct_pf_lf(host, [dest[b * layers + layer] for b in range(buffers)], host_indices, dst_indices, layer, page)
stream.synchronize()
for b in range(buffers):
    for layer in range(layers):
        expected = torch.full_like(expected_source[b * layers + layer], 91)
        for sp, dp in zip(source_pages, dest_pages):
            expected[dp * page:(dp + 1) * page].copy_(expected_source[b * layers + layer][sp * page:(sp + 1) * page])
        assert torch.equal(dest[b * layers + layer].cpu(), expected), "H2D mismatch or untouched page corruption"
from sglang.srt.mem_cache.pool_host.common import _cuda_host_unregister
for h in host:
    _cuda_host_unregister(h)
assert int(runtime.cudaGetLastError()[0]) == 0
print(json.dumps({"event": "DIRECT_PASS", "allocator": "patched-owned-pinned", "family": family, "pages": count, "non_default_stream": True, "full_buffer_bytes_compared": True, "cleanup_checked": True, "seconds": round(time.monotonic() - started, 3)}), flush=True)
