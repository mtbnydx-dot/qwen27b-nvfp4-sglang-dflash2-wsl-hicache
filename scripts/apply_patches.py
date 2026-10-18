"""Apply only the reviewed patches to the exact pinned SGLang sources.

Used during image build; never targets a running serving container.
"""
import hashlib
import importlib.util
from pathlib import Path
import subprocess


PATCHES = (
    (
        "kernels/jit/csrc/kvcacheio/staged_write_back.cuh",
        "a110d5b27c98f4ef3e120aeba6b9e502f26a89e2dabf1973b8ee90327989a48b",
        "0001-sglang-pr40571-host-pointer-guard.patch",
    ),
    (
        "srt/mem_cache/pool_host/common.py",
        "e35b267eb56a99c9f72422911db870e6da15816bc534aabea5b12f4865956ec5",
        "0002-wsl-owned-pinned-allocator.patch",
    ),
)


def apply(package_root, patch_root):
    # Check every source before changing either file. No fuzz or version guessing.
    for relative, expected, patch_name in PATCHES:
        target = package_root / relative
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        if actual != expected:
            raise RuntimeError(f"Source mismatch for {relative}; refuse patching")
        subprocess.run(
            ["patch", "--dry-run", "--batch", "--forward", "--fuzz=0",
             str(target), str(patch_root / patch_name)], check=True,
        )
    for relative, _, patch_name in PATCHES:
        subprocess.run(
            ["patch", "--batch", "--forward", "--fuzz=0",
             str(package_root / relative), str(patch_root / patch_name)], check=True,
        )
    # Compile without importing torch or needing a GPU during docker build.
    target = package_root / "srt/mem_cache/pool_host/common.py"
    compile(target.read_text(), str(target), "exec")
    print("Both pinned HiCache patches applied successfully")


if __name__ == "__main__":
    spec = importlib.util.find_spec("sglang")
    if spec is None or spec.origin is None:
        raise RuntimeError("SGLang package not found")
    apply(Path(spec.origin).parent, Path(__file__).parent)
