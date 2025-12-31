#!/usr/bin/env python3
"""Generate SHA256 checksum for the memory_field.dll file.
Usage: python tools/generate_sha256.py <path-to-dll> [--out <path-to-sha256>]
"""
import hashlib
import sys
from pathlib import Path

if len(sys.argv) < 2:
    print("Usage: python tools/generate_sha256.py <path-to-dll> [--out <out-file>]")
    sys.exit(2)

p = Path(sys.argv[1])
if not p.exists():
    print(f"File not found: {p}")
    sys.exit(1)

out = None
if '--out' in sys.argv:
    idx = sys.argv.index('--out')
    out = Path(sys.argv[idx+1])
else:
    out = p.with_suffix(p.suffix + '.sha256')

h = hashlib.sha256()
with p.open('rb') as f:
    for chunk in iter(lambda: f.read(8192), b''):
        h.update(chunk)
digest = h.hexdigest()
# Write in common format: sha256sum style
out.write_text(f"{digest}  {p.name}\n")
print(f"Wrote {out} (sha256: {digest})")
