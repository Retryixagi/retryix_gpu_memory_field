#!/usr/bin/env python3
"""
Verify a memory_field release artifact: download (from URL or local), compare SHA256, and attempt to load the DLL (Windows).
Usage:
  python tools/verify_memory_field_release.py --dll <path-or-url> --sha <path-or-url>
"""
import argparse
import hashlib
import requests
import tempfile
import os
import sys
from pathlib import Path
import shutil

parser = argparse.ArgumentParser()
parser.add_argument('--dll', required=True, help='Path or URL to memory_field.dll')
parser.add_argument('--sha', required=True, help='Path or URL to memory_field.dll.sha256')
args = parser.parse_args()

def download_to_temp(url_or_path):
    if url_or_path.startswith('http://') or url_or_path.startswith('https://'):
        r = requests.get(url_or_path, stream=True)
        r.raise_for_status()
        fd, tmp = tempfile.mkstemp()
        with os.fdopen(fd, 'wb') as f:
            for chunk in r.iter_content(8192):
                f.write(chunk)
        return Path(tmp)
    else:
        p = Path(url_or_path)
        if not p.exists():
            raise FileNotFoundError(str(p))
        return p

try:
    dll_p = download_to_temp(args.dll)
    sha_p = download_to_temp(args.sha)
    # read expected
    expected = sha_p.read_text().strip().split()[0]
    # compute actual
    h = hashlib.sha256()
    with dll_p.open('rb') as f:
        for chunk in iter(lambda: f.read(8192), b''):
            h.update(chunk)
    actual = h.hexdigest()
    print(f"Expected: {expected}")
    print(f"Actual:   {actual}")
    if actual != expected:
        print("ERROR: SHA256 mismatch")
        sys.exit(2)
    print("SHA256 OK")

    # Attempt to load DLL on Windows
    if os.name == 'nt':
        import ctypes
        try:
            lib = ctypes.WinDLL(str(dll_p))
            handle = lib._handle
            print(f"Loaded DLL, handle={handle}")
            # try to free
            ctypes.windll.kernel32.FreeLibrary(handle)
            print("Freed library")
        except Exception as e:
            print(f"ERROR: failed to load DLL: {e}")
            sys.exit(3)
    else:
        print("Non-Windows OS: skipping dynamic load test. SHA verification succeeded.")

    print("Verification succeeded")
finally:
    # clean up temp files if we downloaded them
    # Only remove if they were temp files (not a literal path)
    if args.dll.startswith('http'):
        try:
            os.remove(dll_p)
        except Exception:
            pass
    if args.sha.startswith('http'):
        try:
            os.remove(sha_p)
        except Exception:
            pass
