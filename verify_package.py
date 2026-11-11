"""Verify this delivered code snapshot using Python's standard library only."""
from pathlib import Path
import ast
import hashlib
import json

def main():
    root = Path(__file__).resolve().parent
    manifest = json.loads((root / 'MANIFEST.json').read_text(encoding='utf-8'))
    failures = []
    for record in manifest['files']:
        p = root / record['path']
        if not p.is_file():
            failures.append('Missing: ' + record['path'])
        elif hashlib.sha256(p.read_bytes()).hexdigest() != record['sha256']:
            failures.append('Changed: ' + record['path'])
    scripts = list(root.rglob('*.py'))
    scripts = [p for p in scripts if 'rerun_outputs' not in p.relative_to(root).parts]
    for p in scripts:
        ast.parse(p.read_text(encoding='utf-8-sig'), filename=str(p))
    for directory in ['original_code', 'patched_project']:
        if len(list((root / directory).glob('*.py'))) != 12:
            failures.append(directory + ': expected 12 source files')
    if failures:
        raise SystemExit('\n'.join(failures))
    print('PASS: %d file hashes; %d Python sources; 12 original and 12 corrected scripts.'
          % (len(manifest['files']), len(scripts)))
    print('This checks the delivered snapshot, not mathematical correctness.')

if __name__ == '__main__':
    main()
