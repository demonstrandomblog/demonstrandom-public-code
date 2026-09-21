# AI-assisted research code; no blanket human or mathematical review is claimed.
"""Run the bundled game-design examples and mathematical, PID, and polynomial checks."""
from pathlib import Path
import argparse
import hashlib
import json
import sys
import time


def sha(path):
    return hashlib.sha256(path.read_text(encoding='utf-8').encode('utf-8')).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, help='Optional JSON report path')
    args = parser.parse_args()
    if not __debug__:
        parser.error('Run without -O: these regression checks require assertions.')
    import game_control as module
    import numpy, scipy, sympy, torch, z3
    from checks.mathematical import verify
    from checks.pid import check_pid
    from checks.polynomial import check

    base = Path(__file__).resolve().parent
    manifest = json.loads((base / 'provenance.json').read_text(encoding='utf-8'))
    # Test the installed package, and require it to match the bundled source.
    if sha(Path(module.__file__)) != manifest['generated_files']['src/game_control/__init__.py']:
        raise RuntimeError('Installed game_control does not match this checkout; reinstall it.')
    for name, digest in manifest['generated_files'].items():
        if sha(base / name) != digest:
            raise RuntimeError(f'Source/provenance mismatch: {name}')
    started = time.monotonic()
    report = {'status': 'running', 'source_sha256': manifest['source_sha256'],
              'generated_files': manifest['generated_files'], 'suites': {}}
    try:
        examples = base / 'examples.py'
        exec(compile(examples.read_text(encoding='utf-8'), str(examples), 'exec'), vars(module))
        report['vignette_blocks'] = manifest['vignette_blocks']
        print('All ten article vignettes executed.', flush=True)
        for name, check_function in [('mathematical_regression', verify), ('pid', check_pid), ('polynomial', check)]:
            report['suites'][name] = check_function(module)
            print(f'{name} passed.', flush=True)
        report['verifier_sources'] = {p.relative_to(base).as_posix(): sha(p) for p in [base / 'verify.py', *sorted((base / 'checks').glob('*.py'))]}
        report['runtime'] = {'python': sys.version.split()[0], 'numpy': numpy.__version__,
                             'scipy': scipy.__version__, 'sympy': sympy.__version__,
                             'torch': torch.__version__, 'z3': z3.get_version_string()}
        report['status'] = 'passed'
    except Exception as exc:
        report['status'] = 'failed'
        report['error'] = f'{type(exc).__name__}: {exc}'
        raise
    finally:
        report['elapsed_seconds'] = round(time.monotonic() - started, 2)
        report['verified_at_utc'] = time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())
        if args.output:
            args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
