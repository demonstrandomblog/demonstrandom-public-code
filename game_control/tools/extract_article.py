# AI-assisted research code; no blanket human or mathematical review is claimed.
"""Reproduce the implementation and examples from an explicit article source."""
from pathlib import Path
import argparse
import ast
import hashlib
import json
import re

SOURCE = 'demonstrandom/game_theory/posts/engineering_game_types/index.qmd'
BLOCK = re.compile(r'^```python\n(.*?)^```', re.M | re.S)


def extract(post):
    text = post.read_text(encoding='utf-8')
    start = text.index('# Appendix: Atomic Operations')
    appendix = text[start:]
    chunks = ['# Game design via payoff maps, target regions, and constrained control commands.\n'
              '# Payoffs use trailing dimensions (players, *actions); Map composes transformations.\n'
              '# AI-generated research code; no full human review is recorded.\n'
              '# Local usage, solver contracts, and examples are in this package README.\n']
    symbols = []
    names = set()
    for block in BLOCK.finditer(appendix):
        lines = block[1].splitlines()
        source_line = text[:start + block.start(1)].count('\n') + 1
        for node in ast.parse(block[1]).body:
            if not isinstance(node, (ast.Import, ast.ImportFrom, ast.ClassDef, ast.FunctionDef)):
                continue
            first = min([node.lineno] + [d.lineno for d in getattr(node, 'decorator_list', [])])
            segment = '\n'.join(lines[first - 1:node.end_lineno])
            generated_line = ''.join(chunks).count('\n') + 1
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)):
                if node.name in names:
                    raise ValueError(f'Duplicate appendix definition: {node.name}')
                names.add(node.name)
                symbols.append({'name': node.name, 'kind': type(node).__name__,
                                'source_line': source_line + first - 1,
                                'generated_line': generated_line})
            chunks.append(segment + '\n\n')
    if not {'PIDState', 'pid_rollout', 'solve_polynomial'} <= names:
        raise ValueError('Current PID or polynomial implementation is missing')
    examples = []
    counts = []
    section = text.split('# Vignettes', 1)[1].split('# Conclusion', 1)[0]
    for match in re.finditer(r'^## (\d+)\. (.*?)(?=^## \d+\.|\Z)', section, re.M | re.S):
        number = int(match[1])
        blocks = BLOCK.findall(match[2].split('### Example:', 1)[1])
        if not blocks:
            raise ValueError(f'No concrete blocks in vignette {number}')
        counts.append([number, len(blocks)])
        title = re.sub(r'\s*\{[^}]*\}\s*$', '', match[2].splitlines()[0])
        examples.append(f'# Example {number}: {title}\n' + '\n'.join(blocks))
    if not examples:
        raise ValueError('No vignettes found')
    return ''.join(chunks), '\n\n'.join(examples) + '\n', symbols, counts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('command', choices=['check', 'generate'])
    parser.add_argument('--source', type=Path, required=True, help='Engineering Game Types index.qmd')
    parser.add_argument('--output', type=Path, help='Package directory; required for generate')
    args = parser.parse_args()
    if args.command == 'generate' and args.output is None:
        parser.error('generate requires --output')
    output = args.output or Path(__file__).resolve().parents[1]
    module, examples, symbols, counts = extract(args.source)
    artifacts = {'src/game_control/__init__.py': module.encode('utf-8'), 'examples.py': examples.encode('utf-8')}
    if args.command == 'check':
        manifest = json.loads((output / 'provenance.json').read_text(encoding='utf-8'))
        if hashlib.sha256(args.source.read_text(encoding='utf-8').encode('utf-8')).hexdigest() != manifest['source_sha256']:
            raise RuntimeError('Article differs from the recorded source revision.')
        for name, data in artifacts.items():
            if (output / name).read_text(encoding='utf-8').encode('utf-8') != data:
                raise RuntimeError(f'Generated content differs: {name}')
        if symbols != manifest['definitions'] or counts != manifest['vignette_blocks']:
            raise RuntimeError('Definition or vignette provenance differs.')
        print(f'Article matches: {len(symbols)} definitions and {len(counts)} vignettes.')
    else:
        for name, data in artifacts.items():
            path = output / name
            if path.exists() and path.read_text(encoding='utf-8').encode('utf-8') != data:
                raise RuntimeError(f'Refusing to overwrite different content: {path}')
        for name, data in artifacts.items():
            path = output / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        print(f'Generated {len(symbols)} definitions and {len(counts)} vignettes. Revalidate before release.')


if __name__ == '__main__':
    main()
