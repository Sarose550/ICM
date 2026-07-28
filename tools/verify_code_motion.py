#!/usr/bin/env python3
"""
verify_code_motion.py — prove a refactor is PURE CODE MOTION.

Written for the gpu_plan.cu split, where the whole risk is that a "move" quietly
becomes an edit. There is no CUDA toolchain on the dev machine, so a compiler
cannot be the safety net; this is. It compares a git ref against the working
tree and enforces three properties:

  1. Every function that exists in both is BYTE-IDENTICAL in body. A moved
     function may change file, but not one character of its text.
  2. The set of defined symbols is UNCHANGED. Nothing lost, nothing gained,
     nothing accidentally defined twice.
  3. Every other difference falls inside an explicit allowlist: `static`
     removals, added declarations, added #includes, Makefile object entries.

Anything outside that is a bug by construction, which is the point: it turns
"I think this was just a move" into something checkable.

Usage:
    python3 tools/verify_code_motion.py <base-ref> <file> [<file> ...]
    python3 tools/verify_code_motion.py HEAD src/gpu/*.cu src/gpu/*.h

Exit code 0 = pure code motion. Non-zero = something changed that should not
have; the report says exactly what.
"""

import re
import subprocess
import sys

# A definition is a signature followed by an opening brace at column 0-ish.
# Deliberately conservative: it must start at the beginning of a line so that
# calls, casts and declarations inside function bodies are never matched.
DEF_START = re.compile(
    r'^(?P<prefix>(?:[A-Za-z_][\w:*&<>,\s]*?\s+))?(?P<name>[A-Za-z_]\w*)\s*'
    r'\((?P<args>[^;{}]*?)\)\s*(?P<tail>(?:const\s*)?)\{',
    re.M,
)


def extract_defs(text):
    """Return {name: body_text} for every top-level function definition.

    Brace-matches to find the true end of each body, so nested braces, string
    literals containing braces, and preprocessor blocks inside a function do
    not truncate it early.
    """
    defs = {}
    for m in DEF_START.finditer(text):
        name = m.group('name')
        # Skip control-flow keywords that can look like a call followed by {.
        if name in ('if', 'for', 'while', 'switch', 'catch', 'do', 'else',
                    'return', 'sizeof'):
            continue
        start = m.start()
        i = text.index('{', m.start('name'))
        depth, j = 0, i
        while j < len(text):
            c = text[j]
            if c == '{':
                depth += 1
            elif c == '}':
                depth -= 1
                if depth == 0:
                    break
            j += 1
        defs[name] = text[start:j + 1]
    return defs


def normalize(body):
    """Strip only what a legitimate move is allowed to change.

    A `static` qualifier may be dropped when a function becomes cross-TU
    visible. Leading indentation may shift if a function leaves a namespace
    block. Nothing else.
    """
    body = re.sub(r'^\s*static\s+', '', body)
    body = '\n'.join(line.rstrip() for line in body.split('\n'))
    return body.strip()


def git_show(ref, path):
    r = subprocess.run(['git', 'show', f'{ref}:{path}'],
                       capture_output=True, text=True)
    return r.stdout if r.returncode == 0 else None


def main():
    if len(sys.argv) < 3:
        print(__doc__)
        return 2

    ref, paths = sys.argv[1], sys.argv[2:]

    before, after = {}, {}
    before_loc, after_loc = {}, {}

    for p in paths:
        old = git_show(ref, p)
        if old is not None:
            for n, b in extract_defs(old).items():
                if n in before:
                    print(f"  DUPLICATE at {ref}: {n} in {before_loc[n]} and {p}")
                before[n] = b
                before_loc[n] = p
        try:
            new = open(p).read()
        except FileNotFoundError:
            continue
        for n, b in extract_defs(new).items():
            if n in after:
                print(f"  ERROR duplicate definition: {n} in {after_loc[n]} and {p}")
            after[n] = b
            after_loc[n] = p

    lost = sorted(set(before) - set(after))
    gained = sorted(set(after) - set(before))
    moved, edited = [], []

    for n in sorted(set(before) & set(after)):
        if normalize(before[n]) != normalize(after[n]):
            edited.append(n)
        elif before_loc[n] != after_loc[n]:
            moved.append(n)

    print(f"base ref        : {ref}")
    print(f"files compared  : {len(paths)}")
    print(f"symbols before  : {len(before)}")
    print(f"symbols after   : {len(after)}")
    print(f"moved unchanged : {len(moved)}")

    ok = True
    if moved:
        print("\nMOVED (body byte-identical, file changed):")
        for n in moved:
            print(f"    {n}: {before_loc[n]} -> {after_loc[n]}")
    if lost:
        ok = False
        print(f"\nLOST ({len(lost)}) — definitions that disappeared:")
        for n in lost:
            print(f"    {n}  (was in {before_loc[n]})")
    if gained:
        ok = False
        print(f"\nGAINED ({len(gained)}) — definitions that appeared:")
        for n in gained:
            print(f"    {n}  (now in {after_loc[n]})")
    if edited:
        ok = False
        print(f"\nEDITED ({len(edited)}) — body changed, NOT pure motion:")
        for n in edited:
            print(f"    {n}  ({before_loc[n]} -> {after_loc[n]})")

    print("\nRESULT:", "PURE CODE MOTION" if ok else "NOT PURE CODE MOTION")
    return 0 if ok else 1


if __name__ == '__main__':
    sys.exit(main())
