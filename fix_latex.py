#!/usr/bin/env python3
"""Fix LaTeX rendering issues in GitHub markdown files.

GitHub's markdown parser runs BEFORE the math renderer, so:
- \\  in $$...$$ → \   (backslash escape consumed)
- \{  in $...$  → {   (backslash escape consumed)
- \|  in $...$  → |   (backslash escape consumed)
- _x$ ... $_    → <em> (underscore emphasis pairing across math boundaries)

Fixes:
1. Display math: single-line $$...$$ → multi-line with $$ on own lines
   (prevents markdown from processing backslash escapes in display math)
2. Display math: $$\\begin{...} → split onto separate lines
3. Inline math: \\{ → \\lbrace, \\} → \\rbrace (safe LaTeX synonyms)
4. Inline math: \\| → \\Vert (safe LaTeX synonym)
5. Inline math: _ → \\_ (markdown unescapes before MathJax sees it,
   but escaped _ doesn't participate in emphasis pairing)
"""

import re
import os
import glob


def fix_display_math(content):
    """Convert display math to multi-line format with $$ on own lines.

    When $$ is on its own line, GitHub treats the block as raw math
    and does NOT run markdown processing on the content.
    """
    lines = content.split('\n')
    result = []

    for line in lines:
        stripped = line.strip()
        indent = line[:len(line) - len(line.lstrip())] if stripped else ''

        # Case 1: $$content$$ all on one line
        if (stripped.startswith('$$') and stripped.endswith('$$')
                and len(stripped) > 4 and stripped[2:-2].strip()):
            math = stripped[2:-2]
            result.append(f'{indent}$$')
            result.append(f'{indent}{math}')
            result.append(f'{indent}$$')
            continue

        # Case 2: $$content... (opening $$ with content, no closing)
        if (stripped.startswith('$$') and not stripped.endswith('$$')
                and len(stripped) > 2):
            rest = stripped[2:]
            result.append(f'{indent}$$')
            result.append(f'{indent}{rest}')
            continue

        # Case 3: ...content$$ (closing $$ with content, no opening)
        if (stripped.endswith('$$') and not stripped.startswith('$$')
                and len(stripped) > 2):
            rest = stripped[:-2]
            result.append(f'{indent}{rest}')
            result.append(f'{indent}$$')
            continue

        result.append(line)

    return '\n'.join(result)


def find_inline_math_spans(line):
    """Find all inline $...$ spans. Returns list of (start, end) tuples.
    Skips $$ delimiters."""
    spans = []
    i = 0
    n = len(line)
    while i < n:
        if line[i] == '$':
            if i + 1 < n and line[i + 1] == '$':
                i += 2          # skip $$
                continue
            # search for closing $
            j = i + 1
            while j < n:
                if line[j] == '\\' and j + 1 < n:
                    j += 2      # skip escaped char
                    continue
                if line[j] == '$':
                    if j + 1 < n and line[j + 1] == '$':
                        j += 2  # skip $$
                        continue
                    spans.append((i, j + 1))
                    i = j + 1
                    break
                j += 1
            else:
                i += 1          # no closing $
        else:
            i += 1
    return spans


def fix_math_content(math):
    """Fix the interior of a single inline $...$ expression."""
    # \{ → \lbrace   (only when \ is not itself escaped)
    math = re.sub(r'(?<!\\)\\\{', r'\\lbrace ', math)
    # \} → \rbrace
    math = re.sub(r'(?<!\\)\\\}', r'\\rbrace ', math)
    # \| → \Vert
    math = re.sub(r'(?<!\\)\\\|', r'\\Vert ', math)
    # _ → \_   (unescaped subscript underscores)
    math = re.sub(r'(?<!\\)_', r'\\_', math)
    return math


def fix_inline_math(content):
    """Fix inline math on lines that are NOT inside display-math blocks."""
    lines = content.split('\n')
    result = []
    in_display = False

    for line in lines:
        stripped = line.strip()

        if stripped == '$$':
            in_display = not in_display
            result.append(line)
            continue

        if in_display:
            result.append(line)
            continue

        spans = find_inline_math_spans(line)
        if not spans:
            result.append(line)
            continue

        parts = []
        prev = 0
        for start, end in spans:
            parts.append(line[prev:start])
            math = line[start + 1 : end - 1]
            parts.append(f'${fix_math_content(math)}$')
            prev = end
        parts.append(line[prev:])
        result.append(''.join(parts))

    return '\n'.join(result)


def fix_file(filepath):
    with open(filepath, 'r') as f:
        content = f.read()
    original = content
    content = fix_display_math(content)   # must come first
    content = fix_inline_math(content)
    if content != original:
        with open(filepath, 'w') as f:
            f.write(content)
        return True
    return False


def main():
    base = os.path.expanduser('~/git/teaching')
    files = sorted(glob.glob(os.path.join(base, '**/*.md'), recursive=True))
    changed = 0
    for f in files:
        rel = os.path.relpath(f, base)
        if fix_file(f):
            print(f'Fixed: {rel}')
            changed += 1
    print(f'\n{changed}/{len(files)} files modified')


if __name__ == '__main__':
    main()
