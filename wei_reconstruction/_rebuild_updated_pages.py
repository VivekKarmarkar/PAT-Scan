#!/usr/bin/env python3
"""
One-shot helper: generate wei_pageN_updated.tex for N in 1..9 by copying
the originals and applying the footer-convention change documented in
NOTES.md §2a.

Mechanical substitutions (applied per file):
  1. `\\setcounter{page}{<orig>}`  →  `\\setcounter{page}{1}`
  2. `\\cfoot{\\small \\thepage}`   →  `\\cfoot{\\small <orig> $\\to$ \\arabic{page}}`

Originals (wei_page1.tex .. wei_page9.tex) are NOT modified — this script
only WRITES wei_page{n}_updated.tex files alongside them.
"""

import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

for n in range(1, 10):
    orig_page = 125 + n  # journal page number (126..134)
    src = f"wei_page{n}.tex"
    dst = f"wei_page{n}_updated.tex"

    with open(src, "r") as f:
        content = f.read()

    old_setcounter = f"\\setcounter{{page}}{{{orig_page}}}"
    new_setcounter = "\\setcounter{page}{1}"

    old_cfoot = "\\cfoot{\\small \\thepage}"
    new_cfoot = f"\\cfoot{{\\small {orig_page} $\\to$ \\arabic{{page}}}}"

    # Sanity check: both old strings must be present before we substitute
    assert old_setcounter in content, f"{src}: could not find {old_setcounter!r}"
    assert old_cfoot in content, f"{src}: could not find {old_cfoot!r}"

    new_content = content.replace(old_setcounter, new_setcounter)
    new_content = new_content.replace(old_cfoot, new_cfoot)

    with open(dst, "w") as f:
        f.write(new_content)

    print(f"wrote {dst} (orig page {orig_page} -> footer '{orig_page} -> N')")

print("\nAll 9 _updated.tex files written. Now compile with pdflatex.")
