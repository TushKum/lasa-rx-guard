# LASA-RxGuard — paper source

IEEE conference format (`IEEEtran`). Every number in the paper comes from
`outputs/reports/*.json` in this repository; nothing is illustrative.

## Upload to Overleaf

1. Zip this folder (or use the `lasa-rx-guard-paper.zip` already built):

   ```bash
   cd paper && zip -r ../lasa-rx-guard-paper.zip . -x "*.aux" "*.bbl" "*.blg" "*.log" "*.out"
   ```

2. Overleaf → **New Project** → **Upload Project** → drop the zip.
3. Set the compiler to **pdfLaTeX** (Menu → Compiler). `IEEEtran` ships with
   Overleaf, so nothing else is needed.
4. Press **Recompile**. Citations need two passes; Overleaf handles this.

## Files

| File | Purpose |
|---|---|
| `main.tex` | The paper |
| `references.bib` | 25 references, all verified against publisher/index records |
| `figures/synth_preview.png` | Synthesised samples across three tiers |
| `figures/demo_red.png` | Live risk-flagging output (unused by default — see below) |

## Before you submit

- **Fill in the author block.** `Your College Name` and `City` are
  placeholders in `main.tex`.
- **Add co-authors** if this is a team submission.
- `figures/demo_red.png` is included but not placed in the text. Add it with
  `\includegraphics` if you want a system-output figure.

## Regenerating the numbers

```bash
python -m lasa.evaluation.evaluate          # Table II, IV
python -m lasa.risk.evaluate_lasa --markdown  # Tables V, VI, VII
```
