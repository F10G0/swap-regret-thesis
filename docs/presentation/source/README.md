# Bachelor's Thesis Defense Slides

LaTeX/Beamer source for the defense of **An Empirical Comparison of Regret-Minimizing Algorithms**.

## Build

From the project directory:

```bash
latexmk -pdf main.tex
```

The bibliography is handled by `biber` through `latexmk`.

## Structure

- `main.tex`: document setup, title slide, closing slide, backup integration, and final References section.
- `Slides/00_...` to `Slides/12_...`: 13 source files for the 14-slide main presentation; the title slide is defined directly in `main.tex`, and some result slides use Beamer overlays.
- `Slides/Backup/`: seven Q&A backup slides:
  1. formal action-regret definitions and relations;
  2. expected action regret versus distribution regret and pseudo-regret;
  3. literature guarantees and comparison caveats;
  4. Blum--Mansour and Ito reductions;
  5. exact HFA and LRW reward processes;
  6. formal CE/CCE conditions and the CE/CCE distance computation;
  7. finite-horizon scaling protocol.
- `figures/`: vector result figures used by the presentation.
- `bibliography.bib`: bibliography database used by citations throughout the deck and by the References section.

## Presentation structure

The main presentation is followed by a closing **Thank you.** slide, then the Q&A backup material, and finally the References section.

The References section contains only bibliography entries that are actually cited somewhere in the presentation; `\nocite{*}` is intentionally not used.

## Notes

- Main trajectory experiments use `T = 10^6` and 12 independent replicates unless stated otherwise.
- The common empirical evaluator is expected action regret.
- Fitted parameters in the horizon-scaling material are finite-horizon descriptors, not asymptotic regret-rate estimates.
- Backup slides are intended only for questions and are not part of the timed 15-minute main presentation.
