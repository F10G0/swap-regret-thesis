# An Empirical Comparison of Regret-Minimizing Algorithms

LaTeX source for the bachelor thesis.

## Build

```bash
make pdf
```

or directly with `latexmk`:

```bash
latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
```

A standard LaTeX installation with BibTeX is required.

## Project structure

- `main.tex` -- main document entry point
- `settings.tex` -- document and package configuration
- `bibliography.bib` -- bibliography database
- `chapters/` -- thesis chapters and appendix
- `pages/` -- front matter (cover, title page, and abstract)
- `figures/regret/` -- one-player experiment figures used by the final thesis
- `figures/games/` -- repeated-game figures used by the final thesis
- `logos/` -- institutional logos used by the front matter

Only figure files referenced by the final submitted thesis are retained here.
The broader experimental framework and regenerable experiment outputs are maintained
elsewhere in this repository rather than in this archival LaTeX source.

The compiled final thesis PDF is stored separately from this source directory.
