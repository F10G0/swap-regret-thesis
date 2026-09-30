# Thesis Documentation

This directory contains the documents and source material associated with the
bachelor thesis **An Empirical Comparison of Regret-Minimizing Algorithms**.

It is separate from the main experiment outputs, which are stored under
`results/`.

## Structure

### `thesis/`

Contains the final submitted bachelor thesis and a buildable version of its
LaTeX source.

- `An_Empirical_Comparison_of_Regret_Minimizing_Algorithms.pdf`
  is the final submitted thesis and is preserved unchanged as the authoritative
  archival artifact.
- `source/`
  contains the corresponding LaTeX source, bibliography, front matter, build
  files, and the figures required to build the final thesis.

The source tree has been cleaned for repository use: duplicate build artifacts,
unused experimental figures, and auxiliary figure-archive files that are not
required by the final thesis have been removed. The manuscript source and all
assets required to build the final document are retained.

The submitted PDF should not be modified to reflect later changes to the
codebase or documentation.

### `presentation/`

Contains the bachelor thesis defense presentation and its LaTeX/Beamer source.

- `Bachelors_Thesis_Defense_Presentation.pdf`
  is the final presentation used for the thesis defense.
- `source/`
  contains the corresponding LaTeX/Beamer source, bibliography, figures,
  slides, backup material, and theme files.

These files are preserved as the historical version used for the defense.

### `proposal/`

Contains material from the planning and scoping stage of the thesis.

- `supervisor_project_description.pdf`
  is the original project description provided for the bachelor thesis.
- `bachelor_thesis_proposal.pdf`
  is the subsequent thesis proposal.
- `proposal.tex`
  contains the LaTeX source of the proposal.
- `refs.bib`
  contains the proposal bibliography.

These files are retained to document the development of the project from its
initial scope to the final thesis.

## External literature

Third-party research papers used during the project are not redistributed in
this repository. The relevant literature is cited in the thesis, presentation,
and their bibliography files.

## Generated experiment outputs

General experiment outputs such as raw CSV files, generated figure collections,
caches, dashboard-created games, and other runtime artifacts belong under
`results/` or the corresponding runtime output directories.

Figures that are directly required to build the thesis or presentation are
retained with their respective source trees under `docs/`.

The material in this directory is preserved primarily for documentation,
reproducibility, and archival purposes.
