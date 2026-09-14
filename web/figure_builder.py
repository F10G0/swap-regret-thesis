"""Presentation-only selection of complete, compatible replicate groups."""

from hashlib import sha256
import json
from pathlib import Path
import tempfile
from threading import Lock

from experiments.plots import figure_paths, publish_figure_pair
from experiments.plots.style import PUBLICATION_STYLE_VERSION, profile_label
from experiments.result_schema import REGRET_NAMES
from experiments.algorithm_labels import algorithm_label
from experiments.scenarios.adversarial import ENVIRONMENT_LABELS
from web.validation import FigureSelection, validate_leaf_filename


FIGURE_BUILDER_VERSION = 1
VIEWS = {"average": "Average regret (R / T)", "sqrt_scaling": "Scaling (R / sqrt(T))"}


def _digest(value) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class FigureBuilder:
    def __init__(self, service):
        self.service = service
        self.output_dir = service.results_dir / "cache" / "figure_builder"
        self._lock = Lock()

    def _contexts(self, mode: str) -> dict:
        if mode not in {"fixed", "adversarial"}:
            raise ValueError("Unknown experiment mode")
        presentations = self.service.game_presentations if mode == "fixed" else {}
        contexts = {}
        for group in self.service.result_snapshot(mode).groups("builder"):
            first = group.records[0]
            if mode == "fixed" and first.scope not in presentations:
                continue
            info = first.details
            profile = "_vs_".join(first.profile)
            label = presentations[first.scope]["label"] if mode == "fixed" else ENVIRONMENT_LABELS[first.scope]
            for player in range(len(first.profile) if mode == "fixed" else 1):
                context_id = group.context_id(player)
                context = contexts.setdefault(context_id, {
                    "id": context_id, "mode": mode, "scope": first.scope, "scope_label": label,
                    "feedback_mode": first.feedback_mode, "player": player,
                    "batch_label": (
                        f"T={first.horizon:,} · "
                        + (f"K={info.n_actions} · " if mode == "adversarial" else "")
                        + f"seed {info.seed if mode == 'fixed' else info.base_learner_seed} · "
                        + (f"env seed {info.base_environment_seed} · " if mode == "adversarial" and info.base_environment_seed is not None else "")
                        + (f"{info.stationary_method} · " if mode == "fixed" else "")
                        + "replicates " + ",".join(map(str, group.replicates))
                        + f" · {context_id[:6]}"
                    ),
                    "profiles": [], "result_keys": [], "_paths": {},
                })
                metrics = first.metrics(player)
                if not metrics:
                    continue
                context["profiles"].append({"id": profile, "label": profile_label(profile.split("_vs_")), "metrics": metrics})
                context["_paths"][profile] = group.paths
                context["result_keys"].extend([first.group_id] if mode == "fixed" else [path.name for path in group.paths])
        for context in contexts.values():
            context["profiles"].sort(key=lambda profile: profile["id"])
        return {key: context for key, context in contexts.items() if context["profiles"]}

    def catalog(self, mode: str) -> dict:
        contexts = self._contexts(mode)
        return {
            "contexts": [{key: value for key, value in context.items() if not key.startswith("_")}
                         for context in sorted(contexts.values(), key=lambda c: (c["scope"], c["feedback_mode"], c["player"], c["id"]))],
            "metrics": [{"id": name, "label": name.title()} for name in REGRET_NAMES],
            "views": [{"id": name, "label": label} for name, label in VIEWS.items()],
            "scaling": [{"scope": record.scope, "scope_label": ENVIRONMENT_LABELS[record.scope],
                         "feedback_mode": record.feedback_mode, "player": 0,
                         "profiles": [{"id": record.profile[0], "label": algorithm_label(record.profile[0])}]}
                        for record in self.service.result_snapshot("scaling").records] if mode == "adversarial" else [],
        }

    @staticmethod
    def _sources(paths) -> list:
        return [(str(path.resolve()), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
                for path in paths for stat in [path.stat()]]

    def build(self, selection) -> dict:
        return self._build(selection, self._contexts(selection.mode).get(selection.context_id), {})

    def build_collection(self, selection) -> dict:
        context = self._contexts(selection.mode).get(selection.context_id)
        paths = [path for profile in sorted(set(selection.profiles))
                 for path in context["_paths"].get(profile, [])] if context else []
        sources = self._sources(paths)
        loaded = {}  # All six views share one read of each selected replicate.
        figures = [self._build(FigureSelection(selection.mode, selection.context_id, metric, view, selection.profiles),
                               context, loaded)
                   for metric in REGRET_NAMES for view in VIEWS]
        if self._sources(paths) != sources:
            raise ValueError("Results changed during rendering. Refresh and try again.")
        return {"figures": figures, "profiles": figures[0]["profiles"]}

    def _build(self, selection, context, loaded) -> dict:
        from experiments.plots import plot_regret as fixed
        from experiments.plots import plot_adversarial as adversarial

        profiles = tuple(sorted(set(selection.profiles)))
        if not profiles:
            raise ValueError("Select at least one algorithm profile")
        if selection.metric not in REGRET_NAMES or selection.view not in VIEWS:
            raise ValueError("Unknown regret metric or view")
        if context is None:
            raise ValueError("This result set is unavailable. Refresh the available profiles.")
        available = {profile["id"] for profile in context["profiles"] if selection.metric in profile["metrics"]}
        if not set(profiles) <= available:
            raise ValueError("A selected profile is incompatible with this result set or metric")
        paths = [path for profile in profiles for path in context["_paths"][profile]]
        sources = self._sources(paths)
        artifact_id = _digest({
            "builder_version": FIGURE_BUILDER_VERSION, "style_version": PUBLICATION_STYLE_VERSION,
            "context": context["id"], "metric": selection.metric, "view": selection.view,
            "profiles": profiles, "sources": sources,
        })
        filename = f"{context['scope']}_{context['feedback_mode']}_{selection.metric}_{selection.view}_p{context['player']}_{artifact_id}.png"
        output_path = self.output_dir / filename
        source_keys = dict(zip(paths, sources))

        def load(path):
            if loaded.get(path, (None,))[0] != source_keys[path]:
                loader = fixed.load_rows if selection.mode == "fixed" else adversarial.load_adversarial_rows
                loaded[path] = (source_keys[path], loader(path))
            return loaded[path][1]

        with self._lock:
            if not all(path.is_file() for path in figure_paths(output_path)):
                self.output_dir.mkdir(parents=True, exist_ok=True)
                with tempfile.TemporaryDirectory(prefix=".selection-", dir=self.output_dir) as directory:
                    temporary = Path(directory)
                    if selection.mode == "fixed":
                        # Preserve each complete replicate group and the existing averaging code.
                        groups = [[load(path) for path in context["_paths"][profile]]
                                  for profile in profiles]
                        fixed.plot_regret(context["scope"], groups, selection.metric, context["player"],
                                          selection.view == "average", temporary)
                        source_path = next(temporary.glob("*.png"))
                    else:
                        rows = [(path, load(path)) for path in paths]
                        source_path = temporary / filename
                        adversarial._plot_regret(rows, context["scope"], context["feedback_mode"],
                                                 int(rows[0][1][0]["n_actions"]), selection.metric,
                                                 selection.view == "average", source_path)
                    if self._sources(paths) != sources:
                        raise ValueError("Results changed during rendering. Refresh and try again.")
                    publish_figure_pair(source_path, output_path)
        return {"artifact_id": artifact_id, "filename": filename,
                "pdf_filename": output_path.with_suffix(".pdf").name,
                "profiles": list(profiles),
                "metric": selection.metric, "view": selection.view,
                "title": f"{context['scope_label']} · Player {context['player']} · {selection.metric.title()} · {VIEWS[selection.view]}"}

    def artifact_path(self, filename: str) -> Path:
        suffix = Path(filename).suffix.lower()
        if suffix not in {".png", ".pdf"}:
            raise ValueError("Invalid figure format")
        filename = validate_leaf_filename(filename, suffix)
        path = self.output_dir / filename
        if not path.is_file() or path.resolve().parent != self.output_dir.resolve():
            raise FileNotFoundError(filename)
        # Flask resolves relative send_file paths against its app directory,
        # whereas result directories are relative to the working directory.
        return path.resolve()
