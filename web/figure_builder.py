"""Presentation-only selection of complete, compatible replicate groups."""

from collections import defaultdict
from dataclasses import replace
from hashlib import sha256
import json
from pathlib import Path
import tempfile
from threading import Lock

import numpy as np

from experiments.algorithm_labels import algorithm_profile_label
from experiments.plots import figure_paths, publish_figure_pair
from experiments.plots.pdf_information import ENDPOINT_STATISTICS_DESCRIPTION, format_value_summary
from experiments.plots.style import profile_series_style, regret_axis_label, regret_comparison_axis_label, regret_series_style
from experiments.results import average_regret_column, regret_column
from experiments.result_schema import REGRET_NAMES
from experiments.scenarios.adversarial import ENVIRONMENT_LABELS
from experiments.scenarios.cross_play import FEEDBACK_MODE_LABELS
from web.validation import FigureSelection, validate_leaf_filename


VIEWS = {"average": "Average regret (R / T)", "sqrt_scaling": "Scaling (R / sqrt(T))"}
HORIZON_SCALING_VIEW = "horizon_scaling"
VIEW_LABELS = VIEWS | {HORIZON_SCALING_VIEW: "Horizon scaling"}


def _digest(value) -> str:
    return sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


class FigureBuilder:
    def __init__(self, service):
        self.service = service
        self.output_dir = service.results_dir / "cache" / "figure_builder"
        self._lock = Lock()

    def _contexts(self, mode: str, results=None) -> dict:
        if mode not in {"fixed", "adversarial"}:
            raise ValueError("Unknown experiment mode")
        presentations = self.service.game_presentations if mode == "fixed" else {}
        groups = (results if results is not None else self.service.result_snapshot(mode)).groups("builder")
        compatible = defaultdict(list)
        for group in groups:
            first = group.records[0]
            info = first.details
            if mode == "fixed":
                if first.scope not in presentations:
                    continue
                key = (first.scope, info.game_payoff_digest, info.seed, info.stationary_method,
                       first.runtime_environment, first.runtime_fingerprint, tuple(group.replicates))
            else:
                key = (first.scope, info.base_learner_seed, info.base_environment_seed,
                       first.runtime_environment, first.runtime_fingerprint, tuple(group.replicates))
            compatible[key].append(group)
        contexts = {}
        for key, family in compatible.items():
            first = family[0].records[0]
            info = first.details
            players = range(len(first.profile)) if mode == "fixed" else range(1)
            for player in players:
                context_id = _digest((mode, key, player))[:24]
                base_seed = info.seed if mode == "fixed" else info.base_learner_seed
                context = {
                    "id": context_id, "mode": mode, "scope": first.scope,
                    "scope_label": (presentations[first.scope]["label"] if mode == "fixed"
                                    else ENVIRONMENT_LABELS[first.scope]),
                    "player": player, "base_seed": base_seed,
                    "batch_label": f"Seed {base_seed} · {context_id[:6]}",
                    "actions": [], "profiles": [], "result_keys": [], "_paths": {},
                    "_replicates": tuple(family[0].replicates),
                }
                profile_data = {}
                for group in family:
                    result = group.records[0]
                    profile = "_vs_".join(result.profile)
                    metrics = set(result.metrics(player))
                    if not metrics:
                        continue
                    data = profile_data.get(profile)
                    if data is None:
                        data = profile_data[profile] = {
                            "id": profile, "feedback_mode": result.feedback_mode,
                            "label": algorithm_profile_label(result.profile), "metrics": metrics,
                            "horizons": set(), "availability": defaultdict(set),
                        }
                    else:
                        data["metrics"].intersection_update(metrics)
                    horizon = str(result.horizon)
                    data["horizons"].add(result.horizon)
                    if mode == "fixed":
                        context["_paths"].setdefault(horizon, {})[profile] = group.paths
                    else:
                        action = str(result.details.n_actions)
                        data["availability"][action].add(result.horizon)
                        context["_paths"].setdefault(action, {}).setdefault(horizon, {})[profile] = group.paths
                    context["result_keys"].append(result.group_id)
                profiles = []
                for _, data in sorted(profile_data.items()):
                    profile = {key: value for key, value in data.items() if key != "availability"}
                    profile["metrics"] = sorted(data["metrics"])
                    profile["horizons"] = sorted(data["horizons"])
                    if mode == "adversarial":
                        profile["availability"] = {
                            action: sorted(horizons) for action, horizons in sorted(data["availability"].items(), key=lambda item: int(item[0]))
                        }
                        profile["actions"] = sorted(map(int, data["availability"]))
                    profiles.append(profile)
                context["profiles"] = profiles
                if context["profiles"]:
                    context["horizons"] = sorted({horizon for profile in profiles for horizon in profile["horizons"]})
                    if mode == "adversarial":
                        context["actions"] = sorted(map(int, context["_paths"]))
                    context["feedback_modes"] = sorted({profile["feedback_mode"] for profile in profiles})
                    context["comparison_modes"] = ["regrets", "profiles"]
                    if mode == "adversarial":
                        context["comparison_modes"].append("actions")
                    if any(len(profile["horizons"]) >= 2 for profile in profiles):
                        context["comparison_modes"].append("horizons")
                    contexts[context_id] = context
        return contexts

    def catalog(self, mode: str, results=None) -> dict:
        contexts = self._contexts(mode, results)
        return {
            "contexts": [{key: value for key, value in context.items() if not key.startswith("_")}
                         for context in sorted(contexts.values(), key=lambda c: (c["scope"], c["player"], c["id"]))],
            "metrics": [{"id": name, "label": name.title()} for name in REGRET_NAMES],
            "views": [{"id": name, "label": label} for name, label in VIEW_LABELS.items()],
        }

    @staticmethod
    def _sources(paths) -> list:
        return [(str(path.resolve()), stat.st_size, stat.st_mtime_ns, stat.st_ctime_ns)
                for path in paths for stat in [path.stat()]]

    def build_collection(self, selection) -> dict:
        return self._collection(selection, render=True)

    def cached_collection(self, selection) -> dict:
        return self._collection(selection, render=False)

    def _collection(self, selection, render: bool) -> dict:
        generation = self.service._derived_generation() if render else None
        context = self._contexts(selection.mode).get(selection.context_id)
        profiles = selection.profiles
        if profiles == ("all",):
            if context is None:
                raise ValueError("This result set is unavailable. Refresh the available profiles.")
            profiles = tuple(profile["id"] for profile in context["profiles"]
                             if (selection.feedback == "both" or profile["feedback_mode"] == selection.feedback)
                             and set(REGRET_NAMES) <= set(profile["metrics"])
                             and (selection.comparison_mode != "horizons"
                                  or (selection.mode == "fixed" and len(profile["horizons"]) >= 2)
                                  or (selection.mode == "adversarial"
                                      and len(profile["availability"].get(selection.action, ())) >= 2))
                             and self._selected_paths(replace(selection, profiles=(profile["id"],)), context))
            if not profiles:
                raise ValueError("No compatible algorithm profiles are available for this selection")
        selections = ([replace(selection, profiles=(profile,)) for profile in profiles]
                      if selection.comparison_mode in {"regrets", "horizons"} else [selection])
        paths = list(dict.fromkeys(path for item in selections
                                  for path in (self._selected_paths(item, context) if context else [])))
        sources = self._sources(paths)
        loaded = {}
        if selection.comparison_mode == "regrets":
            figure_specs = [("all", view) for view in VIEWS]
        elif selection.comparison_mode == "horizons":
            figure_specs = [("all", HORIZON_SCALING_VIEW)]
        else:
            figure_specs = [(metric, view) for metric in REGRET_NAMES for view in VIEWS]
        figures = []
        for item in selections:
            for metric, view in figure_specs:
                figure = self._build(FigureSelection(item.mode, item.context_id, item.comparison_mode,
                                                     metric, view, item.profiles, item.action,
                                                     item.horizon), context, loaded, render, generation)
                if figure is None:
                    return {"figures": [], "profiles": list(profiles),
                            "comparison_mode": selection.comparison_mode, "cached": False}
                figures.append(figure)
        if self._sources(paths) != sources:
            raise ValueError("Results changed during rendering. Refresh and try again.")
        if render and generation != self.service._derived_generation():
            raise RuntimeError("derived artifact generation was invalidated")
        return {"figures": figures, "profiles": list(profiles),
                "comparison_mode": selection.comparison_mode, "cached": True}

    @staticmethod
    def _selected_paths(selection, context) -> list[Path]:
        if selection.mode == "fixed":
            horizons = sorted(context["_paths"], key=int) if selection.comparison_mode == "horizons" else [selection.horizon]
            return [path for horizon in horizons for profile in selection.profiles
                    for path in context["_paths"].get(horizon, {}).get(profile, [])]
        if selection.comparison_mode == "horizons":
            profile = selection.profiles[0]
            action_paths = context["_paths"].get(selection.action, {})
            return [path for horizon in sorted(action_paths, key=int)
                    for path in action_paths[horizon].get(profile, [])]
        if selection.comparison_mode == "actions":
            profile = selection.profiles[0]
            return [path for action in sorted(context["_paths"], key=int)
                    for path in context["_paths"][action].get(selection.horizon, {}).get(profile, [])]
        horizon_paths = context["_paths"].get(selection.action, {}).get(selection.horizon, {})
        return [path for profile in selection.profiles for path in horizon_paths.get(profile, [])]

    @staticmethod
    def _information_rows(selection, context, profiles) -> list[tuple[str, str]]:
        profile_data = {profile["id"]: profile for profile in context["profiles"]}
        selected = [profile_data[profile] for profile in profiles]
        feedback_modes = {profile["feedback_mode"] for profile in selected}
        rows = [
            ("Figure", {"profiles": "Algorithm comparison", "regrets": "Regret-notion comparison",
                        "actions": "Action-space comparison", "horizons": "Horizon scaling"}[selection.comparison_mode]),
            ("Game" if selection.mode == "fixed" else "Environment", context["scope_label"]),
        ]
        if selection.comparison_mode == "horizons":
            rows.insert(1, ("Compare", "Horizons"))
        if selection.mode == "fixed":
            rows.append(("Player", str(context["player"])))
        elif selection.comparison_mode == "actions":
            actions = [action for action in sorted(context["_paths"], key=int)
                       if profiles[0] in context["_paths"][action].get(selection.horizon, {})]
            rows.append(("Actions", ", ".join(actions)))
        elif selection.comparison_mode == "horizons":
            rows.append(("Actions", selection.action))
        else:
            rows.append(("Actions", selection.action))
        rows.append(("Feedback", "Both" if len(feedback_modes) > 1 else FEEDBACK_MODE_LABELS[next(iter(feedback_modes))]))
        if selection.comparison_mode == "profiles":
            labels = []
            for profile in selected:
                label = profile["label"]
                if len(feedback_modes) > 1:
                    label += f" [{FEEDBACK_MODE_LABELS[profile['feedback_mode']]}]"
                labels.append(label)
            rows.append(("Algorithms", ", ".join(labels)))
        else:
            rows.append(("Profile" if selection.mode == "fixed" else "Algorithm", selected[0]["label"]))
        if selection.comparison_mode == "horizons":
            horizon_paths = (context["_paths"] if selection.mode == "fixed"
                             else context["_paths"].get(selection.action, {}))
            horizons = [horizon for horizon in sorted(horizon_paths, key=int)
                        if profiles[0] in horizon_paths[horizon]]
            rows.extend([
                ("Horizons", ", ".join(f"{int(horizon):,}" for horizon in horizons)),
                ("Replicates", str(len(context["_replicates"]))),
                ("Seed", str(context["base_seed"])),
                ("Regrets", "External, Internal, Swap"),
                ("Aggregation", "Mean final cumulative action regret across replicates"),
                ("Fit model", "log R_T = α log T + log c"),
                ("Fit points", str(len(horizons))),
                ("Fit range", f"{int(horizons[0]):,} to {int(horizons[-1]):,}"),
            ])
            return rows
        rows.extend([("Horizon", f"{int(selection.horizon):,}"),
                     ("Replicates", str(len(context["_replicates"]))), ("Seed", str(context["base_seed"]))])
        if selection.comparison_mode == "regrets":
            rows.append(("Regrets", "External, Internal, Swap"))
        else:
            rows.append(("Metric", f"{selection.metric.title()} regret"))
        rows.append(("View", "R / T" if selection.view == "average" else "R / sqrt(T)"))
        return rows

    def _build(self, selection, context, loaded, render: bool, generation: int | None) -> dict | None:
        profiles = tuple(sorted(set(selection.profiles)))
        if not profiles:
            raise ValueError("Select at least one algorithm profile")
        if selection.comparison_mode not in {"profiles", "regrets", "actions", "horizons"} or selection.view not in VIEW_LABELS:
            raise ValueError("Unknown regret metric or view")
        if (selection.comparison_mode == "horizons") != (selection.view == HORIZON_SCALING_VIEW):
            raise ValueError("The horizon-scaling view is available only for horizon comparison")
        if selection.comparison_mode not in {"regrets", "horizons"} and selection.metric not in REGRET_NAMES:
            raise ValueError("Unknown regret metric or view")
        if selection.comparison_mode == "regrets" and (selection.metric != "all" or len(profiles) != 1):
            raise ValueError("Regret-notion comparison requires exactly one profile and all regret notions")
        if selection.comparison_mode == "actions" and (selection.mode != "adversarial" or len(profiles) != 1):
            raise ValueError("Action-space comparison requires exactly one one-player profile")
        if selection.comparison_mode == "horizons" and len(profiles) != 1:
            raise ValueError("Horizon comparison requires exactly one profile")
        if selection.comparison_mode == "horizons" and selection.metric != "all":
            raise ValueError("Horizon comparison includes all regret notions")
        if context is None:
            raise ValueError("This result set is unavailable. Refresh the available profiles.")
        required_metrics = REGRET_NAMES if selection.comparison_mode in {"regrets", "horizons"} else (selection.metric,)
        available = {profile["id"] for profile in context["profiles"]
                     if set(required_metrics) <= set(profile["metrics"])}
        if not set(profiles) <= available:
            raise ValueError("A selected profile is incompatible with this result set or metric")
        paths = self._selected_paths(selection, context)
        if not paths:
            raise ValueError("No compatible results are available for this action and horizon selection")
        sources = self._sources(paths)
        artifact_identity = {
            "context": context["id"], "comparison_mode": selection.comparison_mode,
            "metric": selection.metric, "view": selection.view,
            "profiles": profiles, "horizon": selection.horizon, "sources": sources,
        }
        if selection.mode == "adversarial":
            artifact_identity["action"] = selection.action
        artifact_id = _digest(artifact_identity)
        dimension = f"p{context['player']}" if selection.mode == "fixed" else (
            "actions" if selection.comparison_mode == "actions" else f"k{selection.action}")
        profile_modes = {profile["id"]: profile["feedback_mode"] for profile in context["profiles"]}
        feedback = "both" if len({profile_modes[profile] for profile in profiles}) > 1 else profile_modes[profiles[0]]
        filename = f"{context['scope']}_{feedback}_{selection.metric}_{selection.view}_{dimension}_{artifact_id}.png"
        output_path = self.output_dir / filename
        title_metric = "Regret notions" if selection.view == HORIZON_SCALING_VIEW else (
            "Regret notions" if selection.comparison_mode == "regrets" else selection.metric.title())
        title_dimension = f"Player {context['player']}" if selection.mode == "fixed" else (
            "Action spaces" if selection.comparison_mode == "actions" else f"K={selection.action}")
        if selection.comparison_mode in {"regrets", "horizons"}:
            title_dimension += " · " + next(profile["label"] for profile in context["profiles"]
                                            if profile["id"] == profiles[0])
        result = {"artifact_id": artifact_id, "filename": filename,
                  "pdf_filename": output_path.with_suffix(".pdf").name,
                  "profiles": list(profiles),
                  "comparison_mode": selection.comparison_mode,
                  "metric": selection.metric, "view": selection.view,
                  "title": f"{context['scope_label']} · {title_dimension} · {title_metric} · {VIEW_LABELS[selection.view]}"}
        if all(path.is_file() for path in figure_paths(output_path)):
            return result
        if not render:
            return None

        from experiments.plots import plot_adversarial as adversarial
        from experiments.plots import plot_regret as fixed
        from experiments.scenarios.adversarial import load_adversarial_rows

        source_keys = dict(zip(paths, sources))

        def load(path):
            if loaded.get(path, (None,))[0] != source_keys[path]:
                loader = fixed.load_rows if selection.mode == "fixed" else load_adversarial_rows
                loaded[path] = (source_keys[path], loader(path))
            return loaded[path][1]

        average = selection.view == "average"
        sqrt_scaling = selection.view == "sqrt_scaling"

        def selected_runs(profile, action=selection.action, horizon=selection.horizon):
            profile_paths = (context["_paths"][str(horizon)][profile] if selection.mode == "fixed"
                             else context["_paths"][str(action)][str(horizon)][profile])
            return [load(path) for path in profile_paths]

        def final_values(profile, metric, action=selection.action, horizon=selection.horizon,
                         normalization=selection.view):
            runs = selected_runs(profile, action, horizon)
            column = average_regret_column(metric) if normalization == "average" else regret_column(metric)
            values = np.asarray([
                fixed.aggregate_final_metric([run], context["player"], column)
                if selection.mode == "fixed" else adversarial.aggregate_final_adversarial_regret([run], column)
                for run in runs
            ])
            if normalization == "sqrt_scaling":
                values /= np.sqrt(int(horizon))
            return values

        def curve(profile, metric, label, style, action=selection.action):
            runs = selected_runs(profile, action)
            column = average_regret_column(metric) if average else regret_column(metric)
            if selection.mode == "fixed":
                x, y = fixed.aggregate_metric_curve(runs, context["player"], column,
                                                    divide_by_sqrt_time=sqrt_scaling)
            else:
                x, y = adversarial.aggregate_adversarial_regret(runs, column,
                                                                 scale_by_sqrt_time=sqrt_scaling)
            return fixed.RegretCurve(x, y, label, style)

        labels = {profile["id"]: profile["label"] for profile in context["profiles"]}
        if selection.comparison_mode == "horizons":
            horizon_paths = (context["_paths"] if selection.mode == "fixed"
                             else context["_paths"][selection.action])
            horizon_values = [int(horizon) for horizon in sorted(horizon_paths, key=int)
                              if profiles[0] in horizon_paths[horizon]]
            def make_curves():
                curves = []
                for index, metric in enumerate(REGRET_NAMES):
                    means = []
                    for horizon in horizon_values:
                        runs = [load(path) for path in horizon_paths[str(horizon)][profiles[0]]]
                        column = regret_column(metric)
                        mean = (fixed.aggregate_final_metric(runs, context["player"], column)
                                if selection.mode == "fixed"
                                else adversarial.aggregate_final_adversarial_regret(runs, column))
                        means.append(mean)
                    curves.append(fixed.RegretCurve(np.asarray(horizon_values), np.asarray(means),
                                  f"{metric.title()} regret", regret_series_style(metric, index, len(REGRET_NAMES))))
                return curves
            endpoint_horizon = horizon_values[-1]
            endpoint_series = [(f"{metric.title()} R/T",
                                final_values(profiles[0], metric, horizon=endpoint_horizon, normalization="average"))
                               for metric in REGRET_NAMES]
            y_label = None
        elif selection.comparison_mode == "profiles":
            def make_curves():
                return [curve(profile, selection.metric, labels[profile], profile_series_style(index, len(profiles)))
                        for index, profile in enumerate(profiles)]
            endpoint_horizon = int(selection.horizon)
            endpoint_series = [(labels[profile], final_values(profile, selection.metric)) for profile in profiles]
            y_label = regret_axis_label(selection.metric, selection.view)
        elif selection.comparison_mode == "regrets":
            def make_curves():
                return [curve(profiles[0], metric, f"{metric.title()} regret",
                              regret_series_style(metric, index, len(REGRET_NAMES)))
                        for index, metric in enumerate(REGRET_NAMES)]
            endpoint_horizon = int(selection.horizon)
            endpoint_series = [(metric.title(), final_values(profiles[0], metric)) for metric in REGRET_NAMES]
            y_label = regret_comparison_axis_label(selection.view)
        else:
            actions = [action for action in sorted(context["_paths"], key=int)
                       if profiles[0] in context["_paths"][action].get(selection.horizon, {})]
            def make_curves():
                return [curve(profiles[0], selection.metric, f"K={action}",
                              profile_series_style(index, len(actions)), action)
                        for index, action in enumerate(actions)]
            endpoint_horizon = int(selection.horizon)
            endpoint_series = [(f"K={action}", final_values(profiles[0], selection.metric, action))
                               for action in actions]
            y_label = regret_axis_label(selection.metric, selection.view)

        with self._lock:
            if not all(path.is_file() for path in figure_paths(output_path)):
                with tempfile.TemporaryDirectory(prefix=".selection-", dir=self.service.results_dir) as directory:
                    source_path = Path(directory) / filename
                    information_rows = self._information_rows(selection, context, profiles)
                    curves = make_curves()
                    information_rows.extend([
                        ("Final endpoint at T" if selection.view == HORIZON_SCALING_VIEW else "Final endpoints at T",
                         f"{endpoint_horizon:,}"),
                        ("Endpoint statistics", ENDPOINT_STATISTICS_DESCRIPTION),
                        *((label, format_value_summary(values)) for label, values in endpoint_series),
                    ])
                    if selection.view == HORIZON_SCALING_VIEW:
                        fixed.plot_horizon_scaling(curves, source_path,
                                                   information_rows=information_rows)
                    elif selection.comparison_mode == "regrets":
                        fixed.plot_regret_curves(curves, y_label, source_path,
                                                 information_rows=information_rows, legend_ncol=1)
                    else:
                        fixed.plot_regret_curves(curves, y_label, source_path, information_rows=information_rows)
                    if self._sources(paths) != sources:
                        raise ValueError("Results changed during rendering. Refresh and try again.")
                    def publish() -> None:
                        self.output_dir.mkdir(parents=True, exist_ok=True)
                        publish_figure_pair(source_path, output_path)

                    self.service._publish_derived_artifact(generation, publish)
        return result

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
