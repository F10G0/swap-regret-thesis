"""Web-facing definitions for opt-in trajectory comparisons."""

from dataclasses import dataclass
from hashlib import sha256
import colorsys
import json
from pathlib import Path


_HIGH_CONTRAST_COLORS = (
    "#1f77b4",
    "#ff7f0e",
    "#2ca02c",
    "#d62728",
    "#9467bd",
    "#8c564b",
    "#e377c2",
    "#4d4d4d",
    "#bcbd22",
    "#17becf",
    "#332288",
    "#44aa99",
    "#aa4499",
    "#117733",
    "#882255",
    "#6699cc",
)


@dataclass(frozen=True)
class TrajectoryComparisonMember:
    group_id: str
    label: str
    color: str
    algorithm_profile: tuple[str, ...]
    stationary_method: str
    replicate_indices: tuple[int, ...]
    player_seed_schedule: tuple[tuple[int, ...], ...]
    input_paths: tuple[Path, ...]

    def public_data(self) -> dict:
        return {
            "group_id": self.group_id,
            "label": self.label,
            "color": self.color,
            "algorithm_profile": list(self.algorithm_profile),
            "stationary_method": self.stationary_method,
            "replicate_count": len(self.replicate_indices),
            "replicate_indices": list(self.replicate_indices),
            "player_seed_schedule": [
                list(seeds) for seeds in self.player_seed_schedule
            ],
        }


@dataclass(frozen=True)
class TrajectoryComparisonDefinition:
    members: tuple[TrajectoryComparisonMember, ...]
    compatibility_key: tuple
    final_interval_segments: int
    focus_final_interval: bool
    artifact_id: str
    comparison_view: str = "geometry"

    @property
    def input_paths(self) -> tuple[Path, ...]:
        return tuple(
            path
            for member in self.members
            for path in member.input_paths
        )


@dataclass(frozen=True)
class TrajectoryComparisonResult:
    definition: TrajectoryComparisonDefinition
    output_path: Path

    def public_data(self, image_url: str, pdf_url: str | None = None) -> dict:
        return {
            "artifact_id": self.definition.artifact_id,
            "image_url": image_url,
            "pdf_url": pdf_url or image_url,
            "members": [
                member.public_data()
                for member in self.definition.members
            ],
            "final_interval_segments": (
                self.definition.final_interval_segments
            ),
            "focus_final_interval": (
                self.definition.focus_final_interval
            ),
            "comparison_view": self.definition.comparison_view,
        }


def stable_member_color(group_id: str) -> str:
    hue = int(sha256(group_id.encode("utf-8")).hexdigest()[:8], 16)
    red, green, blue = colorsys.hsv_to_rgb(
        (hue % 360) / 360.0,
        0.72,
        0.72,
    )
    return "#{:02x}{:02x}{:02x}".format(
        round(255 * red),
        round(255 * green),
        round(255 * blue),
    )


def comparison_member_colors(group_ids) -> dict[str, str]:
    """Assign deterministic contrasting colors to one comparison set."""
    canonical_ids = sorted(set(group_ids))
    if len(canonical_ids) == 1:
        return {
            canonical_ids[0]: stable_member_color(canonical_ids[0])
        }
    colors = list(_HIGH_CONTRAST_COLORS[:len(canonical_ids)])
    used = set(colors)
    for position in range(len(colors), len(canonical_ids)):
        hue = (0.61803398875 * position) % 1.0
        saturation = 0.65 + 0.2 * (position % 2)
        value = 0.62 + 0.18 * ((position // 2) % 2)
        while True:
            red, green, blue = colorsys.hsv_to_rgb(
                hue,
                saturation,
                value,
            )
            color = "#{:02x}{:02x}{:02x}".format(
                round(255 * red),
                round(255 * green),
                round(255 * blue),
            )
            if color not in used:
                break
            hue = (hue + 0.037) % 1.0
        colors.append(color)
        used.add(color)
    return dict(zip(canonical_ids, colors))


def comparison_artifact_id(
    compatibility_key: tuple,
    members: tuple[TrajectoryComparisonMember, ...],
    final_interval_segments: int,
    focus_final_interval: bool,
    geometry_version: int,
    projection_version: int,
    render_version: int,
    comparison_view: str = "geometry",
) -> str:
    payload = {
        "compatibility": compatibility_key,
        "members": [
            {
                "group_id": member.group_id,
                "files": [path.name for path in member.input_paths],
            }
            for member in members
        ],
        "final_interval_segments": final_interval_segments,
        "focus_final_interval": focus_final_interval,
        "geometry_version": geometry_version,
        "projection_version": projection_version,
        "render_version": render_version,
    }
    if comparison_view != "geometry":
        payload["comparison_view"] = comparison_view
    serialized = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    )
    return sha256(serialized.encode("utf-8")).hexdigest()[:24]
