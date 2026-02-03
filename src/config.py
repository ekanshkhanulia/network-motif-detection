from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict


@dataclass(frozen=True)
class DatasetConfig:
    filename: str
    directed: bool
    description: str
    max_degree: int | None = None  # Optional degree filter for high-degree nodes


DATASETS: Dict[str, DatasetConfig] = {
    "Amazon0302": DatasetConfig(
        filename="Amazon0302.txt",
        directed=True,
        description="Amazon product co-purchasing network (directed)",
        # max_degree=None: No filtering (per original Wernicke 2005 paper)
        # Set to e.g. 50 if performance is too slow for k>=4
    ),
    "CA-AstroPh": DatasetConfig(
        filename="CA-AstroPh.txt",
        directed=False,
        description="ArXiv Astro Physics co-authorship network (undirected)",
        # max_degree=None: No filtering (per original Wernicke 2005 paper)
    ),
    "Email-Enron": DatasetConfig(
        filename="Email-Enron.txt",
        directed=False,  # Stored as directed but semantically undirected (email exchanges)
        description="Enron email communication network",
    ),
    "Wiki-Vote": DatasetConfig(
        filename="Wiki-Vote.txt",
        directed=True,
        description="Wikipedia adminship vote network (directed)",
        # max_degree=None: No filtering (per original Wernicke 2005 paper)
    ),
    "roadNet-CA": DatasetConfig(
        filename="roadNet-CA.txt",
        directed=False,
        description="California road network (undirected)",
        # Note: Large network with ~2M nodes, may require more time/samples
    ),
}


def resolve_data_path(data_dir: Path, dataset_key: str) -> Path:
    if dataset_key not in DATASETS:
        raise KeyError(
            f"Unknown dataset '{dataset_key}'. Available: {sorted(DATASETS.keys())}"
        )
    return (Path(data_dir) / DATASETS[dataset_key].filename).resolve()
