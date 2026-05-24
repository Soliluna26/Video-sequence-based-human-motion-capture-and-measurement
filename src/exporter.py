"""Data export module.

Exports kinematic measurements to CSV, JSON, and optionally MAT formats.
"""

import csv
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np


def export_csv(
    time_sec: np.ndarray,
    data_dict: Dict[str, np.ndarray],
    output_path: str,
    include_header: bool = True,
):
    """Export time-series data to CSV.

    Parameters
    ----------
    time_sec : np.ndarray
        Time column, shape (T,).
    data_dict : dict
        {column_name: array shape (T,)}.
    output_path : str
    include_header : bool
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    columns = ["time_sec"] + list(data_dict.keys())
    T = len(time_sec)
    rows = []
    for i in range(T):
        row = [time_sec[i]]
        for key in data_dict:
            val = data_dict[key][i]
            row.append(val)
        rows.append(row)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        if include_header:
            writer.writerow(columns)
        writer.writerows(rows)


def export_json(
    metrics: Dict,
    output_path: str,
    indent: int = 2,
):
    """Export kinematic metrics and metadata as JSON.

    Parameters
    ----------
    metrics : dict
        Any JSON-serializable dictionary of results.
    output_path : str
    indent : int
        JSON indentation level.
    """

    def default_serializer(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, (np.integer,)):
            return int(obj)
        raise TypeError(f"Unserializable type: {type(obj)}")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=indent, default=default_serializer, ensure_ascii=False)


def export_mat(
    data_dict: Dict[str, np.ndarray],
    output_path: str,
):
    """Export data to MATLAB .mat format (requires scipy.io).

    Parameters
    ----------
    data_dict : dict
        {variable_name: array}.
    output_path : str
    """
    from scipy.io import savemat

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # scipy.io.savemat doesn't support pathlib directly on all versions
    savemat(str(output_path), data_dict, do_compression=True)
