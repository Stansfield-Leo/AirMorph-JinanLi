import os
import sys
import math
import heapq
import logging
import warnings
import traceback
from pathlib import Path
from collections import defaultdict, deque

import numpy as np
import nibabel as nib
from scipy.ndimage import label, convolve


PROJECT_ROOT = Path("/home/jinanli24/AirMorph")
SAMPLE_ROOT = PROJECT_ROOT / "sample_data" / "ATM22"
AXIS_NAMES = ["ijk", "ikj", "jik", "jki", "kij", "kji"]
SMALL_THRESHOLDS = [2, 5]


# =============================================================================
# Basic helpers
# =============================================================================

def normalize_case_id(case_id: str) -> str:
    case_id = str(case_id).strip()
    if case_id.isdigit():
        return f"{int(case_id):03d}"
    return case_id


def get_default_case_ids():
    """Use existing numeric case folders if possible. Fallback to 001-059."""
    if SAMPLE_ROOT.exists():
        case_ids = [
            normalize_case_id(p.name)
            for p in sorted(SAMPLE_ROOT.iterdir())
            if p.is_dir() and p.name.isdigit()
        ]
        if case_ids:
            return case_ids
    return [f"{i:03d}" for i in range(1, 60)]


def expand_case_token(token: str):
    token = token.strip()
    if not token:
        return []
    if "-" not in token:
        return [normalize_case_id(token)]
    start, end = token.split("-", 1)
    start_i = int(start)
    end_i = int(end)
    if start_i > end_i:
        raise ValueError(f"Invalid case range: {token}")
    return [f"{i:03d}" for i in range(start_i, end_i + 1)]


def parse_multiple_cases(raw_text: str):
    raw_text = raw_text.replace(",", " ")
    case_ids = []
    for token in raw_text.split():
        case_ids.extend(expand_case_token(token))

    seen = set()
    unique_case_ids = []
    for case_id in case_ids:
        if case_id not in seen:
            seen.add(case_id)
            unique_case_ids.append(case_id)
    return unique_case_ids


def get_case_dir(case_id: str) -> Path:
    return SAMPLE_ROOT / normalize_case_id(case_id)


def get_skeleton_file(case_dir: Path) -> Path:
    skeleton_file = case_dir / "airway_skeleton.nii.gz"
    if not skeleton_file.exists():
        fallback = case_dir / "binary_only" / "airway_skeleton.nii.gz"
        if fallback.exists():
            skeleton_file = fallback
    return skeleton_file


def get_vtk_file(case_dir: Path, case_id: str, axis: str) -> Path:
    return case_dir / f"{case_id}_airway_lines_{axis}.vtk"


def euclidean_distance(p1, p2) -> float:
    return math.sqrt(
        (p1[0] - p2[0]) ** 2
        + (p1[1] - p2[1]) ** 2
        + (p1[2] - p2[2]) ** 2
    )


# =============================================================================
# 1. AirMorph pipeline
# =============================================================================

def is_case_finished(case_dir: Path, case_id: str) -> bool:
    required_outputs = [
        case_dir / "airway_bin.nii.gz",
        case_dir / "airway_skeleton.nii.gz",
        case_dir / "lunglobe.nii.gz",
        case_dir / f"{case_id}_airway_feature_cls.npy",
        case_dir / f"{case_id}_airway_graph.npy",
        case_dir / f"{case_id}_airway_graph_cls.npy",
        case_dir / f"{case_id}_anno.json",
        case_dir / f"{case_id}_parse.nii.gz",
        case_dir / f"{case_id}_pred_lob.nii.gz",
        case_dir / f"{case_id}_pred_seg.nii.gz",
        case_dir / f"{case_id}_pred_sub.nii.gz",
        case_dir / f"{case_id}_skel_parsing.nii.gz",
    ]

    for axis in AXIS_NAMES:
        required_outputs.append(case_dir / f"{case_id}_airway_lines_{axis}.vtk")

    missing = [p for p in required_outputs if not p.exists()]
    if not missing:
        return True

    print(f"[INCOMPLETE] {case_id}: missing {len(missing)} file(s)")
    for p in missing[:10]:
        print(f"  Missing: {p}")
    if len(missing) > 10:
        print(f"  ... and {len(missing) - 10} more")
    return False


def run_airmorph_pipeline(case_ids):
    """Run AirMorph segmentation/classification pipeline."""
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    logging.basicConfig(
        level=logging.INFO,
        format="(%(asctime)s)(%(levelname)s) %(name)s: %(message)s",
    )

    # Import here so analysis-only use does not need MONAI pipeline initialization.
    from monai.transforms import Compose
    from segmentator.airway_segmentator import AirwayAtlasBinaryAirwaySegmentator
    from classifier.airway_classifier import AirwayAtlasMultiAnatomyAirwayClassifier

    sys.path.append(str(PROJECT_ROOT))

    pipelines = Compose([
        AirwayAtlasBinaryAirwaySegmentator(),
        AirwayAtlasMultiAnatomyAirwayClassifier(),
    ])

    for raw_case_id in case_ids:
        case_id = normalize_case_id(raw_case_id)
        case_dir = get_case_dir(case_id)
        image_file = case_dir / "image.nii.gz"

        try:
            if not case_dir.is_dir():
                print(f"[SKIP] {case_id}: case folder not found: {case_dir}")
                continue
            if not image_file.exists():
                print(f"[SKIP] {case_id}: image.nii.gz not found: {image_file}")
                continue
            if is_case_finished(case_dir, case_id):
                print(f"[SKIP] {case_id}: already complete")
                continue

            output_vtk = case_dir / f"{case_id}_airway_lines.vtk"

            print(f"[START] {case_id}")
            print(f"  Case dir        : {case_dir}")
            print(f"  Image           : {image_file}")
            print(f"  Base output VTK : {output_vtk}")

            os.environ["AIRMORPH_OUTPUT_VTK"] = str(output_vtk)
            os.environ["AIRMORPH_REFERENCE_IMAGE"] = str(image_file)

            data = {
                "patient": case_id,
                "file_path": str(case_dir),
                "image_file": str(image_file),
                "output_vtk": str(output_vtk),
                "reference_image": str(image_file),
            }

            pipelines(data)
            print(f"[DONE] {case_id}")

        except Exception:
            print(f"[ERROR] {case_id}")
            traceback.print_exc()

    print("[PIPELINE ALL DONE]")


# =============================================================================
# Shared VTK and graph helpers
# =============================================================================

def read_vtk_lines(vtk_file: Path):
    """Read legacy ASCII VTK PolyData line model."""
    with open(vtk_file, "r") as f:
        raw = [line.strip() for line in f.readlines()]

    num_points = None
    num_lines = None
    point_start = None
    line_start = None

    for idx, line in enumerate(raw):
        if line.startswith("POINTS"):
            parts = line.split()
            num_points = int(parts[1])
            point_start = idx + 1
        if line.startswith("LINES"):
            parts = line.split()
            num_lines = int(parts[1])
            line_start = idx + 1
            break

    if num_points is None or point_start is None:
        raise ValueError("Cannot find POINTS section in VTK file.")
    if num_lines is None or line_start is None:
        raise ValueError("Cannot find LINES section in VTK file.")

    points = []
    for i in range(point_start, point_start + num_points):
        parts = raw[i].split()
        if len(parts) < 3:
            raise ValueError(f"Invalid point line: {raw[i]}")
        points.append((float(parts[0]), float(parts[1]), float(parts[2])))

    edges = []
    for i in range(line_start, line_start + num_lines):
        parts = raw[i].split()
        if len(parts) < 3:
            continue
        n = int(parts[0])
        if n != 2:
            continue
        edges.append((int(parts[1]), int(parts[2])))

    return points, edges


def build_unweighted_graph(num_points: int, edges):
    graph = defaultdict(set)
    for i in range(num_points):
        graph[i] = set()
    for p1, p2 in edges:
        graph[p1].add(p2)
        graph[p2].add(p1)
    return graph


def build_weighted_graph(points, edges):
    graph = defaultdict(list)
    for i in range(len(points)):
        graph[i] = []
    for p1, p2 in edges:
        w = euclidean_distance(points[p1], points[p2])
        graph[p1].append((p2, w))
        graph[p2].append((p1, w))
    return graph


def find_connected_components_unweighted(num_points: int, graph):
    visited = set()
    components = []
    for p in range(num_points):
        if p in visited:
            continue
        queue = deque([p])
        visited.add(p)
        comp = []
        while queue:
            cur = queue.popleft()
            comp.append(cur)
            for nb in graph[cur]:
                if nb not in visited:
                    visited.add(nb)
                    queue.append(nb)
        components.append(comp)
    return components


def compute_component_metrics(component_sizes, total_units):
    component_sizes = sorted(component_sizes, reverse=True)
    if total_units <= 0 or not component_sizes:
        return {
            "total_units": total_units,
            "connected_components": 0,
            "main_component_size": 0,
            "main_component_ratio": 0.0,
            "disconnected_components": 0,
            "small_components": {},
            "top_component_sizes": [],
        }

    main_size = component_sizes[0]
    small_components = {}
    for threshold in SMALL_THRESHOLDS:
        small = [s for s in component_sizes if s < threshold]
        small_components[threshold] = {
            "count": len(small),
            "units": sum(small),
            "burden": sum(small) / total_units if total_units > 0 else 0.0,
        }

    return {
        "total_units": total_units,
        "connected_components": len(component_sizes),
        "main_component_size": main_size,
        "main_component_ratio": main_size / total_units,
        "disconnected_components": max(0, len(component_sizes) - 1),
        "small_components": small_components,
        "top_component_sizes": component_sizes[:20],
    }


def analyze_skeleton_components(skeleton_file: Path):
    result = {
        "name": "airway_skeleton.nii.gz",
        "file": skeleton_file,
        "exists": skeleton_file.exists(),
        "unit": "voxels",
        "type": "skeleton_nifti",
    }
    if not skeleton_file.exists():
        result["error"] = "File not found"
        return result

    img = nib.load(str(skeleton_file))
    data = img.get_fdata()
    binary = (data > 0).astype(np.uint8)
    total_voxels = int(binary.sum())

    if total_voxels == 0:
        result["error"] = "Skeleton is empty"
        result.update(compute_component_metrics([], 0))
        return result

    structure = np.ones((3, 3, 3), dtype=np.uint8)
    labeled, num_components = label(binary, structure=structure)
    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    component_sizes = [int(counts[i]) for i in range(1, num_components + 1) if counts[i] > 0]

    result["error"] = None
    result.update(compute_component_metrics(component_sizes, total_voxels))
    return result


def analyze_vtk_components(vtk_file: Path, axis_name: str = None):
    result = {
        "name": vtk_file.name,
        "axis": axis_name,
        "file": vtk_file,
        "exists": vtk_file.exists(),
        "unit": "points",
        "type": "vtk_graph",
    }
    if not vtk_file.exists():
        result["error"] = "File not found"
        return result

    points, edges = read_vtk_lines(vtk_file)
    num_points = len(points)
    if num_points == 0:
        result["error"] = "VTK has no points"
        result["lines"] = len(edges)
        result["isolated_points"] = 0
        result.update(compute_component_metrics([], 0))
        return result

    graph = build_unweighted_graph(num_points, edges)
    components = find_connected_components_unweighted(num_points, graph)
    component_sizes = [len(c) for c in components]
    isolated_points = sum(1 for p in range(num_points) if len(graph[p]) == 0)

    result["error"] = None
    result["lines"] = len(edges)
    result["isolated_points"] = isolated_points
    result.update(compute_component_metrics(component_sizes, num_points))
    return result


# =============================================================================
# 2. Connected components report
# =============================================================================

def write_component_report_section(f, title: str, result):
    f.write("=" * 80 + "\n")
    f.write(title + "\n")
    f.write("=" * 80 + "\n")
    f.write(f"File: {result.get('file')}\n")
    f.write(f"Exists: {result.get('exists')}\n")
    if result.get("axis") is not None:
        f.write(f"Axis order: {result.get('axis')}\n")
    if result.get("error"):
        f.write("Status: ERROR / MISSING\n")
        f.write(f"Error: {result.get('error')}\n\n")
        return

    f.write("Status: success\n")
    f.write(f"Unit: {result.get('unit')}\n")
    if result.get("type") == "skeleton_nifti":
        f.write(f"Voxels: {result.get('total_units')}\n")
    else:
        f.write(f"Points: {result.get('total_units')}\n")
    if "lines" in result:
        f.write(f"Lines: {result.get('lines')}\n")
    f.write(f"Connected components: {result.get('connected_components')}\n")
    f.write(f"Main component size: {result.get('main_component_size')}\n")
    f.write(f"Main component ratio: {result.get('main_component_ratio'):.4f}\n")
    f.write(f"Disconnected components: {result.get('disconnected_components')}\n")
    if "isolated_points" in result:
        f.write(f"Isolated points: {result.get('isolated_points')}\n")

    for threshold in SMALL_THRESHOLDS:
        small = result.get("small_components", {}).get(threshold)
        if small is None:
            continue
        f.write(f"Small components (<{threshold} {result.get('unit')}): {small['count']}\n")
        f.write(f"Small component units (<{threshold} {result.get('unit')}): {small['units']}\n")
        f.write(f"Small component burden (<{threshold} {result.get('unit')}): {small['burden']:.8f}\n")

    f.write("\nTop component sizes\n")
    f.write("-------------------\n")
    for idx, size in enumerate(result.get("top_component_sizes", []), start=1):
        f.write(f"{idx}: {size} {result.get('unit')}\n")
    f.write("\n")


def run_component_analysis(case_ids):
    for raw_case_id in case_ids:
        case_id = normalize_case_id(raw_case_id)
        case_dir = get_case_dir(case_id)
        if not case_dir.exists():
            print(f"[SKIP] {case_id}: case folder not found: {case_dir}")
            continue

        skeleton_file = get_skeleton_file(case_dir)
        report_file = case_dir / f"{case_id}_component_analysis_report.txt"

        print(f"[START] {case_id}")
        print(f"  Case dir : {case_dir}")
        print(f"  Report   : {report_file}")

        skeleton_result = analyze_skeleton_components(skeleton_file)
        vtk_results = [
            analyze_vtk_components(get_vtk_file(case_dir, case_id, axis), axis)
            for axis in AXIS_NAMES
        ]

        with open(report_file, "w", encoding="utf-8") as f:
            f.write(f"Case ID: {case_id}\n")
            f.write("Skeleton and Axis VTK Component Analysis Report\n\n")
            f.write("Definition\n----------\n")
            f.write("Connected components are calculated from graph connectivity.\n")
            f.write("For airway_skeleton.nii.gz, 3D 26-connectivity is used.\n")
            f.write("For VTK line models, POINTS are graph nodes and LINES are graph edges.\n")
            f.write("Disconnected components are all components outside the largest component.\n")
            f.write("The largest component is treated as the main airway tree.\n\n")
            f.write("Status meaning\n--------------\n")
            f.write("0 = main component\n1 = disconnected component\n2 = isolated point\n\n")

            write_component_report_section(f, "Skeleton NIfTI: airway_skeleton.nii.gz", skeleton_result)
            for vtk_result in vtk_results:
                write_component_report_section(f, f"VTK Graph: {vtk_result.get('name')}", vtk_result)

        print(f"[DONE] {case_id}")


# =============================================================================
# 3. Short fragment burden report
# =============================================================================

def write_short_section(f, title, metrics):
    f.write("=" * 80 + "\n")
    f.write(title + "\n")
    f.write("=" * 80 + "\n")

    if not metrics.get("exists", False):
        f.write("Status: missing\n")
        f.write(f"Error: {metrics.get('error')}\n\n")
        return
    if metrics.get("error"):
        f.write("Status: error\n")
        f.write(f"Error: {metrics.get('error')}\n\n")
        return

    f.write("Status: success\n")
    f.write(f"File: {metrics.get('file')}\n")
    f.write(f"Unit: {metrics.get('unit')}\n")
    f.write(f"Total units: {metrics.get('total_units')}\n")
    if "lines" in metrics:
        f.write(f"Lines: {metrics.get('lines')}\n")
    f.write(f"Connected components: {metrics.get('connected_components')}\n")
    f.write(f"Main component size: {metrics.get('main_component_size')}\n")
    f.write(f"Main component ratio: {metrics.get('main_component_ratio'):.6f}\n")
    f.write(f"Disconnected components: {metrics.get('disconnected_components')}\n")
    if "isolated_points" in metrics:
        f.write(f"Isolated points: {metrics.get('isolated_points')}\n")

    f.write("\nShort fragment burden\n")
    f.write("---------------------\n")
    for threshold in SMALL_THRESHOLDS:
        small = metrics.get("small_components", {}).get(threshold)
        if small is None:
            continue
        f.write(f"Threshold: <{threshold} {metrics.get('unit')}\n")
        f.write(f"Short fragment count: {small['count']}\n")
        f.write(f"Short fragment units: {small['units']}\n")
        f.write(f"Short fragment burden: {small['burden']:.8f}\n\n")

    f.write("Top component sizes\n")
    f.write("-------------------\n")
    for idx, size in enumerate(metrics.get("top_component_sizes", []), start=1):
        f.write(f"{idx}: {size} {metrics.get('unit')}\n")
    f.write("\n")


def run_short_fragment_analysis(case_ids):
    for raw_case_id in case_ids:
        case_id = normalize_case_id(raw_case_id)
        case_dir = get_case_dir(case_id)
        if not case_dir.exists():
            print(f"[SKIP] {case_id}: case folder not found: {case_dir}")
            continue

        skeleton_file = get_skeleton_file(case_dir)
        report_file = case_dir / f"{case_id}_short_fragment_report.txt"

        print(f"[START] {case_id}")
        print(f"  Report: {report_file}")

        skeleton_metrics = analyze_skeleton_components(skeleton_file)
        vtk_metrics_list = [
            analyze_vtk_components(get_vtk_file(case_dir, case_id, axis), axis)
            for axis in AXIS_NAMES
        ]

        with open(report_file, "w", encoding="utf-8") as f:
            f.write(f"Case ID: {case_id}\n")
            f.write("Short Fragment Burden Report\n\n")
            f.write("Definition\n----------\n")
            f.write("Short fragment burden is defined as:\n")
            f.write("short fragment units / total units\n\n")
            f.write("For the NIfTI skeleton, the unit is voxel.\n")
            f.write("For the VTK line model, the unit is point.\n\n")
            f.write("Thresholds used in this report:\n")
            for threshold in SMALL_THRESHOLDS:
                f.write(f"- component size < {threshold}\n")
            f.write("\n")

            write_short_section(f, "Skeleton NIfTI analysis", skeleton_metrics)
            for vtk_metrics in vtk_metrics_list:
                write_short_section(
                    f,
                    f"VTK Graph analysis: {vtk_metrics.get('name')} | axis={vtk_metrics.get('axis')}",
                    vtk_metrics,
                )

        print(f"[DONE] {case_id}")


# =============================================================================
# 4. Terminated branches report
# =============================================================================

def add_skeleton_degree_metrics(metrics, skeleton_file: Path):
    if metrics.get("error"):
        return metrics

    img = nib.load(str(skeleton_file))
    binary = (img.get_fdata() > 0).astype(np.uint8)

    kernel = np.ones((3, 3, 3), dtype=np.uint8)
    kernel[1, 1, 1] = 0
    neighbour_count = convolve(binary, kernel, mode="constant", cval=0)
    degree_values = neighbour_count[binary > 0]

    endpoint_points = int(np.sum(degree_values == 1))
    metrics["isolated_points"] = int(np.sum(degree_values == 0))
    metrics["endpoint_points"] = endpoint_points
    metrics["normal_line_points"] = int(np.sum(degree_values == 2))
    metrics["branch_points"] = int(np.sum(degree_values >= 3))
    metrics["estimated_terminated_branches"] = max(0, endpoint_points - 1)
    return metrics


def add_vtk_degree_metrics(metrics, vtk_file: Path):
    if metrics.get("error"):
        return metrics

    points, edges = read_vtk_lines(vtk_file)
    graph = build_unweighted_graph(len(points), edges)
    degree_values = [len(graph[p]) for p in range(len(points))]

    endpoint_points = sum(1 for d in degree_values if d == 1)
    metrics["isolated_points"] = sum(1 for d in degree_values if d == 0)
    metrics["endpoint_points"] = endpoint_points
    metrics["normal_line_points"] = sum(1 for d in degree_values if d == 2)
    metrics["branch_points"] = sum(1 for d in degree_values if d >= 3)
    metrics["estimated_terminated_branches"] = max(0, endpoint_points - 1)
    return metrics


def write_terminated_section(f, title, metrics):
    f.write("=" * 80 + "\n")
    f.write(title + "\n")
    f.write("=" * 80 + "\n")

    if not metrics.get("exists", False):
        f.write("Status: missing\n")
        f.write(f"Error: {metrics.get('error')}\n\n")
        return
    if metrics.get("error"):
        f.write("Status: error\n")
        f.write(f"Error: {metrics.get('error')}\n\n")
        return

    f.write("Status: success\n")
    f.write(f"File: {metrics.get('file')}\n")
    f.write(f"Unit: {metrics.get('unit')}\n")
    f.write(f"Total units: {metrics.get('total_units')}\n")
    if "lines" in metrics:
        f.write(f"Lines: {metrics.get('lines')}\n")
    f.write(f"Connected components: {metrics.get('connected_components')}\n")
    f.write(f"Main component size: {metrics.get('main_component_size')}\n")
    f.write(f"Main component ratio: {metrics.get('main_component_ratio'):.6f}\n")
    f.write(f"Disconnected components: {metrics.get('disconnected_components')}\n")
    f.write(f"Isolated points, degree=0: {metrics.get('isolated_points')}\n")
    f.write(f"Endpoint points, degree=1: {metrics.get('endpoint_points')}\n")
    f.write(f"Estimated terminated branches: {metrics.get('estimated_terminated_branches')}\n")
    f.write(f"Normal line points, degree=2: {metrics.get('normal_line_points')}\n")
    f.write(f"Branch points, degree>=3: {metrics.get('branch_points')}\n")

    f.write("\nShort fragment burden\n")
    f.write("---------------------\n")
    for threshold in SMALL_THRESHOLDS:
        small = metrics.get("small_components", {}).get(threshold)
        if small is None:
            continue
        f.write(f"Threshold: <{threshold} {metrics.get('unit')}\n")
        f.write(f"Short fragment count: {small['count']}\n")
        f.write(f"Short fragment units: {small['units']}\n")
        f.write(f"Short fragment burden: {small['burden']:.8f}\n\n")

    f.write("Top component sizes\n")
    f.write("-------------------\n")
    for idx, size in enumerate(metrics.get("top_component_sizes", []), start=1):
        f.write(f"{idx}: {size} {metrics.get('unit')}\n")
    f.write("\n")


def run_terminated_branch_analysis(case_ids):
    for raw_case_id in case_ids:
        case_id = normalize_case_id(raw_case_id)
        case_dir = get_case_dir(case_id)
        if not case_dir.exists():
            print(f"[SKIP] {case_id}: case folder not found: {case_dir}")
            continue

        skeleton_file = get_skeleton_file(case_dir)
        report_file = case_dir / f"{case_id}_terminated_branch_report.txt"

        print(f"[START] {case_id}")
        print(f"  Report: {report_file}")

        skeleton_metrics = add_skeleton_degree_metrics(
            analyze_skeleton_components(skeleton_file),
            skeleton_file,
        )

        vtk_metrics_list = []
        for axis in AXIS_NAMES:
            vtk_file = get_vtk_file(case_dir, case_id, axis)
            metrics = analyze_vtk_components(vtk_file, axis)
            metrics = add_vtk_degree_metrics(metrics, vtk_file)
            vtk_metrics_list.append(metrics)

        with open(report_file, "w", encoding="utf-8") as f:
            f.write(f"Case ID: {case_id}\n")
            f.write("Terminated Branch Analysis Report\n\n")
            f.write("Definition\n----------\n")
            f.write("Connected components are calculated from graph connectivity.\n")
            f.write("For airway_skeleton.nii.gz, 3D 26-connectivity is used.\n")
            f.write("For VTK line models, POINTS are graph nodes and LINES are graph edges.\n\n")
            f.write("Endpoint points are defined as points with degree=1.\n")
            f.write("Estimated terminated branches = endpoint points - 1.\n")
            f.write("The subtraction assumes one endpoint is the proximal tracheal opening.\n\n")
            f.write("Short fragment burden is defined as:\nshort fragment units / total units\n\n")
            f.write("Thresholds used for short fragments:\n")
            for threshold in SMALL_THRESHOLDS:
                f.write(f"- component size < {threshold}\n")
            f.write("\n")

            write_terminated_section(f, "Skeleton NIfTI analysis", skeleton_metrics)
            for metrics in vtk_metrics_list:
                write_terminated_section(
                    f,
                    f"VTK Graph analysis: {metrics.get('name')} | axis={metrics.get('axis')}",
                    metrics,
                )

        print(f"[DONE] {case_id}")


# =============================================================================
# 5. Total skeleton length report
# =============================================================================

def analyze_skeleton_length(skeleton_file: Path):
    if not skeleton_file.exists():
        return {"exists": False, "error": f"Skeleton file not found: {skeleton_file}"}

    img = nib.load(str(skeleton_file))
    data = img.get_fdata()
    affine = img.affine
    binary = data > 0
    coords = np.array(np.nonzero(binary)).T
    total_voxels = int(coords.shape[0])

    if total_voxels == 0:
        return {"exists": True, "error": "Skeleton file is empty."}

    voxel_set = set(tuple(map(int, c)) for c in coords)
    offsets = []
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            for dk in [-1, 0, 1]:
                if di == 0 and dj == 0 and dk == 0:
                    continue
                if (di, dj, dk) > (0, 0, 0):
                    offsets.append((di, dj, dk))

    total_length = 0.0
    edge_count = 0
    for i, j, k in voxel_set:
        p1 = (affine @ np.array([i, j, k, 1.0]))[:3]
        for di, dj, dk in offsets:
            nb = (i + di, j + dj, k + dk)
            if nb not in voxel_set:
                continue
            p2 = (affine @ np.array([nb[0], nb[1], nb[2], 1.0]))[:3]
            total_length += euclidean_distance(p1, p2)
            edge_count += 1

    return {
        "exists": True,
        "error": None,
        "file": str(skeleton_file),
        "unit": "voxels",
        "total_voxels": total_voxels,
        "edge_count": edge_count,
        "total_skeleton_length_mm": total_length,
        "average_edge_length_mm": total_length / edge_count if edge_count else 0.0,
    }


def analyze_vtk_length(vtk_file: Path):
    if not vtk_file.exists():
        return {"exists": False, "error": f"VTK file not found: {vtk_file}"}

    points, edges = read_vtk_lines(vtk_file)
    if not points:
        return {"exists": True, "error": "VTK file contains no points."}

    total_length = sum(euclidean_distance(points[p1], points[p2]) for p1, p2 in edges)
    return {
        "exists": True,
        "error": None,
        "file": str(vtk_file),
        "unit": "points",
        "points": len(points),
        "lines": len(edges),
        "total_skeleton_length_mm": total_length,
        "average_line_length_mm": total_length / len(edges) if edges else 0.0,
    }


def write_length_section(f, title, result):
    f.write("=" * 80 + "\n")
    f.write(title + "\n")
    f.write("=" * 80 + "\n")

    if not result.get("exists", False):
        f.write("Status: missing\n")
        f.write(f"Error: {result.get('error')}\n\n")
        return
    if result.get("error"):
        f.write("Status: error\n")
        f.write(f"Error: {result.get('error')}\n\n")
        return

    f.write("Status: success\n")
    f.write(f"File: {result.get('file')}\n")
    f.write(f"Unit: {result.get('unit')}\n")
    if "total_voxels" in result:
        f.write(f"Skeleton voxels: {result.get('total_voxels')}\n")
    if "points" in result:
        f.write(f"Points: {result.get('points')}\n")
    if "edge_count" in result:
        f.write(f"Skeleton edges: {result.get('edge_count')}\n")
    if "lines" in result:
        f.write(f"Lines: {result.get('lines')}\n")
    f.write(f"Total skeleton length: {result.get('total_skeleton_length_mm'):.4f} mm\n")
    if "average_edge_length_mm" in result:
        f.write(f"Average edge length: {result.get('average_edge_length_mm'):.6f} mm\n")
    if "average_line_length_mm" in result:
        f.write(f"Average line length: {result.get('average_line_length_mm'):.6f} mm\n")
    f.write("\n")


def run_total_length_analysis(case_ids):
    for raw_case_id in case_ids:
        case_id = normalize_case_id(raw_case_id)
        case_dir = get_case_dir(case_id)
        if not case_dir.exists():
            print(f"[SKIP] {case_id}: case folder not found: {case_dir}")
            continue

        skeleton_file = get_skeleton_file(case_dir)
        report_file = case_dir / f"{case_id}_skeleton_length_report.txt"

        print(f"[START] {case_id}")
        print(f"  Report: {report_file}")

        skeleton_result = analyze_skeleton_length(skeleton_file)
        vtk_results = []
        for axis in AXIS_NAMES:
            vtk_file = get_vtk_file(case_dir, case_id, axis)
            result = analyze_vtk_length(vtk_file)
            result["axis"] = axis
            result["name"] = vtk_file.name
            vtk_results.append(result)

        with open(report_file, "w", encoding="utf-8") as f:
            f.write(f"Case ID: {case_id}\n")
            f.write("Total Skeleton Length Report\n\n")
            f.write("Definition\n----------\n")
            f.write("Total skeleton length is calculated as the sum of Euclidean distances along skeleton edges.\n")
            f.write("For airway_skeleton.nii.gz, neighbouring skeleton voxels are connected using 3D 26-neighbour rules.\n")
            f.write("For VTK line models, POINTS are treated as spatial coordinates and LINES are treated as skeleton edges.\n")
            f.write("The unit is millimetres if the image affine and VTK coordinates are in physical space.\n\n")

            write_length_section(f, "Skeleton NIfTI length analysis", skeleton_result)
            for result in vtk_results:
                write_length_section(f, f"VTK length analysis: {result.get('name')} | axis={result.get('axis')}", result)

        print(f"[DONE] {case_id}")


# =============================================================================
# 6. Peripheral reach / maximum branch depth report
# =============================================================================

def dijkstra(graph, start):
    dist = {node: float("inf") for node in graph}
    depth = {node: 0 for node in graph}
    dist[start] = 0.0
    heap = [(0.0, 0, start)]

    while heap:
        current_dist, current_depth, node = heapq.heappop(heap)
        if current_dist > dist[node]:
            continue
        for nb, weight in graph[node]:
            new_dist = current_dist + weight
            new_depth = current_depth + 1
            if new_dist < dist[nb]:
                dist[nb] = new_dist
                depth[nb] = new_depth
                heapq.heappush(heap, (new_dist, new_depth, nb))
    return dist, depth


def get_connected_components_weighted(graph):
    visited = set()
    components = []
    for node in graph:
        if node in visited:
            continue
        stack = [node]
        visited.add(node)
        comp = []
        while stack:
            cur = stack.pop()
            comp.append(cur)
            for nb, _ in graph[cur]:
                if nb not in visited:
                    visited.add(nb)
                    stack.append(nb)
        components.append(comp)
    return sorted(components, key=len, reverse=True)


def compute_reach_and_depth(graph):
    if not graph:
        return {"error": "Graph is empty."}

    components = get_connected_components_weighted(graph)
    if not components:
        return {"error": "No connected component found."}

    main_component = set(components[0])
    subgraph = {
        node: [(nb, w) for nb, w in graph[node] if nb in main_component]
        for node in main_component
    }

    degree = {node: len(subgraph[node]) for node in subgraph}
    endpoints = [node for node, d in degree.items() if d == 1]
    branch_points = [node for node, d in degree.items() if d >= 3]
    isolated_points = [node for node, d in degree.items() if d == 0]

    if not endpoints:
        start = next(iter(main_component))
        candidate_nodes = list(main_component)
    else:
        start = endpoints[0]
        candidate_nodes = endpoints

    dist_1, _ = dijkstra(subgraph, start)
    farthest_a = max(candidate_nodes, key=lambda node: dist_1.get(node, float("-inf")))

    dist_2, depth_2 = dijkstra(subgraph, farthest_a)
    farthest_b = max(candidate_nodes, key=lambda node: dist_2.get(node, float("-inf")))

    total_length_mm = 0.0
    edge_count = 0
    seen_edges = set()
    for node in subgraph:
        for nb, weight in subgraph[node]:
            edge_key = tuple(sorted((node, nb)))
            if edge_key in seen_edges:
                continue
            seen_edges.add(edge_key)
            total_length_mm += weight
            edge_count += 1

    return {
        "error": None,
        "total_nodes": len(graph),
        "main_component_nodes": len(main_component),
        "connected_components": len(components),
        "endpoints_degree_1": len(endpoints),
        "branch_points_degree_ge_3": len(branch_points),
        "isolated_points_degree_0": len(isolated_points),
        "total_edges_main_component": edge_count,
        "total_length_mm": total_length_mm,
        "peripheral_reach_mm": dist_2[farthest_b],
        "maximum_branch_depth_edges": depth_2[farthest_b],
        "reach_start_node": farthest_a,
        "reach_end_node": farthest_b,
    }


def build_skeleton_weighted_graph(skeleton_file: Path):
    if not skeleton_file.exists():
        return None, {"exists": False, "error": f"Skeleton file not found: {skeleton_file}"}

    img = nib.load(str(skeleton_file))
    data = img.get_fdata()
    affine = img.affine
    binary = data > 0
    coords = np.array(np.nonzero(binary)).T

    if coords.shape[0] == 0:
        return None, {"exists": True, "error": "Skeleton file is empty."}

    voxel_to_id = {}
    points = []
    for node_id, (i, j, k) in enumerate(coords):
        i, j, k = int(i), int(j), int(k)
        voxel_to_id[(i, j, k)] = node_id
        world = affine @ np.array([i, j, k, 1.0])
        points.append(tuple(world[:3]))

    graph = defaultdict(list)
    for node_id in range(len(points)):
        graph[node_id] = []

    offsets = []
    for di in [-1, 0, 1]:
        for dj in [-1, 0, 1]:
            for dk in [-1, 0, 1]:
                if di == 0 and dj == 0 and dk == 0:
                    continue
                if (di, dj, dk) > (0, 0, 0):
                    offsets.append((di, dj, dk))

    for (i, j, k), node_id in voxel_to_id.items():
        for di, dj, dk in offsets:
            nb_voxel = (i + di, j + dj, k + dk)
            if nb_voxel not in voxel_to_id:
                continue
            nb_id = voxel_to_id[nb_voxel]
            length = euclidean_distance(points[node_id], points[nb_id])
            graph[node_id].append((nb_id, length))
            graph[nb_id].append((node_id, length))

    info = {
        "exists": True,
        "error": None,
        "file": str(skeleton_file),
        "source": "Skeleton NIfTI",
        "unit": "voxels",
    }
    return graph, info


def build_vtk_weighted_graph(vtk_file: Path):
    if not vtk_file.exists():
        return None, {"exists": False, "error": f"VTK file not found: {vtk_file}"}

    points, edges = read_vtk_lines(vtk_file)
    if not points:
        return None, {"exists": True, "error": "VTK file contains no points."}

    graph = build_weighted_graph(points, edges)
    info = {
        "exists": True,
        "error": None,
        "file": str(vtk_file),
        "source": "VTK Graph",
        "unit": "points",
        "points": len(points),
        "lines": len(edges),
    }
    return graph, info


def write_reach_section(f, title, info, metrics):
    f.write("=" * 80 + "\n")
    f.write(title + "\n")
    f.write("=" * 80 + "\n")

    if not info.get("exists", False):
        f.write("Status: missing\n")
        f.write(f"Error: {info.get('error')}\n\n")
        return
    if info.get("error"):
        f.write("Status: error\n")
        f.write(f"Error: {info.get('error')}\n\n")
        return
    if metrics.get("error"):
        f.write("Status: error\n")
        f.write(f"Error: {metrics.get('error')}\n\n")
        return

    f.write("Status: success\n")
    f.write(f"File: {info.get('file')}\n")
    f.write(f"Source: {info.get('source')}\n")
    f.write(f"Unit: {info.get('unit')}\n")
    if "points" in info:
        f.write(f"Points: {info.get('points')}\n")
    if "lines" in info:
        f.write(f"Lines: {info.get('lines')}\n")
    f.write(f"Connected components: {metrics.get('connected_components')}\n")
    f.write(f"Main component nodes: {metrics.get('main_component_nodes')}\n")
    f.write(f"Endpoint points, degree=1: {metrics.get('endpoints_degree_1')}\n")
    f.write(f"Branch points, degree>=3: {metrics.get('branch_points_degree_ge_3')}\n")
    f.write(f"Isolated points, degree=0: {metrics.get('isolated_points_degree_0')}\n")
    f.write(f"Total edges in main component: {metrics.get('total_edges_main_component')}\n")
    f.write(f"Total skeleton length: {metrics.get('total_length_mm'):.4f} mm\n")
    f.write(f"Peripheral reach: {metrics.get('peripheral_reach_mm'):.4f} mm\n")
    f.write(f"Maximum branch depth: {metrics.get('maximum_branch_depth_edges')} edges\n")
    f.write(f"Reach start node: {metrics.get('reach_start_node')}\n")
    f.write(f"Reach end node: {metrics.get('reach_end_node')}\n\n")


def run_peripheral_reach_depth_analysis(case_ids):
    for raw_case_id in case_ids:
        case_id = normalize_case_id(raw_case_id)
        case_dir = get_case_dir(case_id)
        if not case_dir.exists():
            print(f"[SKIP] {case_id}: case folder not found: {case_dir}")
            continue

        skeleton_file = get_skeleton_file(case_dir)
        report_file = case_dir / f"{case_id}_peripheral_reach_depth_report.txt"

        print(f"[START] {case_id}")
        print(f"  Report: {report_file}")

        sections = []
        skeleton_graph, skeleton_info = build_skeleton_weighted_graph(skeleton_file)
        skeleton_metrics = compute_reach_and_depth(skeleton_graph) if skeleton_graph is not None else {"error": skeleton_info.get("error")}
        sections.append(("Skeleton NIfTI peripheral reach / maximum branch depth", skeleton_info, skeleton_metrics))

        for axis in AXIS_NAMES:
            vtk_file = get_vtk_file(case_dir, case_id, axis)
            vtk_graph, vtk_info = build_vtk_weighted_graph(vtk_file)
            vtk_info["axis"] = axis
            vtk_info["name"] = vtk_file.name
            vtk_metrics = compute_reach_and_depth(vtk_graph) if vtk_graph is not None else {"error": vtk_info.get("error")}
            sections.append((f"VTK peripheral reach / maximum branch depth: {vtk_file.name} | axis={axis}", vtk_info, vtk_metrics))

        with open(report_file, "w", encoding="utf-8") as f:
            f.write(f"Case ID: {case_id}\n")
            f.write("Peripheral Reach and Maximum Branch Depth Report\n\n")
            f.write("Definition\n----------\n")
            f.write("Peripheral reach is calculated as the longest shortest-path distance between endpoint nodes in the main connected component.\n")
            f.write("Maximum branch depth is the number of graph edges along this longest endpoint-to-endpoint path.\n")
            f.write("For airway_skeleton.nii.gz, graph edges are built using 3D 26-neighbour skeleton voxels.\n")
            f.write("For VTK line models, POINTS are graph nodes and LINES are graph edges.\n")
            f.write("Distances are reported in millimetres if the coordinates are in physical space.\n\n")
            f.write("Note\n----\n")
            f.write("This is a graph-based estimate. It does not explicitly identify the anatomical tracheal root.\n")
            f.write("For final anatomical interpretation, use the axis-order VTK that is visually aligned in 3D Slicer.\n\n")

            for title, info, metrics in sections:
                write_reach_section(f, title, info, metrics)

        print(f"[DONE] {case_id}")


# =============================================================================
# Interactive menu system
# =============================================================================

MENU = {
    "1": {
        "name": "AirMorph main pipeline",
        "description": "Generate airway_bin, airway_skeleton, lunglobe, anatomy outputs, and six axis-order VTK files.",
        "runner": run_airmorph_pipeline,
    },
    "2": {
        "name": "Connected components analysis",
        "description": "Analyze whether the skeleton and six VTK files contain disconnected components, isolated points, or small components.",
        "runner": run_component_analysis,
    },
    "3": {
        "name": "Short fragment burden analysis",
        "description": "Analyze short fragment count and short fragment burden.",
        "runner": run_short_fragment_analysis,
    },
    "4": {
        "name": "Terminated branches analysis",
        "description": "Analyze endpoints, estimated terminated branches, and branch points.",
        "runner": run_terminated_branch_analysis,
    },
    "5": {
        "name": "Total skeleton length analysis",
        "description": "Analyze total skeleton length.",
        "runner": run_total_length_analysis,
    },
    "6": {
        "name": "Peripheral reach / maximum branch depth analysis",
        "description": "Analyze peripheral reach and maximum branch depth.",
        "runner": run_peripheral_reach_depth_analysis,
    },
}


ALL_ANALYSIS_KEYS = ["2", "3", "4", "5", "6"]
FULL_SYSTEM_KEYS = ["1", "2", "3", "4", "5", "6"]


def print_main_menu():
    print()
    print("=" * 80)
    print("Airway Integrated Interactive System")
    print("=" * 80)
    print("Please select a function:\n")
    for key, item in MENU.items():
        print(f"{key}. {item['name']}")
        print(f"   {item['description']}")
    print("7. Run all 5 analysis functions")
    print("   Run components, short fragments, terminated branches, length, and reach/depth in order.")
    print("8. Full system: main pipeline + all analyses")
    print("   Run the AirMorph main pipeline first, then generate all 5 analysis reports.")
    print("0. Exit\n")


def print_mode_menu():
    print()
    print("-" * 80)
    print("Please select processing mode:")
    print("-" * 80)
    print("1. Batch process all cases")
    print("2. Process one selected case")
    print("3. Process multiple selected cases")
    print("0. Back to previous menu\n")


def select_cases_by_mode():
    while True:
        print_mode_menu()
        mode = input("Enter mode number: ").strip()

        if mode == "0":
            return None

        if mode == "1":
            case_ids = get_default_case_ids()
            print(f"Batch processing will run on {len(case_ids)} case(s).")
            if case_ids:
                print(f"Case range example: {case_ids[0]} - {case_ids[-1]}")
            confirm = input("Confirm batch processing? Enter y to continue, or any other key to cancel: ").strip().lower()
            if confirm == "y":
                return case_ids
            print("[CANCEL] Batch processing cancelled.")
            continue

        if mode == "2":
            raw_case = input("Enter one case ID to process, for example 001 or 4: ").strip()
            if not raw_case:
                print("[ERROR] Case ID cannot be empty.")
                continue
            case_id = normalize_case_id(raw_case)
            case_dir = get_case_dir(case_id)
            if not case_dir.exists():
                print(f"[WARNING] Case folder not found: {case_dir}")
                confirm = input("Continue anyway? Enter y to continue, or any other key to re-enter: ").strip().lower()
                if confirm != "y":
                    continue
            return [case_id]

        if mode == "3":
            print("Enter multiple case IDs. Supported formats: 001 002 003 / 001,002,003 / 001-010 / 001 004 034-059")
            raw_text = input("Enter the case IDs to process: ").strip()
            if not raw_text:
                print("[ERROR] Input cannot be empty.")
                continue
            try:
                case_ids = parse_multiple_cases(raw_text)
            except Exception as e:
                print(f"[ERROR] Failed to parse case IDs: {e}")
                continue
            if not case_ids:
                print("[ERROR] No valid case ID was parsed.")
                continue
            print("The following cases will be processed:")
            print(" ".join(case_ids))
            missing = [case_id for case_id in case_ids if not get_case_dir(case_id).exists()]
            if missing:
                print("[WARNING] The following case folders were not found:")
                print(" ".join(missing))
                confirm = input("Continue anyway? Enter y to continue, or any other key to re-enter: ").strip().lower()
                if confirm != "y":
                    continue
            return case_ids

        print("[ERROR] Invalid mode number. Please try again.")


def run_menu_choice(choice: str, case_ids):
    if choice in MENU:
        item = MENU[choice]
        print("=" * 80)
        print(f"[RUN] {item['name']}")
        print("=" * 80)
        item["runner"](case_ids)
        return

    if choice == "7":
        for key in ALL_ANALYSIS_KEYS:
            item = MENU[key]
            print("=" * 80)
            print(f"[RUN] {item['name']}")
            print("=" * 80)
            item["runner"](case_ids)
        return

    if choice == "8":
        for key in FULL_SYSTEM_KEYS:
            item = MENU[key]
            print("=" * 80)
            print(f"[RUN] {item['name']}")
            print("=" * 80)
            item["runner"](case_ids)
        return

    print("[ERROR] Invalid function number.")


def main():
    if not PROJECT_ROOT.exists():
        raise FileNotFoundError(f"PROJECT_ROOT not found: {PROJECT_ROOT}")

    while True:
        print_main_menu()
        choice = input("Enter function number: ").strip()

        if choice == "0":
            print("[EXIT] Exited.")
            break

        if choice not in MENU and choice not in ["7", "8"]:
            print("[ERROR] Invalid function number. Please try again.")
            continue

        if choice in MENU:
            selected = MENU[choice]
            print(f"Selected: {selected['name']}")
            print(f"Description: {selected['description']}")
        elif choice == "7":
            print("Selected: Run all 5 analysis functions")
        elif choice == "8":
            print("Selected: Full system: main pipeline + all analyses")

        case_ids = select_cases_by_mode()
        if case_ids is None:
            print("[BACK] Returning to function selection.")
            continue

        try:
            run_menu_choice(choice, case_ids)
        except Exception:
            print("[ERROR] An exception occurred while running the selected function.")
            traceback.print_exc()

        next_action = input("Select another function? Enter y to continue, or any other key to exit: ").strip().lower()
        if next_action != "y":
            print("[EXIT] Exited.")
            break


if __name__ == "__main__":
    main()
