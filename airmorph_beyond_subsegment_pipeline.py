"""
Deduplicated AirMorph branch-analysis and beyond-subsegment pipeline.

For each case, the script reads AirMorph outputs from:
    /home/jinanli24/AirMorph/sample_data/ATM22/<case_id>

It creates two output folders inside the same case directory:
    airway_sign_analysis/original_analysis
    airway_sign_analysis/generation_repair

The first stage measures stable branch geometry and QC only. Generation and
parent-child fields are deliberately deferred until unlabeled junction voxels
have been repaired. The second stage repairs the topology, selects or accepts a
root branch, calculates generation, and creates candidate beyond-subsegment
labels and optional three-dimensional label volumes.

The original AirMorph files are never overwritten.
"""

from __future__ import annotations

import argparse
import gc
import heapq
import json
import math
import traceback
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import nibabel as nib
import numpy as np
import pandas as pd
from scipy import ndimage


# =============================================================================
# Configuration
# =============================================================================

PROJECT_ROOT = Path('/home/jinanli24/AirMorph')
SAMPLE_ROOT = PROJECT_ROOT / 'sample_data' / 'ATM22'
OUTPUT_ROOT_NAME = 'airway_sign_analysis'
ANALYSIS_FOLDER_NAME = 'original_analysis'
GENERATION_FOLDER_NAME = 'generation_repair'
LOG_FOLDER_NAME = 'airway_sign_system_logs'
AFFINE_ATOL = 1e-05
EDT_MARGIN_VOXELS = 3
MIN_SEGMENT_VOXELS = 5
MIN_SEGMENT_LENGTH_MM = 2.0
LABEL_PURITY_WARNING = 0.8
COVERAGE_WARNING = 0.95
COVERAGE_FAILURE = 0.9
DIAMETER_LOW_PERCENTILE = 10.0
DIAMETER_HIGH_PERCENTILE = 90.0
MIN_DIAMETER_SAMPLES = 5
BRANCH_DEGREE_THRESHOLD = 3
JUNCTION_LABEL_START = 10000
STRUCTURE_26 = np.ones((3, 3, 3), dtype=np.uint8)
ALL_OFFSETS_26 = [(di, dj, dk) for di in (-1, 0, 1) for dj in (-1, 0, 1) for dk in (-1, 0, 1) if (di, dj, dk) != (0, 0, 0)]
POSITIVE_OFFSETS_26 = [offset for offset in ALL_OFFSETS_26 if offset > (0, 0, 0)]

# =============================================================================
# Data structures
# =============================================================================

@dataclass(frozen=True)
class CaseFiles:
    case_id: str
    case_dir: Path
    airway_mask: Path
    airway_skeleton: Path
    skeleton_parsing: Path
    volume_subsegment: Path | None
    instance_volume_parse: Path | None
    anno_json: Path | None
    class2anno_json: Path | None
    airway_graph_npy: Path | None
    airway_graph_cls_npy: Path | None

@dataclass(frozen=True)
class AnalysisPaths:
    output_dir: Path
    metrics_csv: Path
    image_qc_csv: Path
    labelled_skeleton: Path
    filtered_skeleton: Path
    unassigned_mask: Path
    volume_subsegments: Path | None
    filtered_volume_subsegments: Path | None
    summary_txt: Path

@dataclass
class LoadedCase:
    reference_image: nib.Nifti1Image
    airway_mask: np.ndarray
    skeleton: np.ndarray
    skeleton_labels: np.ndarray
    volume_labels: np.ndarray | None

# =============================================================================
# Case selection and path discovery
# =============================================================================

def normalize_case_id(value: str) -> str:
    value = str(value).strip()
    return f'{int(value):03d}' if value.isdigit() else value

def get_case_dir(case_id: str) -> Path:
    return SAMPLE_ROOT / normalize_case_id(case_id)

def get_default_case_ids() -> list[str]:
    if SAMPLE_ROOT.exists():
        case_ids = [normalize_case_id(path.name) for path in sorted(SAMPLE_ROOT.iterdir()) if path.is_dir() and path.name.isdigit()]
        if case_ids:
            return case_ids
    return [f'{number:03d}' for number in range(1, 60)]

def expand_case_token(token: str) -> list[str]:
    token = token.strip()
    if not token:
        return []
    if '-' not in token:
        return [normalize_case_id(token)]
    start_text, end_text = token.split('-', 1)
    start = int(start_text)
    end = int(end_text)
    if start > end:
        raise ValueError(f'Invalid case range: {token}')
    return [f'{number:03d}' for number in range(start, end + 1)]

def parse_multiple_cases(raw_text: str) -> list[str]:
    raw_text = raw_text.replace(',', ' ')
    case_ids: list[str] = []
    for token in raw_text.split():
        case_ids.extend(expand_case_token(token))
    unique: list[str] = []
    seen: set[str] = set()
    for case_id in case_ids:
        if case_id not in seen:
            seen.add(case_id)
            unique.append(case_id)
    return unique

def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None

def discover_case_files(case_id: str) -> CaseFiles:
    case_id = normalize_case_id(case_id)
    case_dir = get_case_dir(case_id)
    if not case_dir.is_dir():
        raise FileNotFoundError(f'Case folder not found: {case_dir}')
    airway_mask = first_existing([case_dir / 'airway_bin.nii.gz', case_dir / 'binary_only' / 'airway_bin.nii.gz'])
    airway_skeleton = first_existing([case_dir / 'airway_skeleton.nii.gz', case_dir / 'binary_only' / 'airway_skeleton.nii.gz'])
    skeleton_parsing = first_existing([case_dir / f'{case_id}_skel_parsing.nii.gz', case_dir / 'anatomy_only' / f'{case_id}_skel_parsing.nii.gz', case_dir / 'skel_parsing.nii.gz', case_dir / 'anatomy_only' / 'skel_parsing.nii.gz'])
    volume_subsegment = first_existing([case_dir / f'{case_id}_pred_sub.nii.gz', case_dir / 'anatomy_only' / f'{case_id}_pred_sub.nii.gz', case_dir / 'pred_sub.nii.gz', case_dir / 'anatomy_only' / 'pred_sub.nii.gz'])
    instance_volume_parse = first_existing([case_dir / f'{case_id}_parse.nii.gz', case_dir / 'anatomy_only' / f'{case_id}_parse.nii.gz', case_dir / 'parse.nii.gz', case_dir / 'anatomy_only' / 'parse.nii.gz'])
    missing: list[str] = []
    if airway_mask is None:
        missing.append('airway_bin.nii.gz')
    if airway_skeleton is None:
        missing.append('airway_skeleton.nii.gz')
    if skeleton_parsing is None:
        missing.append(f'{case_id}_skel_parsing.nii.gz')
    if missing:
        raise FileNotFoundError(f'Missing required files for case {case_id}: ' + ', '.join(missing))
    return CaseFiles(case_id=case_id, case_dir=case_dir, airway_mask=airway_mask, airway_skeleton=airway_skeleton, skeleton_parsing=skeleton_parsing, volume_subsegment=volume_subsegment, instance_volume_parse=instance_volume_parse, anno_json=first_existing([case_dir / f'{case_id}_anno.json', case_dir / 'anatomy_only' / f'{case_id}_anno.json']), class2anno_json=first_existing([PROJECT_ROOT / 'configs' / 'class2anno.json', case_dir / 'class2anno.json']), airway_graph_npy=first_existing([case_dir / f'{case_id}_airway_graph.npy', case_dir / 'anatomy_only' / f'{case_id}_airway_graph.npy']), airway_graph_cls_npy=first_existing([case_dir / f'{case_id}_airway_graph_cls.npy', case_dir / 'anatomy_only' / f'{case_id}_airway_graph_cls.npy']))

def analysis_paths(files: CaseFiles) -> AnalysisPaths:
    output_dir = files.case_dir / OUTPUT_ROOT_NAME / ANALYSIS_FOLDER_NAME
    case_id = files.case_id
    return AnalysisPaths(output_dir=output_dir, metrics_csv=output_dir / f'{case_id}_original_subsegment_metrics.csv', image_qc_csv=output_dir / f'{case_id}_original_image_level_qc.csv', labelled_skeleton=output_dir / f'{case_id}_original_subsegments_full.nii.gz', filtered_skeleton=output_dir / f'{case_id}_original_subsegments_qc_filtered.nii.gz', unassigned_mask=output_dir / f'{case_id}_original_unassigned_skeleton_voxels.nii.gz', volume_subsegments=output_dir / f'{case_id}_original_volume_subsegments_full.nii.gz' if files.volume_subsegment is not None else None, filtered_volume_subsegments=output_dir / f'{case_id}_original_volume_subsegments_qc_filtered.nii.gz' if files.volume_subsegment is not None else None, summary_txt=output_dir / f'{case_id}_original_summary.txt')

def generation_dir(files: CaseFiles) -> Path:
    return files.case_dir / OUTPUT_ROOT_NAME / GENERATION_FOLDER_NAME

# =============================================================================
# NIfTI and geometry helpers
# =============================================================================

def load_nifti(path: Path) -> tuple[nib.Nifti1Image, np.ndarray]:
    image = nib.load(str(path))
    data = np.nan_to_num(np.asanyarray(image.dataobj))
    if data.ndim != 3:
        raise ValueError(f'Expected a 3D NIfTI: {path}; got shape {data.shape}')
    return (image, data)

def validate_geometry(reference: nib.Nifti1Image, other: nib.Nifti1Image, name: str) -> None:
    if reference.shape != other.shape:
        raise ValueError(f'{name} shape mismatch: {other.shape} vs {reference.shape}')
    if not np.allclose(reference.affine, other.affine, atol=AFFINE_ATOL):
        raise ValueError(f'{name} affine mismatch beyond atol={AFFINE_ATOL}')

def save_nifti(data: np.ndarray, reference: nib.Nifti1Image, output_path: Path, dtype: np.dtype) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    header = reference.header.copy()
    header.set_data_dtype(dtype)
    output = nib.Nifti1Image(data.astype(dtype), reference.affine, header)
    qform, qcode = reference.get_qform(coded=True)
    sform, scode = reference.get_sform(coded=True)
    output.set_qform(qform if qform is not None else reference.affine, int(qcode) if int(qcode) > 0 else 1)
    output.set_sform(sform if sform is not None else reference.affine, int(scode) if int(scode) > 0 else 1)
    nib.save(output, str(output_path))

def world_coordinate(affine: np.ndarray, voxel: tuple[int, int, int] | np.ndarray) -> np.ndarray:
    voxel = np.asarray(voxel, dtype=float)
    homogeneous = np.array([voxel[0], voxel[1], voxel[2], 1.0])
    return (affine @ homogeneous)[:3]

def physical_distance(affine: np.ndarray, point_a: tuple[int, int, int], point_b: tuple[int, int, int]) -> float:
    return float(np.linalg.norm(world_coordinate(affine, point_a) - world_coordinate(affine, point_b)))

def edge_weight(affine: np.ndarray, offset: tuple[int, int, int]) -> float:
    return float(np.linalg.norm(affine[:3, :3] @ np.asarray(offset, dtype=float)))

def voxel_spacing(affine: np.ndarray) -> tuple[float, float, float]:
    return tuple((float(np.linalg.norm(affine[:3, axis])) for axis in range(3)))

def voxel_volume_mm3(affine: np.ndarray) -> float:
    return float(abs(np.linalg.det(affine[:3, :3])))

def crop_to_mask(mask: np.ndarray, margin: int) -> tuple[tuple[slice, slice, slice], np.ndarray]:
    coordinates = np.argwhere(mask)
    if len(coordinates) == 0:
        raise ValueError('Cannot crop an empty mask.')
    lower = np.maximum(coordinates.min(axis=0) - margin, 0)
    upper = np.minimum(coordinates.max(axis=0) + margin + 1, np.asarray(mask.shape))
    slices = tuple((slice(int(start), int(stop)) for start, stop in zip(lower, upper)))
    return (slices, lower)

def coordinates_by_label(label_array: np.ndarray) -> dict[int, np.ndarray]:
    coordinates = np.argwhere(label_array > 0)
    grouped: dict[int, list[np.ndarray]] = defaultdict(list)
    for coordinate in coordinates:
        grouped[int(label_array[tuple(coordinate)])].append(coordinate)
    return {label_id: np.asarray(items, dtype=np.int32) for label_id, items in grouped.items()}

def component_count_for_coordinates(coordinates: np.ndarray) -> int:
    if len(coordinates) == 0:
        return 0
    lower = coordinates.min(axis=0)
    upper = coordinates.max(axis=0) + 1
    shape = tuple((upper - lower).astype(int))
    local = np.zeros(shape, dtype=bool)
    shifted = coordinates - lower
    local[tuple(shifted.T)] = True
    return int(ndimage.label(local, structure=STRUCTURE_26)[1])

def safe_number(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return number if math.isfinite(number) else default

# =============================================================================
# Skeleton geometry and branch measurements
# =============================================================================

def build_weighted_voxel_graph(binary_mask: np.ndarray, affine: np.ndarray) -> tuple[list[tuple[int, int, int]], list[list[tuple[int, float]]]]:
    coordinates = np.argwhere(binary_mask)
    nodes = [tuple((int(value) for value in coordinate)) for coordinate in coordinates]
    node_to_id = {node: node_id for node_id, node in enumerate(nodes)}
    adjacency: list[list[tuple[int, float]]] = [[] for _ in nodes]
    for node_id, (i, j, k) in enumerate(nodes):
        for di, dj, dk in POSITIVE_OFFSETS_26:
            neighbour = (i + di, j + dj, k + dk)
            neighbour_id = node_to_id.get(neighbour)
            if neighbour_id is None:
                continue
            weight = physical_distance(affine, (i, j, k), neighbour)
            adjacency[node_id].append((neighbour_id, weight))
            adjacency[neighbour_id].append((node_id, weight))
    return (nodes, adjacency)

def graph_total_length(adjacency: list[list[tuple[int, float]]]) -> float:
    total = 0.0
    for node_id, neighbours in enumerate(adjacency):
        for neighbour_id, weight in neighbours:
            if node_id < neighbour_id:
                total += weight
    return float(total)

def dijkstra(adjacency: list[list[tuple[int, float]]], start: int) -> tuple[list[float], list[int | None]]:
    distances = [math.inf] * len(adjacency)
    previous: list[int | None] = [None] * len(adjacency)
    distances[start] = 0.0
    heap = [(0.0, start)]
    while heap:
        current_distance, node = heapq.heappop(heap)
        if current_distance > distances[node]:
            continue
        for neighbour, weight in adjacency[node]:
            candidate = current_distance + weight
            if candidate < distances[neighbour]:
                distances[neighbour] = candidate
                previous[neighbour] = node
                heapq.heappush(heap, (candidate, neighbour))
    return (distances, previous)

def reconstruct_path(previous: list[int | None], start: int, end: int) -> list[int]:
    path: list[int] = []
    current: int | None = end
    while current is not None:
        path.append(current)
        if current == start:
            break
        current = previous[current]
    path.reverse()
    return path if path and path[0] == start else []

def longest_endpoint_path(adjacency: list[list[tuple[int, float]]]) -> tuple[list[int], float, int, int, int]:
    degrees = np.asarray([len(neighbours) for neighbours in adjacency])
    endpoint_ids = np.flatnonzero(degrees == 1).tolist()
    branchpoint_count = int(np.sum(degrees >= BRANCH_DEGREE_THRESHOLD))
    isolated_count = int(np.sum(degrees == 0))
    if len(endpoint_ids) < 2:
        return ([], float('nan'), len(endpoint_ids), branchpoint_count, isolated_count)
    first = endpoint_ids[0]
    distances, _ = dijkstra(adjacency, first)
    endpoint_a = max(endpoint_ids, key=lambda node: distances[node])
    distances, previous = dijkstra(adjacency, endpoint_a)
    endpoint_b = max(endpoint_ids, key=lambda node: distances[node])
    path = reconstruct_path(previous, endpoint_a, endpoint_b)
    return (path, float(distances[endpoint_b]), len(endpoint_ids), branchpoint_count, isolated_count)

def turning_angles(path_node_ids: list[int], nodes: list[tuple[int, int, int]], affine: np.ndarray) -> tuple[float, float, float]:
    if len(path_node_ids) < 3:
        return (0.0, 0.0, 0.0)
    points = np.asarray([world_coordinate(affine, nodes[node_id]) for node_id in path_node_ids])
    vectors = np.diff(points, axis=0)
    norms = np.linalg.norm(vectors, axis=1)
    valid = norms > 0
    vectors = vectors[valid]
    norms = norms[valid]
    if len(vectors) < 2:
        return (0.0, 0.0, 0.0)
    unit_vectors = vectors / norms[:, None]
    cosines = np.sum(unit_vectors[:-1] * unit_vectors[1:], axis=1)
    angles = np.degrees(np.arccos(np.clip(cosines, -1.0, 1.0)))
    return (float(np.mean(angles)), float(np.max(angles)), float(np.sum(angles)))

def component_metrics(binary_mask: np.ndarray) -> dict[str, Any]:
    labelled, count = ndimage.label(binary_mask, structure=STRUCTURE_26)
    sizes = np.bincount(labelled.ravel())
    if len(sizes) > 0:
        sizes[0] = 0
    main_size = int(sizes.max()) if len(sizes) > 1 else 0
    total = int(binary_mask.sum())
    return {'component_count': int(count), 'main_component_voxels': main_size, 'main_component_ratio': main_size / total if total > 0 else 0.0, 'component_map': labelled}

def topology_status(component_count: int, endpoint_count: int, branchpoint_count: int) -> str:
    if component_count > 1:
        return 'disconnected'
    if branchpoint_count > 0 or endpoint_count > 2:
        return 'branched_label'
    if endpoint_count == 2:
        return 'valid_path'
    if endpoint_count == 1:
        return 'truncated'
    if endpoint_count == 0:
        return 'loop_or_closed'
    return 'unknown'

def calculate_radius_map(airway_mask: np.ndarray, spacing: tuple[float, float, float]) -> tuple[np.ndarray, np.ndarray]:
    crop_slices, crop_origin = crop_to_mask(airway_mask, EDT_MARGIN_VOXELS)
    radius_map = ndimage.distance_transform_edt(airway_mask[crop_slices], sampling=spacing).astype(np.float32)
    return (radius_map, crop_origin)

def sample_diameters(coordinates: np.ndarray, radius_map: np.ndarray, crop_origin: np.ndarray) -> np.ndarray:
    values: list[float] = []
    shape = np.asarray(radius_map.shape)
    for coordinate in coordinates:
        local = np.asarray(coordinate, dtype=int) - crop_origin
        if np.any(local < 0) or np.any(local >= shape):
            continue
        radius = float(radius_map[tuple(local)])
        if radius > 0:
            values.append(2.0 * radius)
    return np.asarray(values, dtype=float)

def match_volume_label(coordinates: np.ndarray, volume_labels: np.ndarray | None) -> tuple[int, float, int]:
    if volume_labels is None:
        return (-1, float('nan'), 0)
    sampled = volume_labels[tuple(coordinates.T)]
    sampled = sampled[sampled > 0].astype(np.int32)
    if len(sampled) == 0:
        return (0, 0.0, 0)
    counts = Counter(sampled.tolist())
    matched_label, count = counts.most_common(1)[0]
    return (int(matched_label), float(count / len(sampled)), len(counts))

def analyze_branch(case_id: str, branch_id: int, skeleton: np.ndarray, skeleton_labels: np.ndarray, volume_labels: np.ndarray | None, radius_map: np.ndarray, crop_origin: np.ndarray, affine: np.ndarray, voxel_volume: float) -> tuple[dict[str, Any], bool]:
    branch_mask = skeleton & (skeleton_labels == branch_id)
    coordinates = np.argwhere(branch_mask)
    skeleton_voxels = int(len(coordinates))
    component_info = component_metrics(branch_mask)
    component_count = int(component_info['component_count'])
    component_map = component_info['component_map']
    main_component_mask = np.zeros_like(branch_mask, dtype=bool)
    if component_count > 0:
        component_sizes = np.bincount(component_map.ravel())
        component_sizes[0] = 0
        main_component_id = int(np.argmax(component_sizes))
        main_component_mask = component_map == main_component_id
    nodes, adjacency = build_weighted_voxel_graph(main_component_mask, affine)
    main_component_length_mm = graph_total_length(adjacency)
    path_node_ids, longest_path_length_mm, endpoint_count, branchpoint_count, isolated_count = longest_endpoint_path(adjacency)
    total_length_mm = 0.0
    for component_id in range(1, component_count + 1):
        _, component_adjacency = build_weighted_voxel_graph(component_map == component_id, affine)
        total_length_mm += graph_total_length(component_adjacency)
    start_voxel: tuple[int, int, int] | None = None
    end_voxel: tuple[int, int, int] | None = None
    straight_distance_mm = float('nan')
    tortuosity = float('nan')
    mean_turn_angle_deg = float('nan')
    max_turn_angle_deg = float('nan')
    cumulative_turn_angle_deg = float('nan')
    if path_node_ids:
        start_voxel = nodes[path_node_ids[0]]
        end_voxel = nodes[path_node_ids[-1]]
        straight_distance_mm = physical_distance(affine, start_voxel, end_voxel)
        if straight_distance_mm > 0:
            tortuosity = longest_path_length_mm / straight_distance_mm
        mean_turn_angle_deg, max_turn_angle_deg, cumulative_turn_angle_deg = turning_angles(path_node_ids, nodes, affine)
    diameters = sample_diameters(coordinates, radius_map, crop_origin)
    diameter_sample_count = int(len(diameters))
    diameter_mean_mm = float('nan')
    diameter_median_mm = float('nan')
    diameter_p10_mm = float('nan')
    diameter_p90_mm = float('nan')
    diameter_min_mm = float('nan')
    diameter_max_mm = float('nan')
    diameter_std_mm = float('nan')
    stenosis_index = float('nan')
    ectasia_index = float('nan')
    if diameter_sample_count > 0:
        diameter_mean_mm = float(np.mean(diameters))
        diameter_median_mm = float(np.median(diameters))
        diameter_p10_mm = float(np.percentile(diameters, DIAMETER_LOW_PERCENTILE))
        diameter_p90_mm = float(np.percentile(diameters, DIAMETER_HIGH_PERCENTILE))
        diameter_min_mm = float(np.min(diameters))
        diameter_max_mm = float(np.max(diameters))
        diameter_std_mm = float(np.std(diameters))
        if diameter_sample_count >= MIN_DIAMETER_SAMPLES and diameter_median_mm > 0:
            stenosis_index = max(0.0, 1.0 - diameter_p10_mm / diameter_median_mm)
            ectasia_index = max(0.0, diameter_p90_mm / diameter_median_mm - 1.0)
    matched_volume_label, volume_label_purity, distinct_volume_labels = match_volume_label(coordinates, volume_labels)
    volume_voxels = 0
    volume_mm3 = float('nan')
    if volume_labels is not None and matched_volume_label > 0:
        volume_voxels = int(np.sum(volume_labels == matched_volume_label))
        volume_mm3 = volume_voxels * voxel_volume
    is_short_by_voxels = skeleton_voxels < MIN_SEGMENT_VOXELS
    is_short_by_length = total_length_mm < MIN_SEGMENT_LENGTH_MM
    is_short = is_short_by_voxels or is_short_by_length
    mixed_volume_label = volume_labels is not None and volume_label_purity < LABEL_PURITY_WARNING
    keep_in_qc = not is_short and component_count == 1 and (not mixed_volume_label)
    quality_flags: list[str] = []
    if is_short_by_voxels:
        quality_flags.append('short_voxel_count')
    if is_short_by_length:
        quality_flags.append('short_length')
    if component_count > 1:
        quality_flags.append('disconnected_label')
    if branchpoint_count > 0:
        quality_flags.append('internal_branchpoints')
    if endpoint_count != 2:
        quality_flags.append(f'endpoint_count_{endpoint_count}')
    if mixed_volume_label:
        quality_flags.append('mixed_volume_label')
    if diameter_sample_count < MIN_DIAMETER_SAMPLES:
        quality_flags.append('insufficient_diameter_samples')
    row: dict[str, Any] = {'case_id': case_id, 'subsegment_label': int(branch_id), 'generation': -1, 'parent_label': -1, 'child_count': 0, 'local_complexity_2gen': 0, 'skeleton_voxels': skeleton_voxels, 'connected_components': component_count, 'main_component_voxels': int(component_info['main_component_voxels']), 'main_component_ratio': float(component_info['main_component_ratio']), 'endpoint_count_main_component': endpoint_count, 'branchpoint_count_main_component': branchpoint_count, 'isolated_count_main_component': isolated_count, 'topology_status': topology_status(component_count, endpoint_count, branchpoint_count), 'total_length_mm': float(total_length_mm), 'main_component_length_mm': float(main_component_length_mm), 'longest_path_length_mm': float(longest_path_length_mm), 'straight_distance_mm': straight_distance_mm, 'tortuosity': tortuosity, 'mean_turn_angle_deg': mean_turn_angle_deg, 'max_turn_angle_deg': max_turn_angle_deg, 'cumulative_turn_angle_deg': cumulative_turn_angle_deg, 'diameter_sample_count': diameter_sample_count, 'diameter_mean_mm': diameter_mean_mm, 'diameter_median_mm': diameter_median_mm, 'diameter_p10_mm': diameter_p10_mm, 'diameter_p90_mm': diameter_p90_mm, 'diameter_min_mm': diameter_min_mm, 'diameter_max_mm': diameter_max_mm, 'diameter_std_mm': diameter_std_mm, 'stenosis_index_p10_median': stenosis_index, 'ectasia_index_p90_median': ectasia_index, 'matched_volume_label': matched_volume_label, 'volume_label_purity': volume_label_purity, 'distinct_volume_labels_on_skeleton': distinct_volume_labels, 'volume_voxels': volume_voxels, 'volume_mm3': volume_mm3, 'divergence_angle_estimate_deg': float('nan'), 'is_short_segment': bool(is_short), 'keep_in_qc_filtered_output': bool(keep_in_qc), 'quality_flags': ';'.join(quality_flags) if quality_flags else 'none'}
    if start_voxel is not None and end_voxel is not None:
        row.update({'start_i': int(start_voxel[0]), 'start_j': int(start_voxel[1]), 'start_k': int(start_voxel[2]), 'end_i': int(end_voxel[0]), 'end_j': int(end_voxel[1]), 'end_k': int(end_voxel[2])})
    else:
        row.update({'start_i': -1, 'start_j': -1, 'start_k': -1, 'end_i': -1, 'end_j': -1, 'end_k': -1})
    return (row, keep_in_qc)

# =============================================================================
# Anatomical-name loading
# =============================================================================

def extract_name(value: Any) -> str | None:
    if isinstance(value, str):
        return value
    if isinstance(value, (list, tuple)):
        strings = [item for item in value if isinstance(item, str) and item.strip()]
        return strings[-1] if strings else None
    if isinstance(value, dict):
        for key in ('subsegment', 'subseg', 'subsegment_name', 'name', 'label', 'anatomical_name', 'anatomy'):
            if key in value:
                name = extract_name(value[key])
                if name:
                    return name
        for child in value.values():
            name = extract_name(child)
            if name:
                return name
    return None

def load_instance_names(path: Path | None) -> dict[int, str]:
    if path is None:
        return {}
    try:
        raw = json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return {}
    result: dict[int, str] = {}
    if isinstance(raw, dict):
        for key, value in raw.items():
            try:
                instance_id = int(key)
            except Exception:
                continue
            name = extract_name(value)
            if name:
                result[instance_id] = name
    return result

def load_class_names(path: Path | None) -> dict[int, str]:
    if path is None:
        return {}
    try:
        raw = json.loads(path.read_text(encoding='utf-8'))
    except Exception:
        return {}
    candidate = raw
    if isinstance(raw, dict):
        for key in ('subseg', 'subsegment', 'subsegments'):
            if key in raw and isinstance(raw[key], (dict, list)):
                candidate = raw[key]
                break
    zero_based: dict[int, str] = {}
    if isinstance(candidate, dict):
        for key, value in candidate.items():
            try:
                index = int(key)
            except Exception:
                continue
            name = extract_name(value)
            if name:
                zero_based[index] = name
    elif isinstance(candidate, list):
        for index, value in enumerate(candidate):
            name = extract_name(value)
            if name:
                zero_based[index] = name
    return {index + 1: name for index, name in zero_based.items()}

# =============================================================================
# Junction repair and topology construction
# =============================================================================

def multi_source_fill_labels(skeleton: np.ndarray, labels: np.ndarray, affine: np.ndarray) -> tuple[np.ndarray, dict[tuple[int, int, int], float]]:
    repaired = labels.astype(np.int16, copy=True)
    coordinates = [tuple(map(int, coordinate)) for coordinate in np.argwhere(skeleton)]
    coordinate_set = set(coordinates)
    if not coordinates:
        raise ValueError('The original skeleton is empty.')
    distance = {coordinate: math.inf for coordinate in coordinates}
    owner = {coordinate: 0 for coordinate in coordinates}
    heap: list[tuple[float, int, int, int, int]] = []
    for coordinate in coordinates:
        label_id = int(labels[coordinate])
        if label_id <= 0:
            continue
        distance[coordinate] = 0.0
        owner[coordinate] = label_id
        heapq.heappush(heap, (0.0, label_id, coordinate[0], coordinate[1], coordinate[2]))
    if not heap:
        raise ValueError('The labelled skeleton contains no positive labels.')
    weights = {offset: edge_weight(affine, offset) for offset in ALL_OFFSETS_26}
    while heap:
        current_distance, label_id, i, j, k = heapq.heappop(heap)
        coordinate = (i, j, k)
        if current_distance > distance[coordinate] + 1e-07:
            continue
        if label_id != owner[coordinate] and abs(current_distance - distance[coordinate]) <= 1e-07:
            continue
        for offset in ALL_OFFSETS_26:
            neighbour = (i + offset[0], j + offset[1], k + offset[2])
            if neighbour not in coordinate_set:
                continue
            candidate = current_distance + weights[offset]
            old_distance = distance[neighbour]
            old_owner = owner[neighbour]
            better = candidate < old_distance - 1e-07
            tied_better = abs(candidate - old_distance) <= 1e-07 and (old_owner == 0 or label_id < old_owner)
            if better or tied_better:
                distance[neighbour] = candidate
                owner[neighbour] = label_id
                heapq.heappush(heap, (candidate, label_id, neighbour[0], neighbour[1], neighbour[2]))
    for coordinate in coordinates:
        repaired[coordinate] = owner[coordinate]
    return (repaired, distance)

def touching_branch_labels(component_coordinates: list[tuple[int, int, int]], original_labels: np.ndarray) -> list[int]:
    shape = original_labels.shape
    touching: set[int] = set()
    for i, j, k in component_coordinates:
        for di, dj, dk in ALL_OFFSETS_26:
            ni = i + di
            nj = j + dj
            nk = k + dk
            if not (0 <= ni < shape[0] and 0 <= nj < shape[1] and (0 <= nk < shape[2])):
                continue
            value = int(original_labels[ni, nj, nk])
            if value > 0:
                touching.add(value)
    return sorted(touching)

def build_topology_graph(skeleton: np.ndarray, original_labels: np.ndarray) -> tuple[dict[str, set[str]], pd.DataFrame, np.ndarray]:
    graph: dict[str, set[str]] = defaultdict(set)
    branch_ids = sorted((int(value) for value in np.unique(original_labels) if int(value) > 0))
    for branch_id in branch_ids:
        graph[f'B{branch_id}']
    skeleton_coordinates = [tuple(map(int, coordinate)) for coordinate in np.argwhere(skeleton)]
    skeleton_set = set(skeleton_coordinates)
    direct_pairs: set[tuple[int, int]] = set()
    for i, j, k in skeleton_coordinates:
        label_a = int(original_labels[i, j, k])
        if label_a <= 0:
            continue
        for di, dj, dk in POSITIVE_OFFSETS_26:
            neighbour = (i + di, j + dj, k + dk)
            if neighbour not in skeleton_set:
                continue
            label_b = int(original_labels[neighbour])
            if label_b > 0 and label_b != label_a:
                direct_pairs.add(tuple(sorted((label_a, label_b))))
    for label_a, label_b in direct_pairs:
        node_a = f'B{label_a}'
        node_b = f'B{label_b}'
        graph[node_a].add(node_b)
        graph[node_b].add(node_a)
    unassigned = skeleton & (original_labels == 0)
    junction_map, junction_count = ndimage.label(unassigned, structure=STRUCTURE_26)
    grouped: dict[int, list[tuple[int, int, int]]] = defaultdict(list)
    for coordinate in np.argwhere(junction_map > 0):
        voxel = tuple(map(int, coordinate))
        grouped[int(junction_map[voxel])].append(voxel)
    rows: list[dict[str, Any]] = []
    junction_aware = original_labels.astype(np.int16, copy=True)
    for component_id in range(1, junction_count + 1):
        coordinates = grouped.get(component_id, [])
        touching = touching_branch_labels(coordinates, original_labels)
        junction_node = f'J{component_id}'
        graph[junction_node]
        for branch_id in touching:
            branch_node = f'B{branch_id}'
            graph[junction_node].add(branch_node)
            graph[branch_node].add(junction_node)
        output_label = JUNCTION_LABEL_START + component_id
        for coordinate in coordinates:
            junction_aware[coordinate] = output_label
        if len(touching) >= 3:
            junction_type = 'bifurcation_or_multifurcation'
        elif len(touching) == 2:
            junction_type = 'connection'
        elif len(touching) == 1:
            junction_type = 'unlabelled_tail'
        else:
            junction_type = 'orphan'
        rows.append({'junction_id': component_id, 'junction_output_label': output_label, 'voxel_count': len(coordinates), 'touching_branch_count': len(touching), 'touching_branch_labels': ';'.join(map(str, touching)), 'junction_type': junction_type})
    return (graph, pd.DataFrame(rows), junction_aware)

def graph_connected_components(graph: dict[str, set[str]]) -> list[set[str]]:
    unseen = set(graph)
    components: list[set[str]] = []
    while unseen:
        start = min(unseen)
        component: set[str] = set()
        queue = deque([start])
        unseen.remove(start)
        while queue:
            node = queue.popleft()
            component.add(node)
            for neighbour in graph.get(node, set()):
                if neighbour in unseen:
                    unseen.remove(neighbour)
                    queue.append(neighbour)
        components.append(component)
    return components

def branch_world_statistics(branch_ids: list[int], original_labels: np.ndarray, affine: np.ndarray) -> dict[int, dict[str, float]]:
    statistics: dict[int, dict[str, float]] = {}
    for branch_id in branch_ids:
        coordinates = np.argwhere(original_labels == branch_id)
        if len(coordinates) == 0:
            statistics[branch_id] = {'max_z': -math.inf, 'mean_z': -math.inf, 'voxel_count': 0.0}
            continue
        homogeneous = np.c_[coordinates.astype(float), np.ones(len(coordinates))]
        world = (affine @ homogeneous.T).T[:, :3]
        statistics[branch_id] = {'max_z': float(np.max(world[:, 2])), 'mean_z': float(np.mean(world[:, 2])), 'voxel_count': float(len(coordinates))}
    return statistics

def choose_root_label(branch_ids: list[int], skeleton: np.ndarray, original_labels: np.ndarray, repaired_labels: np.ndarray, affine: np.ndarray, metrics: pd.DataFrame, instance_names: dict[int, str]) -> tuple[int, str]:
    branch_set = set(branch_ids)
    trachea_candidates = [branch_id for branch_id, name in instance_names.items() if branch_id in branch_set and 'trache' in str(name).lower()]
    kernel = np.ones((3, 3, 3), dtype=np.uint8)
    kernel[1, 1, 1] = 0
    neighbour_count = ndimage.convolve(skeleton.astype(np.uint8), kernel, mode='constant', cval=0)
    endpoints = np.argwhere(skeleton & (neighbour_count == 1))
    superior_endpoint_label = 0
    if len(endpoints) > 0:
        superior_endpoint = max(endpoints, key=lambda coordinate: world_coordinate(affine, coordinate)[2])
        superior_endpoint_label = int(repaired_labels[tuple(superior_endpoint)])
    if superior_endpoint_label in trachea_candidates:
        return (superior_endpoint_label, 'superior_endpoint_named_trachea')
    world_stats = branch_world_statistics(branch_ids, original_labels, affine)
    metric_lookup: dict[int, dict[str, float]] = {}
    if 'subsegment_label' in metrics.columns:
        for _, row in metrics.iterrows():
            branch_id = int(row['subsegment_label'])
            metric_lookup[branch_id] = {'diameter': safe_number(row.get('diameter_median_mm', 0.0)), 'length': safe_number(row.get('total_length_mm', 0.0)), 'voxels': safe_number(row.get('skeleton_voxels', 0.0))}
    if trachea_candidates:
        best = max(trachea_candidates, key=lambda branch_id: (metric_lookup.get(branch_id, {}).get('diameter', 0.0), metric_lookup.get(branch_id, {}).get('length', 0.0), world_stats[branch_id]['max_z'], metric_lookup.get(branch_id, {}).get('voxels', 0.0), -branch_id))
        return (best, 'ranked_trachea_candidate')
    if superior_endpoint_label in branch_set:
        return (superior_endpoint_label, 'superior_endpoint')
    best = max(branch_ids, key=lambda branch_id: (metric_lookup.get(branch_id, {}).get('diameter', 0.0), metric_lookup.get(branch_id, {}).get('length', 0.0), metric_lookup.get(branch_id, {}).get('voxels', 0.0), world_stats[branch_id]['max_z'], -branch_id))
    return (best, 'largest_diameter_then_length')

def root_branch_tree(graph: dict[str, set[str]], root_label: int) -> tuple[dict[int, int], dict[int, int], dict[int, list[int]], list[tuple[int, int, str]], set[str]]:
    root_node = f'B{root_label}'
    if root_node not in graph:
        raise ValueError(f'Root branch {root_label} is absent from the graph.')
    parent = {root_label: -1}
    generation = {root_label: 0}
    children: dict[int, list[int]] = defaultdict(list)
    edges: list[tuple[int, int, str]] = []
    visited = {root_node}
    queue = deque([(root_node, root_label)])
    while queue:
        node, most_recent_branch = queue.popleft()
        for neighbour in sorted(graph.get(node, set())):
            if neighbour in visited:
                continue
            visited.add(neighbour)
            if neighbour.startswith('B'):
                child = int(neighbour[1:])
                if child != most_recent_branch:
                    parent[child] = most_recent_branch
                    generation[child] = generation[most_recent_branch] + 1
                    children[most_recent_branch].append(child)
                    via = node if node.startswith('J') else 'direct_contact'
                    edges.append((most_recent_branch, child, via))
                queue.append((neighbour, child))
            else:
                queue.append((neighbour, most_recent_branch))
    for branch_id in list(children):
        children[branch_id] = sorted(set(children[branch_id]))
    return (parent, generation, children, edges, visited)

# =============================================================================
# Beyond-subsegment mapping and 3D volume construction
# =============================================================================

def branch_to_class_mapping(repaired_coordinates: dict[int, np.ndarray], metrics: pd.DataFrame, volume_classes: np.ndarray | None) -> dict[int, tuple[int, float]]:
    result: dict[int, tuple[int, float]] = {}
    required = {'subsegment_label', 'matched_volume_label', 'volume_label_purity'}
    if required.issubset(metrics.columns):
        for _, row in metrics.iterrows():
            branch_id = int(row['subsegment_label'])
            class_id = int(row['matched_volume_label'])
            purity = float(row['volume_label_purity'])
            if branch_id > 0 and class_id > 0:
                result[branch_id] = (class_id, purity)
    for branch_id, coordinates in repaired_coordinates.items():
        if branch_id in result:
            continue
        if volume_classes is None:
            result[branch_id] = (0, 0.0)
            continue
        values = volume_classes[tuple(coordinates.T)]
        values = values[values > 0].astype(np.int32)
        if len(values) == 0:
            result[branch_id] = (0, 0.0)
            continue
        counts = Counter(values.tolist())
        class_id, count = counts.most_common(1)[0]
        result[branch_id] = (int(class_id), float(count / len(values)))
    return result

def stable_child_order(child_ids: list[int], centroids: dict[int, np.ndarray]) -> list[int]:
    return sorted(child_ids, key=lambda child_id: (round(float(centroids[child_id][2]), 6), round(float(centroids[child_id][1]), 6), round(float(centroids[child_id][0]), 6), child_id))

def build_beyond_mapping(branch_ids: list[int], parent: dict[int, int], generation: dict[int, int], children: dict[int, list[int]], branch_classes: dict[int, tuple[int, float]], class_names: dict[int, str], instance_names: dict[int, str], centroids: dict[int, np.ndarray]) -> pd.DataFrame:
    same_class_children: dict[int, list[int]] = defaultdict(list)
    for branch_id in branch_ids:
        parent_id = parent.get(branch_id, -1)
        if parent_id > 0 and branch_classes.get(parent_id, (0, 0))[0] == branch_classes.get(branch_id, (0, 0))[0]:
            same_class_children[parent_id].append(branch_id)
    for parent_id in list(same_class_children):
        same_class_children[parent_id] = stable_child_order(same_class_children[parent_id], centroids)
    roots_by_class: dict[int, list[int]] = defaultdict(list)
    for branch_id in branch_ids:
        class_id = branch_classes.get(branch_id, (0, 0))[0]
        parent_id = parent.get(branch_id, -1)
        if parent_id <= 0 or branch_classes.get(parent_id, (0, 0))[0] != class_id:
            roots_by_class[class_id].append(branch_id)
    for class_id in list(roots_by_class):
        roots_by_class[class_id] = sorted(roots_by_class[class_id], key=lambda branch_id: (generation.get(branch_id, 10 ** 9), branch_id))
    local_generation: dict[int, int] = {}
    local_path: dict[int, str] = {}
    class_group: dict[int, int] = {}
    for class_id, roots in sorted(roots_by_class.items()):
        for group_index, root in enumerate(roots, start=1):
            local_generation[root] = 0
            class_group[root] = group_index
            local_path[root] = '' if len(roots) == 1 else f'R{group_index}'
            queue = deque([root])
            while queue:
                current = queue.popleft()
                ordered_children = same_class_children.get(current, [])
                for sibling_index, child in enumerate(ordered_children, start=1):
                    local_generation[child] = local_generation[current] + 1
                    class_group[child] = group_index
                    prefix = local_path[current]
                    local_path[child] = f'{prefix}.{sibling_index}' if prefix else str(sibling_index)
                    queue.append(child)
    ordered_branches = sorted(branch_ids, key=lambda branch_id: (generation.get(branch_id, 10 ** 9), branch_id))
    beyond_numeric = {branch_id: index + 1 for index, branch_id in enumerate(ordered_branches)}
    rows: list[dict[str, Any]] = []
    for branch_id in ordered_branches:
        class_id, purity = branch_classes.get(branch_id, (0, 0.0))
        base_name = instance_names.get(branch_id) or class_names.get(class_id) or f'Class{class_id}'
        path = local_path.get(branch_id, '')
        beyond_name = f'{base_name}.{path}' if path else base_name
        rows.append({'branch_instance_id': branch_id, 'beyond_numeric_label': beyond_numeric[branch_id], 'proposed_beyond_name': beyond_name, 'parent_branch_instance_id': parent.get(branch_id, -1), 'children_branch_instance_ids': ';'.join(map(str, children.get(branch_id, []))), 'absolute_generation': generation.get(branch_id, -1), 'airmorph_class_label': class_id, 'airmorph_anatomical_name': base_name, 'class_mapping_purity': purity, 'class_local_tree_group': class_group.get(branch_id, -1), 'generation_below_airmorph_class': local_generation.get(branch_id, -1), 'hierarchical_path_within_class': path, 'naming_status': 'anatomical_name_available' if branch_id in instance_names or class_id in class_names else 'class_id_only'})
    return pd.DataFrame(rows)

def reconstruct_instance_volume(airway_mask: np.ndarray, repaired_labels: np.ndarray) -> np.ndarray:
    crop_slices, _ = crop_to_mask(airway_mask | (repaired_labels > 0), margin=3)
    mask_crop = airway_mask[crop_slices]
    labels_crop = repaired_labels[crop_slices]
    seeds = labels_crop > 0
    if not np.any(seeds):
        raise ValueError('No repaired skeleton labels are available for volume reconstruction.')
    _, nearest_indices = ndimage.distance_transform_edt(~seeds, return_indices=True)
    nearest_labels = labels_crop[tuple(nearest_indices)]
    output_crop = np.where(mask_crop, nearest_labels, 0).astype(np.int16)
    output = np.zeros(airway_mask.shape, dtype=np.int16)
    output[crop_slices] = output_crop
    return output

# =============================================================================
# Per-case pipeline
# =============================================================================

class AirMorphCasePipeline:
    """Load one case once and run analysis and generation from shared data."""

    def __init__(self, case_id: str, overwrite: bool=False, root_override: int | None=None, reconstruct_volume: bool=True, save_distance: bool=False) -> None:
        self.files = discover_case_files(case_id)
        self.paths = analysis_paths(self.files)
        self.generation_output_dir = generation_dir(self.files)
        self.overwrite = overwrite
        self.root_override = root_override
        self.reconstruct_volume = reconstruct_volume
        self.save_distance = save_distance
        self.loaded: LoadedCase | None = None

    def load(self) -> LoadedCase:
        if self.loaded is not None:
            return self.loaded
        mask_image, mask_data = load_nifti(self.files.airway_mask)
        skeleton_image, skeleton_data = load_nifti(self.files.airway_skeleton)
        parsing_image, parsing_data = load_nifti(self.files.skeleton_parsing)
        validate_geometry(skeleton_image, mask_image, 'airway mask')
        validate_geometry(skeleton_image, parsing_image, 'skeleton parsing')
        volume_labels: np.ndarray | None = None
        if self.files.volume_subsegment is not None:
            volume_image, volume_data = load_nifti(self.files.volume_subsegment)
            validate_geometry(skeleton_image, volume_image, 'volume subsegment')
            volume_labels = np.rint(volume_data).astype(np.int32)
        self.loaded = LoadedCase(reference_image=skeleton_image, airway_mask=np.asarray(mask_data > 0, dtype=bool), skeleton=np.asarray(skeleton_data > 0, dtype=bool), skeleton_labels=np.rint(parsing_data).astype(np.int32), volume_labels=volume_labels)
        return self.loaded

    def analysis_complete(self) -> bool:
        required = [self.paths.metrics_csv, self.paths.image_qc_csv, self.paths.labelled_skeleton, self.paths.filtered_skeleton, self.paths.unassigned_mask, self.paths.summary_txt]
        if self.paths.volume_subsegments is not None:
            required.append(self.paths.volume_subsegments)
        if self.paths.filtered_volume_subsegments is not None:
            required.append(self.paths.filtered_volume_subsegments)
        return all((path.exists() for path in required))

    def generation_complete(self) -> bool:
        case_id = self.files.case_id
        required = [self.generation_output_dir / f'{case_id}_instance_generation_mapping.csv', self.generation_output_dir / f'{case_id}_generation_repair_qc.csv', self.generation_output_dir / f'{case_id}_generation_repair_report.txt', self.generation_output_dir / f'{case_id}_repaired_branch_skeleton.nii.gz', self.generation_output_dir / f'{case_id}_junction_aware_skeleton.nii.gz', self.generation_output_dir / f'{case_id}_beyond_skeleton_labels.nii.gz']
        if self.reconstruct_volume:
            required.append(self.generation_output_dir / f'{case_id}_beyond_instance_volume.nii.gz')
        return all((path.exists() for path in required))

    def run_analysis(self) -> pd.DataFrame:
        if self.analysis_complete() and (not self.overwrite):
            print(f'[SKIP] {self.files.case_id}: original analysis already exists')
            return pd.read_csv(self.paths.metrics_csv)
        data = self.load()
        self.paths.output_dir.mkdir(parents=True, exist_ok=True)
        print('\n' + '=' * 80)
        print(f'[ORIGINAL ANALYSIS] CASE {self.files.case_id}')
        print('=' * 80)
        print('Case directory    :', self.files.case_dir)
        print('Airway mask       :', self.files.airway_mask)
        print('Airway skeleton   :', self.files.airway_skeleton)
        print('Skeleton parsing  :', self.files.skeleton_parsing)
        print('Volume labels     :', self.files.volume_subsegment)
        print('Output directory  :', self.paths.output_dir)
        skeleton_voxels = int(data.skeleton.sum())
        assigned_mask = data.skeleton & (data.skeleton_labels > 0)
        assigned_voxels = int(assigned_mask.sum())
        unassigned_voxels = skeleton_voxels - assigned_voxels
        coverage = assigned_voxels / skeleton_voxels if skeleton_voxels else 0.0
        if coverage < COVERAGE_FAILURE:
            coverage_status = 'failed'
        elif coverage < COVERAGE_WARNING:
            coverage_status = 'warning'
        else:
            coverage_status = 'passed'
        branch_ids = sorted((int(value) for value in np.unique(data.skeleton_labels[assigned_mask]) if int(value) > 0))
        if not branch_ids:
            raise ValueError('No positive AirMorph skeleton labels were found.')
        radius_map, crop_origin = calculate_radius_map(data.airway_mask, voxel_spacing(data.reference_image.affine))
        voxel_volume = voxel_volume_mm3(data.reference_image.affine)
        full_labels = np.where(data.skeleton, data.skeleton_labels, 0).astype(np.int32)
        filtered_labels = np.zeros_like(full_labels, dtype=np.int32)
        rows: list[dict[str, Any]] = []
        kept_branch_ids: list[int] = []
        for branch_id in branch_ids:
            row, keep = analyze_branch(case_id=self.files.case_id, branch_id=branch_id, skeleton=data.skeleton, skeleton_labels=data.skeleton_labels, volume_labels=data.volume_labels, radius_map=radius_map, crop_origin=crop_origin, affine=data.reference_image.affine, voxel_volume=voxel_volume)
            rows.append(row)
            if keep:
                filtered_labels[data.skeleton & (data.skeleton_labels == branch_id)] = branch_id
                kept_branch_ids.append(branch_id)
        metrics = pd.DataFrame(rows).sort_values('subsegment_label', kind='stable')
        metrics.to_csv(self.paths.metrics_csv, index=False)
        save_nifti(full_labels, data.reference_image, self.paths.labelled_skeleton, np.int32)
        save_nifti(filtered_labels, data.reference_image, self.paths.filtered_skeleton, np.int32)
        unassigned_mask = (data.skeleton & (data.skeleton_labels <= 0)).astype(np.uint8)
        save_nifti(unassigned_mask, data.reference_image, self.paths.unassigned_mask, np.uint8)
        if data.volume_labels is not None and self.paths.volume_subsegments is not None:
            save_nifti(data.volume_labels, data.reference_image, self.paths.volume_subsegments, np.int32)
            kept_volume_labels = sorted({int(row['matched_volume_label']) for row in rows if row['keep_in_qc_filtered_output'] and int(row['matched_volume_label']) > 0})
            filtered_volume = np.where(np.isin(data.volume_labels, kept_volume_labels), data.volume_labels, 0).astype(np.int32)
            if self.paths.filtered_volume_subsegments is not None:
                save_nifti(filtered_volume, data.reference_image, self.paths.filtered_volume_subsegments, np.int32)
        whole_map, whole_count = ndimage.label(data.skeleton, structure=STRUCTURE_26)
        whole_sizes = np.bincount(whole_map.ravel())
        if len(whole_sizes) > 0:
            whole_sizes[0] = 0
        main_component_voxels = int(whole_sizes.max()) if len(whole_sizes) > 1 else 0
        main_component_ratio = main_component_voxels / skeleton_voxels if skeleton_voxels else 0.0
        qc = {'case_id': self.files.case_id, 'analysis_target': 'original_airmorph_outputs', 'orientation_code': ''.join(nib.aff2axcodes(data.reference_image.affine)), 'array_order': 'original_ijk_no_transpose', 'airway_mask_source': self.files.airway_mask.name, 'skeleton_source': self.files.airway_skeleton.name, 'skeleton_parsing_source': self.files.skeleton_parsing.name, 'volume_subsegment_source': self.files.volume_subsegment.name if self.files.volume_subsegment is not None else 'missing', 'skeleton_voxels': skeleton_voxels, 'assigned_skeleton_voxels': assigned_voxels, 'unassigned_skeleton_voxels': unassigned_voxels, 'subsegment_label_coverage': coverage, 'coverage_status': coverage_status, 'number_of_subsegment_labels': len(branch_ids), 'number_of_qc_kept_labels': len(kept_branch_ids), 'whole_skeleton_connected_components': int(whole_count), 'whole_skeleton_main_component_voxels': main_component_voxels, 'whole_skeleton_main_component_ratio': main_component_ratio, 'labels_with_multiple_components': int(np.sum(metrics['connected_components'] > 1)), 'labels_with_internal_branchpoints': int(np.sum(metrics['branchpoint_count_main_component'] > 0)), 'short_labels': int(np.sum(metrics['is_short_segment'])), 'low_volume_label_purity_labels': int(np.sum(metrics['volume_label_purity'].notna() & (metrics['volume_label_purity'] < LABEL_PURITY_WARNING))), 'generation_fields_status': 'deferred_until_junction_repair', 'affine_tolerance': AFFINE_ATOL, 'minimum_segment_voxels': MIN_SEGMENT_VOXELS, 'minimum_segment_length_mm': MIN_SEGMENT_LENGTH_MM, 'label_purity_warning': LABEL_PURITY_WARNING, 'coverage_warning': COVERAGE_WARNING, 'coverage_failure': COVERAGE_FAILURE, 'diameter_low_percentile': DIAMETER_LOW_PERCENTILE, 'diameter_high_percentile': DIAMETER_HIGH_PERCENTILE, 'minimum_diameter_samples': MIN_DIAMETER_SAMPLES}
        pd.DataFrame([qc]).to_csv(self.paths.image_qc_csv, index=False)
        with self.paths.summary_txt.open('w', encoding='utf-8') as file:
            file.write(f'Case ID: {self.files.case_id}\n')
            file.write('Method: direct measurement of original AirMorph branch-instance labels\n')
            file.write('Generation fields are intentionally deferred until junction repair.\n\n')
            file.write('INPUTS\n------\n')
            file.write(f'Airway mask: {self.files.airway_mask}\n')
            file.write(f'Airway skeleton: {self.files.airway_skeleton}\n')
            file.write(f'Skeleton parsing: {self.files.skeleton_parsing}\n')
            file.write(f'Volume subsegment: {self.files.volume_subsegment}\n\n')
            file.write('QUALITY CONTROL\n---------------\n')
            file.write(f'Skeleton voxels: {skeleton_voxels}\n')
            file.write(f'Assigned voxels: {assigned_voxels}\n')
            file.write(f'Unassigned voxels: {unassigned_voxels}\n')
            file.write(f'Coverage: {coverage:.6f}\n')
            file.write(f'Coverage status: {coverage_status}\n')
            file.write(f'Branch instances: {len(branch_ids)}\n')
            file.write(f'QC-kept branch instances: {len(kept_branch_ids)}\n')
        print(f'[DONE] {self.files.case_id}')
        print('  Coverage          :', f'{coverage:.2%}')
        print('  Branch instances  :', len(branch_ids))
        print('  QC-kept branches  :', len(kept_branch_ids))
        print('  Output directory  :', self.paths.output_dir)
        del radius_map, whole_map
        gc.collect()
        return metrics

    def run_generation(self) -> pd.DataFrame:
        if self.generation_complete() and (not self.overwrite):
            print(f'[SKIP] {self.files.case_id}: generation repair already exists')
            return pd.read_csv(self.generation_output_dir / f'{self.files.case_id}_instance_generation_mapping.csv')
        if not self.analysis_complete():
            self.run_analysis()
        data = self.load()
        self.generation_output_dir.mkdir(parents=True, exist_ok=True)
        print('\n' + '=' * 80)
        print(f'[GENERATION REPAIR] CASE {self.files.case_id}')
        print('=' * 80)
        print('Case directory      :', self.files.case_dir)
        print('Analysis directory  :', self.paths.output_dir)
        print('Output directory    :', self.generation_output_dir)
        metrics = pd.read_csv(self.paths.metrics_csv)
        original_labels = np.where(data.skeleton, data.skeleton_labels, 0).astype(np.int16)
        branch_ids = sorted((int(value) for value in np.unique(original_labels) if int(value) > 0))
        instance_names = load_instance_names(self.files.anno_json)
        class_names = load_class_names(self.files.class2anno_json)
        skeleton_voxels = int(data.skeleton.sum())
        assigned_before = int(np.sum(data.skeleton & (original_labels > 0)))
        coverage_before = assigned_before / skeleton_voxels if skeleton_voxels else 0.0
        print('[STEP] Repairing unlabeled skeleton voxels...')
        repaired, repair_distance = multi_source_fill_labels(data.skeleton, original_labels, data.reference_image.affine)
        assigned_after = int(np.sum(data.skeleton & (repaired > 0)))
        coverage_after = assigned_after / skeleton_voxels if skeleton_voxels else 0.0
        print('[STEP] Building branch-junction topology...')
        graph, junctions, junction_aware = build_topology_graph(data.skeleton, original_labels)
        graph_components = graph_connected_components(graph)
        root_label, root_method = choose_root_label(branch_ids=branch_ids, skeleton=data.skeleton, original_labels=original_labels, repaired_labels=repaired, affine=data.reference_image.affine, metrics=metrics, instance_names=instance_names)
        if self.root_override is not None:
            if self.root_override not in branch_ids:
                raise ValueError(f'Requested root label {self.root_override} is not present.')
            root_label = self.root_override
            root_method = 'user_override'
        print('[STEP] Calculating parent-child tree and generation...')
        parent, generation, children, tree_edges, visited = root_branch_tree(graph, root_label)
        original_coordinates = coordinates_by_label(original_labels)
        repaired_coordinates = coordinates_by_label(repaired)
        centroids: dict[int, np.ndarray] = {}
        for branch_id in branch_ids:
            coordinates = original_coordinates.get(branch_id, np.empty((0, 3), dtype=np.int32))
            if len(coordinates) == 0:
                centroids[branch_id] = np.zeros(3)
                continue
            homogeneous = np.c_[coordinates.astype(float), np.ones(len(coordinates))]
            world = (data.reference_image.affine @ homogeneous.T).T[:, :3]
            centroids[branch_id] = world.mean(axis=0)
        branch_classes = branch_to_class_mapping(repaired_coordinates, metrics, data.volume_labels)
        mapping = build_beyond_mapping(branch_ids, parent, generation, children, branch_classes, class_names, instance_names, centroids)
        merge_columns = [column for column in ('subsegment_label', 'skeleton_voxels', 'total_length_mm', 'tortuosity', 'diameter_median_mm', 'volume_mm3', 'is_short_segment', 'quality_flags') if column in metrics.columns]
        mapping = mapping.merge(metrics[merge_columns], left_on='branch_instance_id', right_on='subsegment_label', how='left').drop(columns=['subsegment_label'], errors='ignore')
        branch_to_beyond = dict(zip(mapping['branch_instance_id'].astype(int), mapping['beyond_numeric_label'].astype(int)))
        lookup = np.zeros(int(repaired.max()) + 1, dtype=np.int16)
        for branch_id, beyond_label in branch_to_beyond.items():
            if 0 <= branch_id < len(lookup):
                lookup[branch_id] = beyond_label
        beyond_skeleton = lookup[repaired]
        case_id = self.files.case_id
        repaired_path = self.generation_output_dir / f'{case_id}_repaired_branch_skeleton.nii.gz'
        junction_aware_path = self.generation_output_dir / f'{case_id}_junction_aware_skeleton.nii.gz'
        junction_mask_path = self.generation_output_dir / f'{case_id}_junction_mask.nii.gz'
        beyond_skeleton_path = self.generation_output_dir / f'{case_id}_beyond_skeleton_labels.nii.gz'
        save_nifti(repaired, data.reference_image, repaired_path, np.int16)
        save_nifti(junction_aware, data.reference_image, junction_aware_path, np.int16)
        save_nifti(data.skeleton & (original_labels == 0), data.reference_image, junction_mask_path, np.uint8)
        save_nifti(beyond_skeleton, data.reference_image, beyond_skeleton_path, np.int16)
        if self.save_distance:
            distance_volume = np.zeros(data.skeleton.shape, dtype=np.float32)
            for coordinate, value in repair_distance.items():
                if math.isfinite(value):
                    distance_volume[coordinate] = value
            save_nifti(distance_volume, data.reference_image, self.generation_output_dir / f'{case_id}_repair_distance_mm.nii.gz', np.float32)
        volume_qc_rows: list[dict[str, Any]] = []
        volume_source = 'not_requested'
        if self.reconstruct_volume:
            print('[STEP] Creating three-dimensional beyond labels...')
            instance_volume: np.ndarray | None = None
            if self.files.instance_volume_parse is not None:
                try:
                    parse_image, parse_data = load_nifti(self.files.instance_volume_parse)
                    validate_geometry(data.reference_image, parse_image, 'AirMorph instance parse')
                    candidate = np.asarray(parse_data, dtype=np.int16)
                    present = {int(value) for value in np.unique(candidate) if int(value) > 0}
                    if present and present.issubset(set(branch_ids)):
                        instance_volume = candidate
                        volume_source = 'existing_airmorph_parse'
                    else:
                        print('[WARNING] parse labels do not match branch IDs; using nearest skeleton.')
                except Exception as error:
                    print(f'[WARNING] Existing parse could not be used: {error}')
            if instance_volume is None:
                instance_volume = reconstruct_instance_volume(data.airway_mask, repaired)
                volume_source = 'nearest_repaired_skeleton_within_airway_bin'
            beyond_volume = lookup[instance_volume]
            save_nifti(beyond_volume, data.reference_image, self.generation_output_dir / f'{case_id}_beyond_instance_volume.nii.gz', np.int16)
            for beyond_label, coordinates in sorted(coordinates_by_label(beyond_volume).items()):
                component_count = component_count_for_coordinates(coordinates)
                volume_qc_rows.append({'beyond_numeric_label': beyond_label, 'volume_voxels': int(len(coordinates)), 'connected_components_26': component_count, 'largest_component_ratio': 1.0 if component_count == 1 else float('nan')})
        mapping_csv = self.generation_output_dir / f'{case_id}_instance_generation_mapping.csv'
        mapping.to_csv(mapping_csv, index=False)
        mapping_json = self.generation_output_dir / f'{case_id}_beyond_label_list.json'
        json_payload: dict[str, Any] = {}
        for _, row in mapping.iterrows():
            children_text = str(row['children_branch_instance_ids'])
            child_ids = [int(value) for value in children_text.split(';') if value and value.lower() != 'nan']
            json_payload[str(int(row['beyond_numeric_label']))] = {'name': row['proposed_beyond_name'], 'branch_instance_id': int(row['branch_instance_id']), 'parent_branch_instance_id': int(row['parent_branch_instance_id']), 'children_branch_instance_ids': child_ids, 'absolute_generation': int(row['absolute_generation']), 'airmorph_class_label': int(row['airmorph_class_label']), 'airmorph_anatomical_name': row['airmorph_anatomical_name'], 'generation_below_airmorph_class': int(row['generation_below_airmorph_class']), 'hierarchical_path_within_class': row['hierarchical_path_within_class']}
        mapping_json.write_text(json.dumps(json_payload, indent=2, ensure_ascii=False), encoding='utf-8')
        pd.DataFrame(tree_edges, columns=['parent_branch_instance_id', 'child_branch_instance_id', 'via']).to_csv(self.generation_output_dir / f'{case_id}_parent_child_edges.csv', index=False)
        junctions.to_csv(self.generation_output_dir / f'{case_id}_junction_components.csv', index=False)
        if volume_qc_rows:
            pd.DataFrame(volume_qc_rows).to_csv(self.generation_output_dir / f'{case_id}_beyond_volume_qc.csv', index=False)
        branch_nodes = {f'B{branch_id}' for branch_id in branch_ids}
        reached_branches = {node for node in visited if node.startswith('B')}
        graph_node_count = len(graph)
        graph_edge_count = sum((len(neighbours) for neighbours in graph.values())) // 2
        graph_cycle_rank = graph_edge_count - graph_node_count + len(graph_components)
        disconnected_repaired_labels = 0
        for branch_id in branch_ids:
            coordinates = repaired_coordinates.get(branch_id, np.empty((0, 3), dtype=np.int32))
            if component_count_for_coordinates(coordinates) > 1:
                disconnected_repaired_labels += 1
        qc = {'case_id': case_id, 'shape': 'x'.join(map(str, data.reference_image.shape)), 'orientation': ''.join(nib.aff2axcodes(data.reference_image.affine)), 'branch_instance_count': len(branch_ids), 'skeleton_voxels': skeleton_voxels, 'assigned_before': assigned_before, 'unassigned_before': skeleton_voxels - assigned_before, 'coverage_before': coverage_before, 'assigned_after': assigned_after, 'unassigned_after': skeleton_voxels - assigned_after, 'coverage_after': coverage_after, 'junction_component_count': len(junctions), 'junction_components_touching_3plus_branches': int(np.sum(junctions.get('touching_branch_count', pd.Series(dtype=int)) >= 3)) if not junctions.empty else 0, 'root_branch_instance_id': root_label, 'root_selection_method': root_method, 'topology_graph_nodes': graph_node_count, 'topology_graph_edges': graph_edge_count, 'topology_graph_components': len(graph_components), 'topology_graph_cycle_rank': graph_cycle_rank, 'reached_branch_instances': len(reached_branches), 'unreached_branch_instances': len(branch_nodes - reached_branches), 'maximum_absolute_generation': max(generation.values()) if generation else -1, 'repaired_labels_with_multiple_components': disconnected_repaired_labels, 'airway_mask_voxels': int(data.airway_mask.sum()), 'reconstructed_volume': bool(self.reconstruct_volume), 'volume_reconstruction_source': volume_source, 'anno_json_used': str(self.files.anno_json) if self.files.anno_json else 'missing', 'class2anno_used': str(self.files.class2anno_json) if self.files.class2anno_json else 'missing', 'airway_graph_available': bool(self.files.airway_graph_npy), 'airway_graph_cls_available': bool(self.files.airway_graph_cls_npy), 'instance_parse_available': bool(self.files.instance_volume_parse)}
        pd.DataFrame([qc]).to_csv(self.generation_output_dir / f'{case_id}_generation_repair_qc.csv', index=False)
        color_table = self.generation_output_dir / f'{case_id}_beyond_labels.ctbl'
        with color_table.open('w', encoding='utf-8') as file:
            file.write('# Color table generated by airmorph_beyond_subsegment_pipeline.py\n')
            file.write('0 Background 0 0 0 0\n')
            for _, row in mapping.iterrows():
                label_value = int(row['beyond_numeric_label'])
                red = (53 * label_value + 67) % 256
                green = (97 * label_value + 29) % 256
                blue = (193 * label_value + 11) % 256
                name = str(row['proposed_beyond_name']).replace(' ', '_')
                file.write(f'{label_value} {name} {red} {green} {blue} 255\n')
        report_path = self.generation_output_dir / f'{case_id}_generation_repair_report.txt'
        with report_path.open('w', encoding='utf-8') as file:
            file.write(f'Case: {case_id}\n')
            file.write('Method: branch measurement, topology-preserving junction repair, generation, and beyond-label proposal\n\n')
            file.write('INPUTS\n------\n')
            file.write(f'Case directory: {self.files.case_dir}\n')
            file.write(f'Airway mask: {self.files.airway_mask}\n')
            file.write(f'Original skeleton: {self.files.airway_skeleton}\n')
            file.write(f'Skeleton parsing: {self.files.skeleton_parsing}\n')
            file.write(f'Instance parse: {self.files.instance_volume_parse}\n\n')
            file.write('QUALITY CONTROL\n---------------\n')
            for key, value in qc.items():
                file.write(f'{key}: {value}\n')
            file.write('\nINTERPRETATION\n--------------\n')
            file.write('Original positive AirMorph branch labels are preserved. Zero-valued skeleton voxels are filled by shortest-path propagation constrained to the original skeleton.\n')
            file.write('The topology graph is constructed with explicit junction nodes before junction ownership is assigned.\n')
            file.write('Beyond-subsegment names are topology-based candidate names and require anatomical review.\n')
            if graph_cycle_rank > 0:
                file.write('WARNING: The topology graph contains cycles or redundant contacts. The exported parent-child relation is a rooted spanning tree.\n')
            if branch_nodes - reached_branches:
                file.write('WARNING: Some branch instances were not reached from the selected root.\n')
        print(f'[DONE] {case_id}')
        print('  Coverage before    :', f'{coverage_before:.2%}')
        print('  Coverage after     :', f'{coverage_after:.2%}')
        print('  Branch instances   :', len(branch_ids))
        print('  Root branch        :', root_label, f'({root_method})')
        print('  Reached branches   :', len(reached_branches), '/', len(branch_ids))
        print('  Maximum generation :', qc['maximum_absolute_generation'])
        print('  Graph cycle rank   :', graph_cycle_rank)
        print('  Output directory   :', self.generation_output_dir)
        del repaired, beyond_skeleton
        gc.collect()
        return mapping

    def run_full(self) -> None:
        self.run_analysis()
        self.run_generation()

# =============================================================================
# Batch control and logs
# =============================================================================

def run_task_for_cases(task: str, case_ids: list[str], overwrite: bool, root_override: int | None=None, reconstruct_volume: bool=True, save_distance: bool=False) -> None:
    successes: list[str] = []
    failures: list[tuple[str, str]] = []
    for case_id in case_ids:
        try:
            pipeline = AirMorphCasePipeline(case_id=case_id, overwrite=overwrite, root_override=root_override if len(case_ids) == 1 else None, reconstruct_volume=reconstruct_volume, save_distance=save_distance)
            if task == 'analysis':
                pipeline.run_analysis()
            elif task == 'generation':
                pipeline.run_generation()
            elif task == 'full':
                pipeline.run_full()
            else:
                raise ValueError(f'Unknown task: {task}')
            successes.append(case_id)
        except Exception as error:
            failures.append((case_id, str(error)))
            print(f'[ERROR] {case_id}: {error}')
            traceback.print_exc()
    log_rows = [{'case_id': case_id, 'task': task, 'status': 'success', 'output_root': str(get_case_dir(case_id) / OUTPUT_ROOT_NAME), 'error': ''} for case_id in successes] + [{'case_id': case_id, 'task': task, 'status': 'failed', 'output_root': str(get_case_dir(case_id) / OUTPUT_ROOT_NAME), 'error': error} for case_id, error in failures]
    if log_rows:
        log_dir = PROJECT_ROOT / LOG_FOLDER_NAME
        log_dir.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(log_rows).to_csv(log_dir / f'latest_{task}_processing_log.csv', index=False)
    print('\n' + '=' * 80)
    print('PROCESSING FINISHED')
    print('=' * 80)
    print('Task             :', task)
    print('Successful cases :', len(successes))
    print('Failed cases     :', len(failures))
    if failures:
        print('Failed IDs       :', ' '.join((case_id for case_id, _ in failures)))
# =============================================================================
# Interactive menu
# =============================================================================

MENU = {'1': {'task': 'analysis', 'name': 'Original AirMorph branch-instance analysis', 'description': 'Measure branch geometry, diameter, volume mapping, and QC.'}, '2': {'task': 'generation', 'name': 'Generation repair and beyond-subsegment proposal', 'description': 'Repair junctions, calculate parent-child generation, and create beyond labels.'}, '3': {'task': 'full', 'name': 'Full workflow', 'description': 'Run branch analysis and generation repair in sequence.'}}

def print_main_menu() -> None:
    print()
    print('=' * 80)
    print('AirMorph Beyond-Subsegment Pipeline')
    print('=' * 80)
    print('Please select a function:\n')
    for key, item in MENU.items():
        print(f"{key}. {item['name']}")
        print(f"   {item['description']}")
    print('0. Exit\n')

def print_mode_menu() -> None:
    print()
    print('-' * 80)
    print('Please select processing mode:')
    print('-' * 80)
    print('1. Batch process all cases')
    print('2. Process one selected case')
    print('3. Process multiple selected cases')
    print('0. Back\n')

def select_cases_by_mode() -> list[str] | None:
    while True:
        print_mode_menu()
        mode = input('Enter mode number: ').strip()
        if mode == '0':
            return None
        if mode == '1':
            case_ids = get_default_case_ids()
            print(f'Batch processing will run on {len(case_ids)} case(s).')
            if case_ids:
                print(f'Case range: {case_ids[0]} - {case_ids[-1]}')
            confirm = input('Confirm? Enter y to continue: ').strip().lower()
            if confirm == 'y':
                return case_ids
            print('[CANCEL] Batch processing cancelled.')
            continue
        if mode == '2':
            raw_case = input('Enter one case ID, for example 001 or 4: ').strip()
            if not raw_case:
                print('[ERROR] Case ID cannot be empty.')
                continue
            case_id = normalize_case_id(raw_case)
            if not get_case_dir(case_id).exists():
                print(f'[WARNING] Case folder not found: {get_case_dir(case_id)}')
                continue
            return [case_id]
        if mode == '3':
            print('Supported formats: 001 002 003 / 001,002,003 / 001-010 / 001 004 034-059')
            raw_text = input('Enter case IDs: ').strip()
            if not raw_text:
                print('[ERROR] Input cannot be empty.')
                continue
            try:
                case_ids = parse_multiple_cases(raw_text)
            except Exception as error:
                print(f'[ERROR] Failed to parse case IDs: {error}')
                continue
            if not case_ids:
                print('[ERROR] No valid case ID was parsed.')
                continue
            missing = [case_id for case_id in case_ids if not get_case_dir(case_id).exists()]
            if missing:
                print('[WARNING] Missing case folders:', ' '.join(missing))
                confirm = input('Continue with existing cases? Enter y: ').strip().lower()
                if confirm != 'y':
                    continue
                case_ids = [case_id for case_id in case_ids if case_id not in missing]
            return case_ids
        print('[ERROR] Invalid mode number.')

def run_interactive() -> None:
    while True:
        print_main_menu()
        choice = input('Enter function number: ').strip()
        if choice == '0':
            print('[EXIT] Exited.')
            return
        if choice not in MENU:
            print('[ERROR] Invalid function number.')
            continue
        item = MENU[choice]
        print(f"Selected: {item['name']}")
        print(f"Description: {item['description']}")
        case_ids = select_cases_by_mode()
        if case_ids is None:
            print('[BACK] Returning to function selection.')
            continue
        overwrite = input('Overwrite existing outputs? Enter y, or press Enter: ').strip().lower() == 'y'
        root_override: int | None = None
        if item['task'] in {'generation', 'full'} and len(case_ids) == 1:
            raw_root = input('Optional root branch label (press Enter for automatic selection): ').strip()
            if raw_root:
                try:
                    root_override = int(raw_root)
                except ValueError:
                    print('[WARNING] Invalid root label; automatic selection will be used.')
        reconstruct_volume = True
        if item['task'] in {'generation', 'full'}:
            skip_volume = input('Skip 3D beyond-volume output? Enter y to skip: ').strip().lower()
            reconstruct_volume = skip_volume != 'y'
        run_task_for_cases(task=item['task'], case_ids=case_ids, overwrite=overwrite, root_override=root_override, reconstruct_volume=reconstruct_volume)
        next_action = input('Select another function? Enter y to continue: ').strip().lower()
        if next_action != 'y':
            print('[EXIT] Exited.')
            return

# =============================================================================
# Command-line interface
# =============================================================================

def parse_cli_cases(arguments: argparse.Namespace) -> list[str]:
    if arguments.all:
        if arguments.cases:
            raise ValueError('Do not combine --all with --cases.')
        return get_default_case_ids()
    if not arguments.cases:
        raise ValueError('Use --cases or --all, or run without arguments for the menu.')
    case_ids: list[str] = []
    for token in arguments.cases:
        case_ids.extend(expand_case_token(token))
    return parse_multiple_cases(' '.join(case_ids))

def main() -> None:
    if not PROJECT_ROOT.exists():
        raise FileNotFoundError(f'PROJECT_ROOT not found: {PROJECT_ROOT}')
    parser = argparse.ArgumentParser(description='AirMorph branch analysis, junction repair, generation, and beyond-subsegment proposal.')
    parser.add_argument('--task', choices=['analysis', 'generation', 'full'], default=None, help='Task to run. Omit all arguments for the interactive menu.')
    parser.add_argument('--cases', nargs='*', default=None, help='Case IDs or ranges, for example --cases 001 003-010')
    parser.add_argument('--all', action='store_true', help='Process all numeric case folders.')
    parser.add_argument('--overwrite', action='store_true', help='Overwrite existing outputs.')
    parser.add_argument('--root-label', type=int, default=None, help='Optional root branch override. Applied only to one case.')
    parser.add_argument('--skip-volume', action='store_true', help='Skip 3D beyond-volume reconstruction.')
    parser.add_argument('--save-distance', action='store_true', help='Save repair-distance NIfTI.')
    arguments = parser.parse_args()
    if arguments.task is None and (not arguments.cases) and (not arguments.all):
        run_interactive()
        return
    if arguments.task is None:
        raise ValueError('--task is required in command-line mode.')
    case_ids = parse_cli_cases(arguments)
    run_task_for_cases(task=arguments.task, case_ids=case_ids, overwrite=arguments.overwrite, root_override=arguments.root_label, reconstruct_volume=not arguments.skip_volume, save_distance=arguments.save_distance)
if __name__ == '__main__':
    main()
