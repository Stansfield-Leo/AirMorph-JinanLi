# AirMorph-JinanLi
Branch-Instance Analysis and Topology Reconstruction of CT-Derived Airway Trees

Overview

This repository contains the implementation developed for the MSc Final Year Project:

Branch-Instance Analysis, Topology Reconstruction and Beyond-Subsegment Labeling of CT-Derived Airway Trees

The project extends AirMorph outputs through a topology-aware downstream pipeline. It focuses on two connected tasks:

Structure-oriented airway evaluation using skeleton- and graph-based measurements.

Branch-level topology reconstruction with junction recovery, hierarchy assignment and branch-specific 3D labels.

The proposed framework does not replace AirMorph. AirMorph provides the upstream airway segmentation, skeleton and anatomical labels, while this project adds structural evaluation and finer branch-level organisation.

Prerequisite: The complete original AirMorph repository is required to run this project, including all source code, configuration files, pretrained models, dependencies and auxiliary resources provided with that repository.

Main Contributions

Structure-oriented evaluation of airway segmentation outputs.

Analysis of continuous branch instances and propagated anatomical volume labels.

Recovery of junction voxels removed during branch parsing.

Reconstruction of a connected airway branch–junction graph.

Assignment of root direction, parent–child relationships and airway generations.

Generation of one computational 3D identifier per continuous branch.

Preservation of the original AirMorph anatomical class for each branch.

Export of reusable case-level reports and branch-mapping tables.

In this project, beyond-subsegment labels are computational branch-specific identifiers. They are not proposed as a clinically validated anatomical nomenclature.

Processing Workflow

AirMorph outputs
    │
    ├── Binary airway mask
    ├── Airway skeleton
    ├── Branch-instance labels
    └── Propagated anatomical labels
    │
    ▼
Structure-oriented evaluation
    ├── Connected components
    ├── Terminal branches
    ├── Skeleton length
    ├── Maximum branch depth
    ├── Peripheral reach
    └── Local fragmentation
    │
    ▼
Branch-instance audit
    │
    ▼
Junction recovery
    │
    ▼
Branch–junction graph reconstruction
    │
    ▼
Root selection and hierarchy assignment
    │
    ▼
Beyond-subsegment label generation
    │
    ▼
CSV reports and optional 3D NIfTI outputs

Input Files

Each case directory should contain the required AirMorph outputs.

Case_001/
├── airway_bin.nii.gz
├── airway_skeleton.nii.gz
├── 001_skel_parsing.nii.gz
└── 001_pred_sub.nii.gz

Input description

File

Description

airway_bin.nii.gz

Binary airway segmentation

airway_skeleton.nii.gz

Complete airway skeleton

001_skel_parsing.nii.gz

Continuous branch-instance labels

001_pred_sub.nii.gz

Propagated AirMorph anatomical labels

All volumes must have compatible dimensions and spatial alignment.

Output Files

Typical outputs include:

Case_001/
├── branch_analysis.csv
├── generation_repair.csv
├── beyond_subsegment_mapping.csv
└── beyond_subsegment_labels.nii.gz

Main outputs

Output

Description

branch_analysis.csv

Branch geometry, quality-control and volume-label mapping

generation_repair.csv

Reconstructed connectivity, parent–child relationships and generations

beyond_subsegment_mapping.csv

Mapping between branch IDs, anatomical classes and new branch-specific labels

beyond_subsegment_labels.nii.gz

Optional branch-specific 3D airway label volume

The final mapping table may contain:

Branch-instance ID

Original anatomical class

Anatomical name

Beyond-subsegment label

Parent branch

Child branches

Generation

Connected junction

Environment

The project was developed in a Linux-based environment using WSL and Conda.

Recommended software

Ubuntu 20.04 or later

Python 3.x

Conda or Miniconda

Python dependencies

NumPy

SciPy

NiBabel

NetworkX

Pandas

Matplotlib

Create and activate an environment:

conda create -n airwayatlas python=3.9 -y
conda activate airwayatlas
pip install numpy scipy nibabel networkx pandas matplotlib

Usage

1. Structure-oriented airway analysis

Run:

python airway_analysis_system.py

The interactive system supports:

Single-case processing

Selected multi-case processing

Batch processing

Connected-component analysis

Short-path analysis

Terminal-branch analysis

Skeleton-length analysis

Peripheral reach and branch-depth analysis

2. Beyond-subsegment pipeline

Run:

python airmorph_beyond_subsegment_pipeline.py

Available modes may include:

Original AirMorph branch-instance analysis

Junction recovery, hierarchy assignment and beyond-subsegment generation

Complete pipeline

The user may also choose whether to overwrite existing outputs and whether to generate the optional 3D branch-label volume.

Structural Measurements

The framework uses complementary skeleton- and graph-based measurements.

Skeleton coverage

Skeleton coverage =
labelled skeleton voxels / total skeleton voxels × 100%

Example structural characteristics

Connected-component count

Terminal-branch count

Total skeleton length

Maximum branch depth

Peripheral extension

Short-path burden

Multi-path anatomical labels

Disconnected propagated labels

Graph reachability

Cycle rank

These measurements complement conventional region-overlap evaluation by describing continuity, topology and distal preservation.

Graph Representation

The reconstructed graph uses two types of nodes:

Branch nodes: continuous branch instances

Junction nodes: recovered junction regions

An edge represents direct contact between a branch instance and a junction region.

After root selection, the undirected graph is traversed to assign:

Parent branch

Child branches

Airway generation

Proximal-to-distal direction

A connected and acyclic graph confirms internal graph consistency, but it does not by itself prove anatomical correctness.

Example Case Results

For the representative case used in the project:

121 branch instances were identified.

113 branches passed the applied quality-control criteria.

8 short paths were flagged.

Skeleton coverage increased from 94.86% to 100.00% after junction recovery.

112 previously unlabelled skeleton voxels were incorporated.

The reconstructed graph contained 179 nodes and 178 edges.

All 121 branch instances were reachable.

The graph contained one connected component.

The cycle rank was 0.

These results demonstrate successful topology reconstruction relative to the input skeleton.

Quality-Control Notes

The current implementation includes exploratory quality-control thresholds for:

Minimum branch voxel count

Minimum physical path length

Label purity

Skeleton coverage

Diameter sample count

These thresholds are intended for computational quality control and should not be interpreted as clinically validated cut-offs.

Limitations

The detailed demonstration was primarily conducted on one example case.

Results depend on the quality of the upstream segmentation and skeleton.

Missing airway branches cannot be reconstructed if they are absent from the input.

False-positive branches may remain in the reconstructed graph.

Root selection affects parent–child direction and generation assignment.

Complex junctions and anatomical names require expert validation.

Beyond-subsegment labels are computational identifiers rather than clinical anatomical names.

Repository Structure

A recommended repository layout is:

.
├── airway_analysis_system.py
├── airmorph_beyond_subsegment_pipeline.py
├── src/
│   ├── io_utils.py
│   ├── skeleton_analysis.py
│   ├── branch_analysis.py
│   ├── junction_recovery.py
│   ├── graph_reconstruction.py
│   ├── hierarchy_assignment.py
│   └── label_propagation.py
├── data/
│   └── Case_001/
├── outputs/
├── figures/
├── requirements.txt
└── README.md

Adjust the structure to match the actual repository.

Data Availability

Patient CT data and AirMorph-derived case outputs are not included in this repository. Users should provide their own appropriately authorised data and corresponding AirMorph outputs.

Acknowledgements

This project was completed as part of the MSc programme in Medical Robotics and Artificial Intelligence at University College London.

Author: Jinan LiFirst Supervisor: Dr Tianqi YangSecond Supervisor: Prof Joseph Jacob

Citation

When using this repository, please cite the associated MSc dissertation:

J. Li, “Branch-Instance Analysis, Topology Reconstruction and
Beyond-Subsegment Labeling of CT-Derived Airway Trees,”
MSc dissertation, University College London, 2026.

Licence

Add the appropriate licence before public release. The repository may depend on external data, code or model licences associated with AirMorph and the source datasets.
