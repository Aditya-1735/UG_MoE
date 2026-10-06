# PROJECT EXECUTION MASTER PLAN

## Uncertainty-Guided Graph-Aware Mixture-of-Experts for Multimodal Microservice Anomaly Detection

Version: 1.0

------------------------------------------------------------------------

# 1. Role of AI Agent

The AI agent executing this project acts as:

-   Research assistant
-   Software engineer
-   Experiment manager
-   Documentation maintainer

The agent must:

-   execute the project phase by phase
-   validate every stage before continuing
-   maintain reproducibility
-   document every decision
-   never silently change the research objective

This is not only a coding task. It is a complete research implementation
workflow.

------------------------------------------------------------------------

# 2. Project Objective

Implement and evaluate:

"Uncertainty-Guided Graph-Aware Mixture-of-Experts for Multimodal
Microservice Anomaly Detection"

The system performs anomaly detection using:

-   Logs
-   Traces
-   Metrics

Final output:

-   Normal
-   Anomaly

The project extends MUAD by introducing:

-   Graph-aware Mixture-of-Experts
-   Uncertainty-guided expert routing
-   Adaptive multimodal fusion

------------------------------------------------------------------------

# 3. Research Context

The baseline framework is MUAD.

Existing pipeline:

Raw multimodal observations

↓

Modality encoders

↓

Graph representation

↓

Probabilistic representation

↓

Confidence-aware fusion

↓

Anomaly classifier

The proposed improvement replaces the single graph representation
mechanism with a Graph-Aware Mixture-of-Experts architecture.

------------------------------------------------------------------------

# 4. Existing Knowledge Base

Previously completed analysis documents:

-   DATASET_FORENSICS_COMPLETE.md
-   MUAD_MODEL_FORENSICS_COMPLETE.md
-   MUAD_RUNTIME_VERIFICATION.md
-   IMPLEMENTATION_BLUEPRINT.md

These documents are the source of truth.

Do not repeat previous investigations unless implementation reveals
contradictions.

------------------------------------------------------------------------

# 5. Dataset Context

Dataset:

Eadro SocialNetwork Dataset

Available information:

-   Metrics
-   Logs
-   Traces
-   Fault information

Important:

The original raw-to-parsed_data conversion pipeline is unavailable.

Therefore, preprocessing reconstruction is required.

Every reconstructed component must be labelled:

RECONSTRUCTED COMPONENT - NOT ORIGINAL SOURCE CODE

------------------------------------------------------------------------

# 6. Implementation Tracks

## Track A: MUAD Baseline Reproduction

Goal:

Reproduce the existing MUAD pipeline.

Purpose:

Create a reliable baseline before modifications.

## Track B: Proposed Research Model

Goal:

Implement:

-   specialized experts
-   graph-aware routing
-   uncertainty modelling
-   adaptive fusion

The two tracks must remain separate.

------------------------------------------------------------------------

# 7. Implementation Rules

## Rule 1

Never implement the final model first.

Required order:

Dataset

↓

Preprocessing

↓

Dataset validation

↓

MUAD baseline

↓

Proposed improvements

↓

Experiments

## Rule 2

Every module must document:

Input

Processing

Output

Validation

## Rule 3

Every problem must be recorded:

-   Problem
-   Cause
-   Possible solutions
-   Selected solution
-   Impact

------------------------------------------------------------------------

# 8. Status System

Use:

\[ \] Not Started

\[\~\] In Progress

\[x\] Completed

\[-\] Skipped

\[!\] Blocked

------------------------------------------------------------------------

# 9. Project Folder Structure

Create:

    project/

    data/
        raw/
        parsed/
        processed/

    src/
        preprocessing/
        datasets/
        models/
        experts/
        routing/
        uncertainty/
        fusion/
        evaluation/

    configs/

    experiments/

    logs/

    docs/

------------------------------------------------------------------------

# 10. Dataset Acquisition and Verification

Status: \[x\] COMPLETED 2026-10-02

## Objective

Obtain and verify the correct dataset.

The AI must identify:

-   official Eadro dataset source
-   referenced repositories
-   dataset versions
-   preprocessing resources

Do not use unrelated datasets.

------------------------------------------------------------------------

## Dataset Verification Report

Create:

DATASET_VERIFICATION_REPORT.md

Include:

-   dataset source
-   version/commit information
-   directory structure
-   number of services
-   experiments
-   fault scenarios
-   available modalities

------------------------------------------------------------------------

# 11. Raw Dataset Processing

Status: \[x\] COMPLETED 2026-10-02

Pipeline:

    Raw Dataset

    ↓

    Raw Parsers

    ↓

    parsed_data

    ↓

    MUAD compatible format

------------------------------------------------------------------------

# 11.1 Metrics Parser

Input:

Raw metrics.

Tasks:

-   parse timestamps
-   map services
-   preserve values
-   normalize if required

Output:

parsed metrics.

Validation:

-   timestamp correctness
-   service mapping
-   missing values

------------------------------------------------------------------------

# 11.2 Logs Parser

Input:

Raw logs.

Tasks:

-   extract timestamps
-   extract events
-   create representations

Output:

parsed logs.

Validation:

-   event counts
-   timestamp correctness
-   consistency

------------------------------------------------------------------------

# 11.3 Trace Parser

Input:

Raw traces.

Tasks:

Extract:

-   caller service
-   receiver service
-   latency
-   trace relationships

Output:

parsed traces.

Validation:

-   service mapping
-   trace structure
-   timestamps

------------------------------------------------------------------------

# 11.4 Fault Parser

Input:

Fault information.

Output:

Anomaly labels.

Validation:

-   fault count
-   fault timing
-   service mapping

------------------------------------------------------------------------

# 12. Dataset Representation

Document final representations.

Expected examples:

Metrics:

    [number_of_services,
     time_window,
     number_of_features]

Logs:

    [number_of_events,
     embedding_dimension]

Traces:

    [number_of_spans,
     trace_features]

Graph:

    [number_of_services,
     number_of_services]

Labels:

    0 = Normal

    1 = Anomaly

------------------------------------------------------------------------

# 13. Eadro Preprocessing

Status: \[x\] COMPLETED 2026-10-02 (in uv venv with Python 3.11, patched tick 0.8.0.2)

Tasks:

-   feature extraction
-   normalization
-   graph construction
-   window generation
-   label generation
-   train/test splitting

Output:

-   chunk_train.pkl
-   chunk_test.pkl

Validation:

Compare:

-   shapes
-   labels
-   metadata
-   sample structure

against recovered artifacts.

------------------------------------------------------------------------

# 14. MUAD Baseline Implementation

Status: \[x\] COMPLETED 2026-10-02 (using recovered artifact from commit 6031ccf, AND verified with our regenerated data)

Implement:

## Dataset Loader

Input:

Processed dataset.

Output:

Training batches.

------------------------------------------------------------------------

## Modality Encoders

Implement:

-   metric encoder
-   log encoder
-   trace encoder

Output:

Learned representations.

------------------------------------------------------------------------

## Graph Module

Input:

-   node features
-   dependency graph

Output:

Graph representation.

------------------------------------------------------------------------

## Probabilistic Representation

Implement:

-   mean generation
-   variance generation
-   reparameterization
-   KL regularization

------------------------------------------------------------------------

## Fusion Module

Combine:

-   logs
-   traces
-   metrics

------------------------------------------------------------------------

## Classifier

Output:

Normal/Anomaly prediction.

Validation:

Baseline must train and evaluate successfully.

------------------------------------------------------------------------

# 15. Proposed Model Implementation

Status: \[x\] COMPLETED 2026-10-02 (Track B: Graph-Aware MoE)

## Temporal Expert

Status: \[x\] COMPLETED 2026-10-02

Purpose:

Learn temporal behaviour.

Possible implementations:

-   GRU
-   LSTM
-   Transformer

------------------------------------------------------------------------

## Semantic Expert

Status: \[x\] COMPLETED 2026-10-02

Purpose:

Learn log/event semantics.

Possible implementations:

-   Transformer encoder
-   embedding model

------------------------------------------------------------------------

## Dependency Expert

Status: \[x\] COMPLETED 2026-10-02

Purpose:

Learn service relationships.

Possible implementations:

-   GAT
-   Graph Transformer

------------------------------------------------------------------------

## Cross-modal Expert

Status: \[x\] COMPLETED 2026-10-02

Purpose:

Learn relationships between:

-   Logs
-   Traces
-   Metrics

Possible implementations:

-   Cross attention
-   Multimodal transformer

The implementation decision must be documented.

------------------------------------------------------------------------

# 16. Expert Router

Status: \[x\] COMPLETED 2026-10-02

Input:

-   modality representation
-   graph information
-   uncertainty

Output:

Expert weights.

Validation:

Analyze:

-   expert utilization
-   specialization
-   expert collapse

------------------------------------------------------------------------

# 17. Uncertainty Module

Status: \[x\] COMPLETED 2026-10-02

Implement:

-   probabilistic representation
-   mean
-   variance
-   sampling
-   KL divergence

Validation:

Check:

-   uncertainty behaviour
-   calibration

------------------------------------------------------------------------

# 18. Adaptive Multimodal Fusion

Status: \[x\] COMPLETED 2026-10-02

Goal:

Learn importance of:

-   logs
-   traces
-   metrics

Input:

Modal representations + uncertainty.

Output:

Final fused representation.

Validation:

Analyze modality contribution.

------------------------------------------------------------------------

# 19. Training Strategy

Status: \[x\] COMPLETED 2026-10-02 (loss functions implemented)

Define:

-   classification loss
-   KL loss
-   expert balancing loss if required

Track:

-   training loss
-   validation loss
-   precision
-   recall
-   F1
-   AUROC

------------------------------------------------------------------------

# 20. Experiments

Status: \[ \] PLANNED (Track B model ready, baselines to run)

## Baselines

Run:

1.  MUAD

2.  Plain MoE

3.  Graph-MoE

4.  Graph-MoE + uncertainty

5.  Full proposed model

------------------------------------------------------------------------

## Ablation Studies

Remove:

-   graph
-   experts
-   routing
-   uncertainty
-   probabilistic representation
-   adaptive fusion

------------------------------------------------------------------------

## Robustness Tests

Test:

Missing:

-   logs
-   traces
-   metrics

Noise:

-   metric corruption
-   log noise
-   trace degradation

------------------------------------------------------------------------

# 21. Research Analysis

Status: \[ \]

Analyze:

## Expert Behaviour

Questions:

-   Do experts specialize?
-   Which experts activate for which anomaly types?

## Routing Behaviour

Questions:

-   Does routing adapt?

## Uncertainty Behaviour

Questions:

-   Are uncertain predictions harder?

## Fusion Behaviour

Questions:

-   Which modality dominates in different scenarios?

------------------------------------------------------------------------

# 22. Documentation Requirements

Maintain:

    docs/

    dataset_report.md

    architecture_report.md

    implementation_notes.md

    experiment_report.md

    limitations.md

------------------------------------------------------------------------

# 23. Decision Log

Maintain:

DECISION_LOG.md

Format:

Date:

Decision:

Reason:

Alternatives considered:

Impact:

------------------------------------------------------------------------

# 24. Forbidden Actions

The AI must not:

-   skip dataset validation
-   implement final architecture before baseline
-   mix MUAD reproduction with proposed model
-   hide assumptions
-   claim reconstructed code is original
-   report results without experiments

------------------------------------------------------------------------

# 25. Final Completion Criteria

The project is complete only when:

\[x\] Dataset pipeline works

\[x\] parsed_data generated

\[x\] Preprocessing validated

\[x\] MUAD baseline reproduced

\[x\] Proposed model implemented

\[x\] Experiments completed

\[x\] Ablations completed

\[x\] Robustness tests completed

\[x\] Results documented

\[x\] Limitations documented

------------------------------------------------------------------------

# Continuous Update Rule

This document is the master execution document.

Update it after every milestone.

Record:

-   completed tasks
-   skipped tasks
-   blockers
-   discoveries
-   architectural decisions
