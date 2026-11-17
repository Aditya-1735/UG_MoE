"""
Data pipeline for the Eadro multimodal microservice datasets (SN = SocialNetwork, TT = TrainTicket),
Zenodo record 10.5281/zenodo.7615394.

    python -m data_pipeline.fetch_eadro                 # download + verify + extract SN and TT
    python -m data_pipeline.build_eadro --system SN     # raw runs -> windowed chunks + metadata + quality report

Outputs go to project/datasets/eadro/{raw,processed}/ (gitignored). The processed directory is
consumed by experiments/common.py (load_splits detects the chunks.pkl format and splits by run).
"""
