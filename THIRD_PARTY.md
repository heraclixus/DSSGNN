# Third-party code and data

This repository vendors or adapts the following code. Each component keeps
its own license where one was distributed.

| Path | Origin | License |
|---|---|---|
| `GOOD_clean/` | GOOD: A Graph Out-of-Distribution Benchmark (Gui et al., NeurIPS 2022 Datasets and Benchmarks), https://github.com/divelab/GOOD | GPL-3.0 (`GOOD_clean/LICENSE`) |
| `external/graph-ebm/` | Graph Energy-based Model reference implementation (Fuchsgruber et al., NeurIPS 2024) | MIT (`external/graph-ebm/LICENSE`) |
| `gnnsafe_ood/graph_ebm.py` | Port of the Graph-EBM scorer into the GNNSafe-style pipeline | MIT, as above |
| `gnnsafe_ood/` | Ported from GNNSafe (Wu et al., ICLR 2023), https://github.com/qitianwu/GraphOOD-GNNSafe | See upstream |
| `gduq/` | G-DeltaUQ node and graph classification code (Trivedi et al., ICLR 2024, arXiv:2401.03350), adapted | See upstream |
| `tfe_utils.py`, `tfe_models.py`, `tfe_load_data.py`, `tfe_training.py`, `tfe_traing_3.py`, `load_geom.py`, `run_baseline.py` | TFE-GNN (Duan et al., 2024) training and data-loading code on which this repository was built | See upstream |
| `splits/`, `new_data/` | geom-gcn splits and raw WebKB/Wikipedia graphs (Pei et al., ICLR 2020), https://github.com/graphdml-uiuc-jlu/geom-gcn | See upstream |
| `new_data2/splits/` | LINKX fixed splits (Lim et al., NeurIPS 2021), https://github.com/CUAI/Non-Homophily-Large-Scale | See upstream |

The heterophilous benchmark graphs (Platonov et al., ICLR 2023) are
downloaded separately from https://github.com/yandex-research/heterophilous-graphs.
