# DSS-GNN Variants Sweep

Comprehensive sweep over Table 1 datasets and all DSS-GNN epistemic uncertainty variants.

## Variants (from method.tex)

| Variant | Description |
|---------|-------------|
| `dssgnn` | S: Base DSS-GNN (best config per dataset) |
| `dssgnn_sg` | S+G: Random branch gates α_n(ω), β_n(ω) |
| `dssgnn_sf` | S+F: Random filter coefficients c_k(ω) |
| `dssgnn_gduq` | S+A: GDUQ-style input anchoring |
| `dssgnn_gduq_sg` | S+G+A: Anchoring + random gates |
| `dssgnn_gduq_sf` | S+F+A: Anchoring + random filter |

## Datasets (16, from Table 1)

Cora, Citeseer, PubMed, Texas, Cornell, Wisconsin, Chameleon, Squirrel, Cora-Full, CS, Physics, Roman-Empire, Amazon-Ratings, Minesweeper, Tolokers, Questions.

## Usage

### Local test (single task)
```bash
python scripts/run_dssgnn_variants_sweep.py --dataset cora --variant dssgnn_sg --runs 2
./scripts/run_dssgnn_variants_local.sh cora dssgnn_sg
```

### Full sweep
Run `run_dssgnn_variants_sweep.py` once per (dataset, variant) pair, for
example as a job array on your cluster.

### Parse results
```bash
# Print summary
python scripts/parse_dssgnn_variants_logs.py

# Write LaTeX table
python scripts/parse_dssgnn_variants_logs.py --update-table

# CSV output
python scripts/parse_dssgnn_variants_logs.py --csv
```

## Output

- **RESULT** lines: `dataset=... variant=... config=... acc_mean=... ece_mean=... mce_mean=... brier_mean=...`
- **tex/table_variants.tex**: LaTeX table (datasets × variants, Acc + ECE)
