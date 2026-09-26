# Best Configs + ECE Experiments (Table 2)

Runs the best DS-GNN configs, TFE-GNN, and G-ΔUQ on all node classification datasets, reporting both **test accuracy** and **ECE**.

## Quick Start

### Local (single dataset, quick test)
```bash
python scripts/run_best_configs_acc_ece.py --dataset cora --model dssgnn --runs 2
# or
./scripts/run_best_configs_acc_ece_local.sh cora 2
```

## After experiments complete

Parse logs and update `tex/table2.tex`:
```bash
python scripts/parse_best_acc_ece_logs.py --update-table
```

## Output format

Each run prints a parseable line:
```
RESULT dataset=cora model=dssgnn config=prop_first_gfrw_dropout08 acc_mean=87.54 acc_std=1.15 ece_mean=0.1234 ece_std=0.02
```

## Best configs (from sweep)

| Dataset | DS-GNN Config |
|---------|---------------|
| cora | prop_first_gfrw_dropout08 |
| citeseer | P1 |
| pubmed | hidden128 |
| texas | prop_first_rmsprop |
| cornell | heterophily_tfe_layers3 |
| wisconsin | dropout06 |
| chameleon | prop_first_sum |
| squirrel | prop_first_gfrw_rmsprop |
| cora-full | dropout06 |
| cs | heterophily_tfe |
| physics | dropout06 |
| roman-empire | heterophily_tfe_layers3 |
| amazon-ratings | heterophily_tfe_layers3 |
| minesweeper | prop_first_gfrw_rmsprop |
| tolokers | prop_first_gfrw_rmsprop |
| questions | prop_first_rmsprop |
