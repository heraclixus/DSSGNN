# OOD Score Choices

This document explains the scalar node-level OOD scores supported by the
GNNSafe-style pipeline in this repo, and how they relate to training.

## Big Picture

The runner now separates:

- **training objective**: controlled by `--method`
- **OOD regularizer**: controlled by `--reg_score_type` when `--use_reg` is on
- **detection score**: controlled by `--score_type`
- **score propagation**: controlled by `--use_prop`

This lets us reuse the same trained backbone while swapping the OOD score.

## Training Methods

- `--method msp`
  - Standard supervised node classification loss only.
  - Default detection score: `msp`.

- `--method gnnsafe`
  - Standard supervised node classification loss.
  - If `--use_reg` is off: plain Energy / GNNSafe-style training.
  - If `--use_reg` is on with `--reg_score_type energy`: GNNSafe++-style energy regularization with OOD-train data.
  - If `--use_reg` is on with `--reg_score_type pred_entropy`: predictive-entropy margin regularization with OOD-train data.
  - If `--use_reg` is on with `--reg_score_type chaos`: DSS-GNN chaos-margin regularization with OOD-train data.
  - Default detection score: `energy`, except it switches to `chaos` for chaos-regularized DSS-GNN runs.

- `--method gduq`
  - Standard supervised node classification loss on the GDUQ-anchored backbone.
  - Default detection score: `std`.

Important:
- `--use_reg` defaults to the original **energy-based** GNNSafe regularizer.
- `--reg_score_type chaos` switches the OOD regularizer to DSS-GNN chaos uncertainty.
- `--dss_lambda_reg` restores the standalone DSS-GNN IND uncertainty penalty during
  backbone training, using `--dss_reg_score_type` to choose the uncertainty summary.
- `--warmup_base_epochs` and `--residual_scale_init` are used by the hybrid
  `gcn_dssres` backbone, which trains a GCN base first and then turns on a DSS residual branch.
- `--mi_weight` controls the hybrid `energy_mutual_info` detection score.
- You can override `--score_type` at evaluation time, but that does not change the
  training loss unless you also change `--method` or `--reg_score_type`.

## Score Types

All scores follow the same convention:

- **higher score = more in-distribution**
- **lower score = more out-of-distribution**

### `msp`

Maximum softmax probability on the predicted logits.

- Formula: `max_c softmax(logits)_c`
- Higher confidence means larger score.
- Supported on all backbones.

Example:

```bash
python run_ood_gnnsafe.py --method msp --backbone dssgnn --score_type msp
```

### `energy`

Energy-style logit score.

- Formula: `T * logsumexp(logits / T)`
- This is the **negative** of the usual energy up to sign convention, chosen so
  that larger means more in-distribution.
- Supported on all backbones.

Example:

```bash
python run_ood_gnnsafe.py --method gnnsafe --backbone gcn --score_type energy
```

### `pred_entropy`

Quadrature-based predictive entropy from the DSS stochastic field.

- Reconstruct quadrature logits `z^(s)` from the final DSS coefficient stack.
- Compute predictive probabilities `p_bar = sum_s w_s softmax(z^(s))`.
- Detection score used by the pipeline: `-H(p_bar)`.

Supported on:

- `backbone=dssgnn`
- `backbone=gcn_dssres`

### `mutual_info`

Quadrature-based mutual information between predictions and the DSS stochastic
state.

- Formula: `MI = H(p_bar) - sum_s w_s H(softmax(z^(s)))`
- Detection score used by the pipeline: `-MI`

Supported on:

- `backbone=dssgnn`
- `backbone=gcn_dssres`

### `energy_mutual_info`

Hybrid IND score that combines deterministic confidence and stochastic
predictive disagreement:

- `energy_mutual_info = energy - lambda * mutual_info`

where `lambda` is `--mi_weight`.

Supported on:

- `backbone=dssgnn`
- `backbone=gcn_dssres`

### `chaos`

DSS-GNN chaos uncertainty score converted into an IND-oriented scalar.

- DSS-GNN uncertainty from the model: `V_chaos = sum_{n>0} ||Z_n||^2`
- Detection score used by the pipeline: `-V_chaos`
- Interpretation:
  - small chaos energy => more confident / more IND
  - large chaos energy => more uncertain / more OOD

Supported on:

- `backbone=dssgnn`
- `backbone=gduq` (chaos uncertainty from the wrapped DSS-GNN base model)

Example:

```bash
python run_ood_gnnsafe.py --method msp --backbone dssgnn --score_type chaos
```

### `chaos_norm`, `chaos_layernorm`

Scale-normalized chaos variants.

- `chaos_norm`: final-layer `V_chaos / (||Z_0||^2 + eps)`
- `chaos_layernorm`: average of the same normalization across DSS layers

These are more scale-aware than raw chaos, but can still be sensitive when the
deterministic channel has very small norm.

### `chaos_ratio`, `chaos_layerratio`

Bounded chaos-fraction variants.

- `chaos_ratio`: `V_chaos / (V_chaos + ||Z_0||^2 + eps)`
- `chaos_layerratio`: average layer-wise version of the same ratio

These stay in `[0, 1]`, which makes them numerically more stable than raw
norm ratios and easier to combine with energy scores.

### `energy_chaos_ratio`, `energy_chaos_layerratio`

Hybrid IND scores that combine confidence and uncertainty:

- `energy_chaos_ratio = energy - lambda * chaos_ratio`
- `energy_chaos_layerratio = energy - lambda * chaos_layerratio`

where `lambda` is `--chaos_weight`.

These are useful when DSS uncertainty works better as a correction to energy
than as a standalone OOD score.

### `std`

Predictive standard deviation from the GDUQ-style anchor model.

- Detection score used by the pipeline: `-mean(std over classes)`
- Higher predictive variance means more OOD-like.

Supported on:

- `backbone=gduq`

Example:

```bash
python run_ood_gnnsafe.py --method gduq --backbone gduq --score_type std
```

## Propagation

`--use_prop` applies the GNNSafe neighborhood propagation rule to the chosen
scalar node score, not just to energy.

That means the following are now valid design choices:

- raw energy
- propagated energy
- raw predictive entropy
- propagated predictive entropy
- raw mutual information
- propagated mutual information
- propagated energy-minus-mutual-information
- raw chaos score
- raw normalized chaos scores
- raw chaos-fraction scores
- hybrid energy-minus-chaos scores
- propagated chaos score
- raw MSP
- propagated MSP
- raw GDUQ std
- propagated GDUQ std

The propagation rule is:

```text
s^{t+1} = alpha * s^t + (1 - alpha) * A_rw * s^t
```

where `s` is the scalar node score.

For DSS-capable backbones, this is the direct analogue of GNNSafe energy
propagation applied to chaos-derived or quadrature-derived uncertainty scores.

## Uncertainty-Regularized OOD Training

When using:

```bash
--method gnnsafe --backbone dssgnn --use_reg --reg_score_type chaos
```

the OOD regularizer switches from energy to chaos uncertainty.

- ID nodes should have **smaller** chaos energy.
- OOD-train nodes should have **larger** chaos energy.
- The implemented loss is a squared margin-ranking penalty:

```text
L_chaos = mean( relu(gamma + U_id - U_ood)^2 )
```

where `U = V_chaos` and `gamma` is `--chaos_margin`.

This gives two new variants:

- `chaos_ft` = chaos regularization without propagation
- `chaos_safe` = chaos regularization with propagated score/regularizer

The same chaos-margin objective can also be applied to the stabilized variants
`chaos_norm`, `chaos_layernorm`, `chaos_ratio`, and `chaos_layerratio` via
`--reg_score_type`.

When using:

```bash
--method gnnsafe --backbone gcn_dssres --use_reg --reg_score_type pred_entropy
```

the regularizer switches to quadrature predictive entropy.

- ID nodes should have **smaller** predictive entropy.
- OOD-train nodes should have **larger** predictive entropy.
- The implemented loss uses the same squared margin-ranking template:

```text
L_predent = mean( relu(gamma + H_id - H_ood)^2 )
```

where `H` is predictive entropy and `gamma` is still controlled by
`--chaos_margin`.

The main local configs for this path are:

- `gcn_dssres_pred_entropy_ft`
- `gcn_dssres_pred_entropy_safe`
- `gcn_dssres_predsafe_energy_mi_prop`

Empirically, this regularizer improves predictive-entropy-based detection more
on feature OOD than on structure OOD, but the strongest overall baseline is
still `gcn_dssres_gnnsafepp` with energy-based training.

## Learned Fusion Detectors

For local score analysis, `scripts/local_ood_density_compare.py` also supports
small learned detectors that combine multiple scalar scores.

The current fusion detector:

- builds a feature vector from score variants such as
  `energy`, `energy_prop`, `chaos_ratio`, `chaos_ratio_prop`,
  `chaos_layerratio`, and `chaos_layerratio_prop`
- standardizes those features
- fits a logistic-regression detector on
  `IND-valid` vs `OOD-train`
- uses the linear decision function as the final IND-oriented score

This is useful when no single handcrafted score separates IND and OOD cleanly,
but different scores carry complementary signal.

The main local fusion configs are:

- `dssgnn_fusion`
  - features: `energy`, `chaos_ratio`, `chaos_layerratio`
- `dssgnn_fusion_prop`
  - features: `energy`, `energy_prop`, `chaos_ratio`,
    `chaos_ratio_prop`, `chaos_layerratio`, `chaos_layerratio_prop`
- `dssgnn_fusion_propcompact`
  - features: `energy_prop`, `chaos_ratio_prop`, `chaos_layerratio_prop`
- `dssgnn_fusion_chaosft`
  - same expanded feature set, but on the `chaos_ratio`-regularized backbone
- `gcn_dssres_fusion_probunc`
  - propagated `energy_prop`, `pred_entropy_prop`, and `mutual_info_prop`
  - intended for the hybrid low-label backbone

At the moment these fusion detectors are exposed through the local analysis
runner, not the main `run_ood_gnnsafe.py` CLI.

## Recommended DSS-GNN Variants

For a clean DSS-GNN score study, the most useful variants are:

1. `msp + dssgnn + score_type=msp`
2. `msp + dssgnn + score_type=energy`
3. `msp + dssgnn + score_type=energy + use_prop`
4. `msp + dssgnn + score_type=chaos`
5. `msp + dssgnn + score_type=chaos + use_prop`
6. `gnnsafe + dssgnn + use_reg + reg_score_type=chaos`
7. `gnnsafe + dssgnn + use_reg + reg_score_type=chaos + use_prop`

The first five are available as the `dssgnn_scores` config set in
`scripts/run_ood_table1_table2.py`, and the chaos-regularized pair is available
as `dssgnn_chaos_reg`.

For local DSS-GNN analysis, the most informative additions are often:

8. `dssgnn_fusion_prop`
9. `dssgnn_fusion_propcompact`
10. `dssgnn_fusion_chaosft`

For the hybrid low-label path, the most informative follow-up comparisons are:

11. `gcn_dssres + score_type=pred_entropy`
12. `gcn_dssres + score_type=pred_entropy + use_prop`
13. `gcn_dssres + score_type=mutual_info`
14. `gcn_dssres + score_type=mutual_info + use_prop`
15. `gcn_dssres + score_type=energy_mutual_info + use_prop`
16. `gcn_dssres_fusion_probunc`

## Example Commands

### GNNSafe baseline

```bash
python run_ood_gnnsafe.py \
  --method gnnsafe --backbone gcn \
  --dataset cora --ood_type structure \
  --score_type energy --use_prop
```

### DSS-GNN with propagated chaos score

```bash
python run_ood_gnnsafe.py \
  --method msp --backbone dssgnn \
  --dataset cora --ood_type structure \
  --score_type chaos --use_prop
```

### DSS-GNN with chaos-based OOD fine-tuning

```bash
python run_ood_gnnsafe.py \
  --method gnnsafe --backbone dssgnn \
  --dataset cora --ood_type structure \
  --use_reg --reg_score_type chaos \
  --score_type chaos
```

### DSS-GNN score sweep

```bash
python scripts/run_ood_table1_table2.py --config_set dssgnn_scores --all --runs 10
```

## Result Naming

When `--score_type` overrides the method default, result names gain a suffix.

Examples:

- `msp_dssgnn_chaos`
- `msp_dssgnn_chaos_prop`
- `gnnsafe_dssgnn_chaos_prop`
- `chaos_ft_dssgnn`
- `chaos_safe_dssgnn`

Default names are unchanged for the original settings:

- `msp_gcn`
- `energy_gcn`
- `gnnsafe_gcn`
- `gduq_gduq`
