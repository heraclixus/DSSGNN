"""Argument parser for GNNSafe OOD pipeline."""


def parser_add_main_args(parser):
    parser.add_argument("--dataset", type=str, default="cora")
    parser.add_argument(
        "--ood_type",
        type=str,
        default="structure",
        choices=["structure", "label", "feature"],
        help="OOD type for cora/amazon/coauthor datasets",
    )
    parser.add_argument("--data_dir", type=str, default="data/")
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--train_prop", type=float, default=0.1)
    parser.add_argument("--valid_prop", type=float, default=0.1)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--epochs", type=int, default=200)

    parser.add_argument("--method", type=str, default="msp", choices=["msp", "gnnsafe", "gduq", "graph_ebm", "mc_dropout", "ensemble"])
    parser.add_argument("--backbone", type=str, default="gcn", choices=["gcn", "dssgnn", "gcn_dssres", "tfe", "gduq"])
    parser.add_argument(
        "--score_type",
        type=str,
        default="auto",
        choices=[
            "auto",
            "msp",
            "energy",
            "pred_entropy",
            "mutual_info",
            "energy_mutual_info",
            "energy_chaos_ratio",
            "energy_chaos_layerratio",
            "chaos",
            "chaos_norm",
            "chaos_layernorm",
            "chaos_ratio",
            "chaos_layerratio",
            "std",
        ],
        help="Detection score. auto = method default (msp, energy, or std).",
    )
    parser.add_argument("--hidden_channels", type=int, default=64)
    parser.add_argument("--num_layers", type=int, default=2)
    parser.add_argument("--dropout", type=float, default=0.0)
    parser.add_argument("--use_bn", action="store_true")
    parser.add_argument("--dss_bn", action="store_true",
                        help="enable per-chaos-channel BatchNorm inside the standalone DSSGNN backbone")
    parser.add_argument(
        "--dss_lambda_reg",
        type=float,
        default=0.0,
        help="Standalone DSS-GNN-style IND uncertainty penalty added during training.",
    )
    parser.add_argument(
        "--dss_reg_score_type",
        type=str,
        default="chaos",
        choices=["chaos", "chaos_norm", "chaos_layernorm", "chaos_ratio", "chaos_layerratio"],
        help="Uncertainty summary used by the DSS IND regularizer.",
    )

    # GNNSafe hyperparams
    parser.add_argument("--T", type=float, default=1.0, help="temperature for Softmax")
    parser.add_argument("--use_reg", action="store_true")
    parser.add_argument("--lamda", type=float, default=1.0)
    parser.add_argument("--m_in", type=float, default=-5)
    parser.add_argument("--m_out", type=float, default=-1)
    parser.add_argument(
        "--reg_score_type",
        type=str,
        default="energy",
        choices=[
            "energy",
            "pred_entropy",
            "mutual_info",
            "chaos",
            "chaos_norm",
            "chaos_layernorm",
            "chaos_ratio",
            "chaos_layerratio",
        ],
        help="OOD regularizer score used when --use_reg is enabled.",
    )
    parser.add_argument(
        "--chaos_margin",
        type=float,
        default=0.01,
        help="Pairwise margin for chaos-based OOD regularization.",
    )
    parser.add_argument(
        "--chaos_weight",
        type=float,
        default=1.0,
        help="Weight for hybrid energy-minus-chaos detection scores.",
    )
    parser.add_argument(
        "--mi_weight",
        type=float,
        default=1.0,
        help="Weight for hybrid energy-minus-mutual-information detection scores.",
    )
    parser.add_argument("--use_prop", action="store_true")
    parser.add_argument("--K", type=int, default=2)
    parser.add_argument("--alpha", type=float, default=0.5)

    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--display_step", type=int, default=1)
    parser.add_argument("--mode", type=str, default="detect", choices=["classify", "detect"])

    # DSS-GNN specific
    parser.add_argument("--K_lp", type=int, default=3)
    parser.add_argument("--K_hp", type=int, default=2)
    parser.add_argument("--P", type=int, default=1)
    parser.add_argument("--use_random_gates", action="store_true")
    parser.add_argument("--P_gate", type=int, default=1)
    parser.add_argument("--S", type=int, default=4, help="Quadrature nodes for DSS-GNN non-intrusive projection.")
    parser.add_argument("--lambda_max", type=float, default=2.0)
    parser.add_argument("--shared_input_lift", action="store_true")
    parser.add_argument("--warmup_base_epochs", type=int, default=50)
    parser.add_argument("--residual_scale_init", type=float, default=0.1)

    # TFE-GNN specific
    parser.add_argument("--hop_lp", type=int, default=2)
    parser.add_argument("--hop_hp", type=int, default=2)
    parser.add_argument("--tfe_combine", type=str, default="sum", choices=["sum", "con", "lp", "hp"])
    parser.add_argument("--eta", type=float, default=0.5)

    # GDUQ specific
    parser.add_argument("--gduq_n_anchors", type=int, default=5)

    # MC-dropout specific
    parser.add_argument(
        "--mc_samples",
        type=int,
        default=20,
        help="Number of stochastic forward passes for the mc_dropout method.",
    )

    # Deep ensemble specific
    parser.add_argument(
        "--ensemble_size",
        type=int,
        default=5,
        help="Number of independently initialized backbones for the ensemble method.",
    )

    # Graph-EBM specific
    parser.add_argument("--graph_ebm_covariance_type", type=str, default="diagonal", choices=["diagonal", "full", "identity", "isotropic"])
    parser.add_argument("--graph_ebm_tied_covariance", action="store_true")
    parser.add_argument("--graph_ebm_gamma_correction", type=float, default=1.0)
    parser.add_argument("--graph_ebm_lambda_independent_energy", type=float, default=1.0)
    parser.add_argument("--graph_ebm_lambda_local_energy", type=float, default=1.0)
    parser.add_argument("--graph_ebm_lambda_group_energy", type=float, default=1.0)
    parser.add_argument("--graph_ebm_alpha", type=float, default=0.5)
    parser.add_argument("--graph_ebm_num_diffusion_steps", type=int, default=10)
    parser.add_argument("--graph_ebm_aggregation", type=str, default="sum", choices=["sum", "logsumexp"])
