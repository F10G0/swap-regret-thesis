GAME_PRESENTATIONS = {
    "rps": {
        "label": "Rock–Paper–Scissors",
        "description": "Core symmetric zero-sum benchmark: unique CE versus a broader CCE set.",
    },
    "rpsls": {
        "label": "Rock–Paper–Scissors–Lizard–Spock",
        "description": "Five-action symmetric zero-sum extension used by Leme et al. (2024) to study learning dynamics.",
    },
    "matching_pennies": {
        "label": "Matching Pennies",
        "description": "Two-action asymmetric zero-sum control for the symmetry boundary and external-versus-swap-regret comparison.",
    },
    "bertrand_standard_o1": {
        "label": "O1 — Standard Bertrand (symmetric)",
        "description": "Economic baseline with 21 prices in [0.05, 1.00], equal costs, and upstream homogeneous-good demand.",
    },
    "bertrand_linear_o2": {
        "label": "O2 — Linear Bertrand (symmetric)",
        "description": "Symmetric differentiated-demand application with 21 prices in [0, 1]; α=0.48, β=0.9, γ=0.6.",
    },
    "bertrand_logit_o3": {
        "label": "O3 — Logit Bertrand (symmetric)",
        "description": "Symmetric logit-demand application with 21 prices in [1, 2], equal costs, and an upstream outside option.",
    },
    "bertrand_linear_o2_prime": {
        "label": "O2′ — Linear Bertrand (asymmetric)",
        "description": "Asymmetric-cost control for linear demand with 21 prices in [0, 1]; costs (0, 0.2).",
    },
    "bertrand_logit_o3_prime": {
        "label": "O3′ — Logit Bertrand (asymmetric)",
        "description": "Asymmetric-cost and quality control for logit demand with 21 prices in [0.5, 2].",
    },
}
