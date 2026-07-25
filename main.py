from experiments.scenarios.cross_play import main as run_cross_play
from experiments.plots.plot_regret import main as plot_regret
from experiments.build_report import main as build_report


def main() -> None:
    print("[1/3] Running experiments...")
    run_cross_play()

    print("[2/3] Generating plots...")
    plot_regret()

    print("[3/3] Building report...")
    build_report()

    print("[done]")


if __name__ == "__main__":
    main()
