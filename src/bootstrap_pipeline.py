from pathlib import Path


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    data_dir = root / "data"
    results_dir = root / "results"

    results_dir.mkdir(parents=True, exist_ok=True)

    summary = [
        "Base pipeline initialized.",
        f"Data directory: {data_dir}",
        f"Results directory: {results_dir}",
        "Next step: each member develops their assigned module.",
    ]

    out_file = results_dir / "base_setup_summary.txt"
    out_file.write_text("\n".join(summary), encoding="utf-8")
    print(f"Created: {out_file}")


if __name__ == "__main__":
    main()
