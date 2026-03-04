import argparse
from clustek.io import parse_lammps_dump

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dump_file")
    parser.add_argument("--out", default="output.csv")
    parser.add_argument("--label-col", default=None)
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--comparison", default="<")

    args = parser.parse_args()

    df = parse_lammps_dump(
        args.dump_file,
        label_col=args.label_col,
        threshold=args.threshold,
        comparison=args.comparison,
    )

    df.to_csv(args.out, index=False)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
