import argparse
import operator
import pandas as pd


# -----------------------------
# Comparison operator mapping
# -----------------------------
OPS = {
    "<": operator.lt,
    "<=": operator.le,
    ">": operator.gt,
    ">=": operator.ge,
}


def parse_lammps_dump(
    dump_file,
    label_col=None,
    threshold=None,
    comparison="<",
):
    with open(dump_file) as f:
        lines = f.readlines()

    i = 0
    rows = []

    if label_col and threshold is None:
        raise ValueError("If --label-col is given, --threshold must also be given.")

    comp_func = OPS.get(comparison)

    while i < len(lines):

        if lines[i].startswith("ITEM: TIMESTEP"):
            step = int(lines[i + 1].strip())
            i += 2

            assert lines[i].startswith("ITEM: NUMBER OF ATOMS")
            N = int(lines[i + 1].strip())
            i += 2

            assert lines[i].startswith("ITEM: BOX BOUNDS")
            #xlo, xhi = map(float, lines[i + 1].split()[:2])
            #ylo, yhi = map(float, lines[i + 2].split()[:2])
            #zlo, zhi = map(float, lines[i + 3].split()[:2])
            xlo, xhi = [round(float(v), 4) for v in lines[i + 1].split()[:2]]
            ylo, yhi = [round(float(v), 4) for v in lines[i + 2].split()[:2]]
            zlo, zhi = [round(float(v), 4) for v in lines[i + 3].split()[:2]]
            i += 4

            assert lines[i].startswith("ITEM: ATOMS")
            header = lines[i].strip().split()[2:]
            i += 1

            col_index = {name: idx for idx, name in enumerate(header)}

            if label_col and label_col not in col_index:
                raise ValueError(f"Column '{label_col}' not found in dump file.")

            for _ in range(N):
                parts = lines[i].split()

                row = {
                    "step": step,
                    "xlo": xlo,
                    "xhi": xhi,
                    "ylo": ylo,
                    "yhi": yhi,
                    "zlo": zlo,
                    "zhi": zhi,
                }

                # Add all columns dynamically
                for name, idx_col in col_index.items():
                    try:
                        val = float(parts[idx_col])
                    except ValueError:
                        val = parts[idx_col]
                    row[name] = val

                # Optional binary label
                if label_col:
                    val = float(parts[col_index[label_col]])
                    row["c_label"] = 1 if comp_func(val, threshold) else 0

                rows.append(row)
                i += 1
        else:
            i += 1

    return pd.DataFrame(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("dump_file")
    parser.add_argument("--out", default="output.csv")

    parser.add_argument("--label-col", default=None,
                        help="Column to threshold for c_label (e.g., c_entr)")
    parser.add_argument("--threshold", type=float, default=None,
                        help="Threshold value for label column")
    parser.add_argument("--comparison", choices=["<", "<=", ">", ">="],
                        default="<", help="Comparison operator")

    args = parser.parse_args()

    df = parse_lammps_dump(
        args.dump_file,
        label_col=args.label_col,
        threshold=args.threshold,
        comparison=args.comparison,
    )

    df.to_csv(args.out, index=False)

    print(f"Wrote {args.out}")
    print(f"Rows: {len(df)}")


if __name__ == "__main__":
    main()