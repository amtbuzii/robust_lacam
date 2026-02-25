"""
Create a figure with three vertically stacked tables (Success rate, Cost, Time)
from the experiment CSV files, matching the layout of the reference image.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_and_prepare_csv(path: Path) -> pd.DataFrame:
    """Load CSV and ensure first column is 'n'; row values shown as n=5, n=10, etc."""
    df = pd.read_csv(path)
    first_col = df.columns[0]
    df = df.rename(columns={first_col: "n"})
    # Display row labels as n=5, n=10, n=15 to match reference figure
    df["n"] = df["n"].astype(int).apply(lambda x: f"n={x}")
    return df


def format_success_cell(val) -> str:
    """Format success rate: integer or empty."""
    if pd.isna(val) or val == "" or str(val).strip().upper() == "N/A":
        return ""
    try:
        return str(int(float(val)))
    except (ValueError, TypeError):
        return str(val)


def format_cost_cell(val) -> str:
    """Format cost: integer or empty."""
    if pd.isna(val) or val == "" or str(val).strip().upper() == "N/A":
        return ""
    try:
        return str(int(round(float(val))))
    except (ValueError, TypeError):
        return str(val)


def format_time_cell(val) -> str:
    """Format time: integer with thousands comma, or empty."""
    if pd.isna(val) or val == "" or str(val).strip().upper() == "N/A":
        return ""
    try:
        n = int(round(float(val)))
        return f"{n:,}"
    except (ValueError, TypeError):
        return str(val)


def draw_table(ax, df: pd.DataFrame, title: str, formatter) -> None:
    """Draw one table on ax with given title and cell formatter (used for k=* columns only)."""
    cols = list(df.columns)
    n_cols = len(cols)

    cell_text = []
    for _, row in df.iterrows():
        row_vals = []
        for c in cols:
            if c == "n":
                row_vals.append(str(row[c]))
            else:
                row_vals.append(formatter(row[c]))
        cell_text.append(row_vals)

    ax.axis("tight")
    ax.axis("off")
    table = ax.table(
        cellText=cell_text,
        colLabels=cols,
        cellLoc="center",
        loc="center",
        bbox=[0, 0, 1, 1],
    )
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2)
    ax.set_title(title, fontsize=12, fontweight="bold", pad=15)

    # Style header row
    for i in range(n_cols):
        table[(0, i)].set_facecolor("#4472C4")
        table[(0, i)].set_text_props(weight="bold", color="white")


def main():
    base = Path(__file__).parent
    success_path = base / "success_rate.csv"
    cost_path = base / "average_soc.csv"  # Sum of Costs (SOC), not makespan
    time_path = base / "average_time.csv"

    if not success_path.exists():
        raise FileNotFoundError(f"Missing {success_path}")
    if not cost_path.exists():
        raise FileNotFoundError(f"Missing {cost_path}")
    if not time_path.exists():
        raise FileNotFoundError(f"Missing {time_path}")

    df_success = load_and_prepare_csv(success_path)
    df_cost = load_and_prepare_csv(cost_path)
    df_time = load_and_prepare_csv(time_path)

    # Build formatted DataFrames for display (so formatters see same structure)
    fig, axes = plt.subplots(3, 1, figsize=(10, 10))

    draw_table(axes[0], df_success, "Success rate", format_success_cell)
    draw_table(axes[1], df_cost, "Cost", format_cost_cell)
    draw_table(axes[2], df_time, "Time (ms)", format_time_cell)

    plt.tight_layout()
    out_png = base / "results_tables.png"
    out_pdf = base / "results_tables.pdf"
    plt.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.savefig(out_pdf, bbox_inches="tight")
    plt.close()
    print(f"Saved {out_png} and {out_pdf}")


if __name__ == "__main__":
    main()
