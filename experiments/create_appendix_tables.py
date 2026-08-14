"""Create dissertation appendix LaTeX tables from summary CSV files.

Inputs:
- results/tables/portfolio_overall_summary.csv
- results/tables/regression_overall_summary.csv

Output:
- results/tables/appendix_orlibrary_portfolio_results.tex
- results/tables/appendix_regression_selected_features.tex
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
TABLES_DIR = ROOT / "results" / "tables"
PORTFOLIO_PATH = TABLES_DIR / "portfolio_overall_summary.csv"
REGRESSION_PATH = TABLES_DIR / "regression_overall_summary.csv"
PORTFOLIO_OUTPUT_PATH = TABLES_DIR / "appendix_orlibrary_portfolio_results.tex"
REGRESSION_OUTPUT_PATH = TABLES_DIR / "appendix_regression_selected_features.tex"


def _latex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\\textbackslash{}",
        "&": r"\\&",
        "%": r"\\%",
        "$": r"\\$",
        "#": r"\\#",
        "_": r"\\_",
        "{": r"\\{",
        "}": r"\\}",
        "~": r"\\textasciitilde{}",
        "^": r"\\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def _fmt_float(value: object, ndigits: int, missing: str = "--") -> str:
    if pd.isna(value):
        return missing
    return f"{float(value):.{ndigits}f}"


def _fmt_int(value: object, missing: str = "--") -> str:
    if pd.isna(value):
        return missing
    return str(int(float(value)))


def _clean_selected_features(value: object) -> str:
    if pd.isna(value):
        return "--"
    text = str(value).strip()
    if not text:
        return "--"

    # Try parsing JSON-like list to normalize whitespace and readability.
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            text = ", ".join(str(item) for item in parsed)
    except Exception:
        pass

    return text


def _to_longtable(
    df: pd.DataFrame,
    caption: str,
    label: str,
    column_headers: list[str],
    column_specs: str,
) -> str:
    lines: list[str] = []
    ncols = len(column_headers)
    header_line = " & ".join(column_headers) + r" \\" 

    lines.append(r"\begin{longtable}{" + column_specs + "}")
    lines.append(r"\caption{" + caption + r"}\label{" + label + r"}\\")
    lines.append(r"\toprule")
    lines.append(header_line)
    lines.append(r"\midrule")
    lines.append(r"\endfirsthead")

    lines.append(r"\toprule")
    lines.append(header_line)
    lines.append(r"\midrule")
    lines.append(r"\endhead")

    lines.append(r"\midrule")
    lines.append(r"\multicolumn{" + str(ncols) + r"}{r}{Continued on next page}\\")
    lines.append(r"\endfoot")

    lines.append(r"\bottomrule")
    lines.append(r"\endlastfoot")

    for _, row in df.iterrows():
        row_values = [_latex_escape(row[col]) for col in df.columns]
        lines.append(" & ".join(row_values) + r" \\")

    lines.append(r"\end{longtable}")
    return "\n".join(lines)


def _build_portfolio_table() -> str:
    if not PORTFOLIO_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {PORTFOLIO_PATH}")

    df = pd.read_csv(PORTFOLIO_PATH)
    df = df[df["experiment"] == "OR-Library"].copy()

    cols = [
        "dataset",
        "K",
        "category",
        "method",
        "variance",
        "risk",
        "return",
        "number_of_selected_assets",
        "solve_time",
        "mip_gap",
        "status",
    ]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in portfolio summary: {missing}")

    out = pd.DataFrame()
    out["dataset"] = df["dataset"].astype(str)
    out["K"] = df["K"].apply(_fmt_int)
    out["category"] = df["category"].astype(str)
    out["method"] = df["method"].astype(str)
    out["variance"] = df["variance"].apply(lambda x: _fmt_float(x, 6))
    out["risk"] = df["risk"].apply(lambda x: _fmt_float(x, 6))
    out["return"] = df["return"].apply(lambda x: _fmt_float(x, 6))
    out["number_of_selected_assets"] = df["number_of_selected_assets"].apply(_fmt_int)
    out["solve_time"] = df["solve_time"].apply(lambda x: _fmt_float(x, 3))
    out["mip_gap"] = df["mip_gap"].apply(lambda x: _fmt_float(x, 3, missing="--"))
    out["status"] = df["status"].apply(lambda x: "--" if pd.isna(x) else str(x))

    out = out.sort_values(["dataset", "K", "category"]).reset_index(drop=True)

    headers = [
        "Dataset",
        "K",
        "Category",
        "Method",
        "Variance",
        "Risk",
        "Return",
        "Selected",
        "Time (s)",
        "MIP Gap",
        "Status",
    ]

    # Landscape with compact font is recommended for this wide table.
    table = _to_longtable(
        out,
        caption="OR-Library portfolio appendix results (full instances).",
        label="tab:appendix_orlibrary_portfolio",
        column_headers=headers,
        column_specs="llp{2.8cm}p{2.5cm}rrrrrcc",
    )

    section = []
    section.append(r"\begin{landscape}")
    section.append(r"\footnotesize")
    section.append(table)
    section.append(r"\normalsize")
    section.append(r"\end{landscape}")
    return "\n".join(section)


def _build_regression_table() -> str:
    if not REGRESSION_PATH.exists():
        raise FileNotFoundError(f"Missing input file: {REGRESSION_PATH}")

    df = pd.read_csv(REGRESSION_PATH)

    allowed_categories = {
        "OLS Baseline",
        "Best LASSO",
        "Best Gurobi",
        "Best IHT",
        "Best Sparse",
        "Best Overall",
    }
    df = df[df["category"].isin(allowed_categories)].copy()

    cols = [
        "dataset",
        "category",
        "method",
        "K",
        "alpha",
        "test_mse",
        "number_of_selected_features",
        "status",
        "selected_feature_names",
    ]
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise KeyError(f"Missing columns in regression summary: {missing}")

    out = pd.DataFrame()
    out["dataset"] = df["dataset"].astype(str)
    out["category"] = df["category"].astype(str)
    out["method"] = df["method"].astype(str)
    out["K"] = df["K"].apply(_fmt_int)
    out["alpha"] = df["alpha"].apply(lambda x: _fmt_float(x, 3, missing="--"))
    out["test_mse"] = df["test_mse"].apply(lambda x: _fmt_float(x, 6))
    out["number_of_selected_features"] = df["number_of_selected_features"].apply(_fmt_int)
    out["status"] = df["status"].apply(lambda x: "--" if pd.isna(x) else str(x))
    out["selected_feature_names"] = df["selected_feature_names"].apply(_clean_selected_features)

    out = out.sort_values(["dataset", "category"]).reset_index(drop=True)

    headers = [
        "Dataset",
        "Category",
        "Method",
        "K",
        "Alpha",
        "Test MSE",
        "Selected",
        "Status",
        "Selected Feature Names",
    ]

    table = _to_longtable(
        out,
        caption="Regression appendix selected-feature results.",
        label="tab:appendix_regression_features",
        column_headers=headers,
        column_specs="llp{2.4cm}rrrccp{8.2cm}",
    )

    section = []
    section.append(r"\begin{landscape}")
    section.append(r"\footnotesize")
    section.append(table)
    section.append(r"\normalsize")
    section.append(r"\end{landscape}")
    return "\n".join(section)


def _latex_file_preamble() -> list[str]:
    return [
        "% Auto-generated appendix table",
        "% Required packages in Overleaf preamble:",
        "% \\usepackage{booktabs}",
        "% \\usepackage{longtable}",
        "% \\usepackage{pdflscape}",
        "% \\usepackage{array}",
        "",
    ]


def main() -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    portfolio_parts = _latex_file_preamble()
    portfolio_parts.append(_build_portfolio_table())
    portfolio_parts.append("")
    PORTFOLIO_OUTPUT_PATH.write_text("\n".join(portfolio_parts), encoding="utf-8")

    regression_parts = _latex_file_preamble()
    regression_parts.append(_build_regression_table())
    regression_parts.append("")
    REGRESSION_OUTPUT_PATH.write_text("\n".join(regression_parts), encoding="utf-8")

    print(f"Saved LaTeX appendix table to: {PORTFOLIO_OUTPUT_PATH}")
    print(f"Saved LaTeX appendix table to: {REGRESSION_OUTPUT_PATH}")


if __name__ == "__main__":
    main()
