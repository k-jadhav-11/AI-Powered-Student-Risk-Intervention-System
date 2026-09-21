"""
Export any DataFrame or matplotlib Figure to the reports/ folder,
with a timestamp so repeated exports don't overwrite each other.
"""
import os
from datetime import datetime
import pandas as pd


# Helper function to ensure the reports folder exists.
def ensure_reports_dir(reports_dir: str = "reports") -> str:
    os.makedirs(reports_dir, exist_ok=True)
    return reports_dir

# Helper function to generate a timestamped filename.
def _timestamped_name(filename: str, ext: str) -> str:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = filename.replace(" ", "_")
    return f"{safe_name}_{stamp}.{ext}"

# Helper function to export a DataFrame to a CSV file in the reports folder.
def export_dataframe_csv(df: pd.DataFrame, filename: str, reports_dir: str = "reports") -> str:
    ensure_reports_dir(reports_dir)
    out_name = _timestamped_name(filename, "csv")
    out_path = os.path.join(reports_dir, out_name)
    df.to_csv(out_path, index=False)
    return out_path


# Helper function to export a matplotlib Figure to a PNG file in the reports folder.
def export_chart_png(fig, filename: str, reports_dir: str = "reports") -> str:
    ensure_reports_dir(reports_dir)
    out_name = _timestamped_name(filename, "png")
    out_path = os.path.join(reports_dir, out_name)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    return out_path
