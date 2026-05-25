from pathlib import Path
from typing import Any

import matplotlib


matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


ENERGY_TABLE = "electric_power"
PROJECT_ROOT = Path(__file__).resolve().parents[3]
REPORTS_FIGURES_DIR = PROJECT_ROOT / "reports" / "figures"
NUMERIC_COLUMNS = [
    "Global_active_power",
    "Global_reactive_power",
    "Voltage",
    "Global_intensity",
    "Sub_metering_1",
    "Sub_metering_2",
    "Sub_metering_3",
]


def _ensure_output_path(output_path: str | Path) -> Path:
    path = Path(output_path)
    if not path.is_absolute():
        if path.parts[:2] == ("reports", "figures"):
            path = PROJECT_ROOT / path
        else:
            path = REPORTS_FIGURES_DIR / path
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _fetch_dataframe(con: Any, sql: str) -> pd.DataFrame:
    try:
        return con.execute(sql).fetchdf()
    except Exception as exc:
        raise ValueError(f"Failed to run visualization query: {exc}") from exc


def _save_figure(output_path: str | Path) -> Path:
    path = _ensure_output_path(output_path)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    return path


def plot_daily_active_power_trend(con: Any, output_path: str | Path) -> Path:
    query = f"""
        SELECT
            CAST(datetime AS DATE) AS day,
            AVG(Global_active_power) AS avg_global_active_power_kw
        FROM {ENERGY_TABLE}
        WHERE datetime IS NOT NULL
          AND Global_active_power IS NOT NULL
        GROUP BY day
        ORDER BY day
    """
    df = _fetch_dataframe(con, query)
    if df.empty:
        raise ValueError("No daily active power data available for plotting.")

    plt.figure(figsize=(10, 4.8))
    plt.plot(df["day"], df["avg_global_active_power_kw"], linewidth=1.4)
    plt.title("Rata-rata Daya Aktif Harian")
    plt.xlabel("Tanggal")
    plt.ylabel("Daya aktif rata-rata (kW)")
    plt.grid(True, alpha=0.3)
    return _save_figure(output_path)


def plot_hourly_consumption_pattern(con: Any, output_path: str | Path) -> Path:
    query = f"""
        SELECT
            EXTRACT(hour FROM datetime) AS hour,
            AVG(Global_active_power) AS avg_global_active_power_kw
        FROM {ENERGY_TABLE}
        WHERE datetime IS NOT NULL
          AND Global_active_power IS NOT NULL
        GROUP BY hour
        ORDER BY hour
    """
    df = _fetch_dataframe(con, query)
    if df.empty:
        raise ValueError("No hourly consumption data available for plotting.")

    plt.figure(figsize=(9, 4.8))
    plt.bar(df["hour"], df["avg_global_active_power_kw"], width=0.75)
    plt.title("Pola Konsumsi Daya per Jam")
    plt.xlabel("Jam")
    plt.ylabel("Daya aktif rata-rata (kW)")
    plt.xticks(range(0, 24))
    plt.grid(axis="y", alpha=0.3)
    return _save_figure(output_path)


def plot_power_distribution(con: Any, output_path: str | Path) -> Path:
    query = f"""
        SELECT
            FLOOR(Global_active_power * 10) / 10.0 AS power_bin_kw,
            COUNT(*) AS record_count
        FROM {ENERGY_TABLE}
        WHERE Global_active_power IS NOT NULL
        GROUP BY power_bin_kw
        ORDER BY power_bin_kw
    """
    df = _fetch_dataframe(con, query)
    if df.empty:
        raise ValueError("No active power distribution data available for plotting.")

    plt.figure(figsize=(10, 4.8))
    plt.bar(df["power_bin_kw"], df["record_count"], width=0.08)
    plt.title("Distribusi Global Active Power")
    plt.xlabel("Global active power (kW)")
    plt.ylabel("Jumlah record")
    plt.grid(axis="y", alpha=0.3)
    return _save_figure(output_path)


def plot_voltage_distribution(con: Any, output_path: str | Path) -> Path:
    query = f"""
        SELECT
            FLOOR(Voltage) AS voltage_bin_v,
            COUNT(*) AS record_count
        FROM {ENERGY_TABLE}
        WHERE Voltage IS NOT NULL
        GROUP BY voltage_bin_v
        ORDER BY voltage_bin_v
    """
    df = _fetch_dataframe(con, query)
    if df.empty:
        raise ValueError("No voltage distribution data available for plotting.")

    plt.figure(figsize=(10, 4.8))
    plt.bar(df["voltage_bin_v"], df["record_count"], width=0.8)
    plt.title("Distribusi Voltage")
    plt.xlabel("Voltage (V)")
    plt.ylabel("Jumlah record")
    plt.grid(axis="y", alpha=0.3)
    return _save_figure(output_path)


def plot_correlation_heatmap(con: Any, output_path: str | Path) -> Path:
    select_parts = []
    aliases = {}
    for left_idx, left_column in enumerate(NUMERIC_COLUMNS):
        for right_idx, right_column in enumerate(NUMERIC_COLUMNS):
            alias = f"corr_{left_idx}_{right_idx}"
            aliases[(left_idx, right_idx)] = alias
            select_parts.append(f"corr({left_column}, {right_column}) AS {alias}")

    query = f"""
        SELECT
            {", ".join(select_parts)}
        FROM {ENERGY_TABLE}
    """
    result = _fetch_dataframe(con, query)

    rows = []
    for left_idx, _ in enumerate(NUMERIC_COLUMNS):
        row = []
        for right_idx, _ in enumerate(NUMERIC_COLUMNS):
            row.append(result.loc[0, aliases[(left_idx, right_idx)]])
        rows.append(row)

    corr_df = pd.DataFrame(rows, index=NUMERIC_COLUMNS, columns=NUMERIC_COLUMNS)
    corr_df = corr_df.apply(pd.to_numeric, errors="coerce")
    if corr_df.isna().all().all():
        raise ValueError("No correlation data available for plotting.")

    labels = [
        "Active",
        "Reactive",
        "Voltage",
        "Intensity",
        "Sub 1",
        "Sub 2",
        "Sub 3",
    ]

    plt.figure(figsize=(8, 6.5))
    image = plt.imshow(corr_df, cmap="coolwarm", vmin=-1, vmax=1)
    plt.colorbar(image, fraction=0.046, pad=0.04, label="Korelasi")
    plt.title("Korelasi Fitur Energi")
    plt.xticks(range(len(labels)), labels, rotation=35, ha="right")
    plt.yticks(range(len(labels)), labels)

    for row_idx in range(len(labels)):
        for col_idx in range(len(labels)):
            value = corr_df.iloc[row_idx, col_idx]
            if pd.notna(value):
                plt.text(
                    col_idx,
                    row_idx,
                    f"{value:.2f}",
                    ha="center",
                    va="center",
                    color="black",
                    fontsize=8,
                )

    return _save_figure(output_path)


def plot_sub_metering_comparison(con: Any, output_path: str | Path) -> Path:
    query = f"""
        SELECT
            AVG(Sub_metering_1) AS avg_sub_metering_1_wh,
            AVG(Sub_metering_2) AS avg_sub_metering_2_wh,
            AVG(Sub_metering_3) AS avg_sub_metering_3_wh
        FROM {ENERGY_TABLE}
    """
    df = _fetch_dataframe(con, query)
    if df.empty:
        raise ValueError("No sub metering data available for plotting.")

    values = [
        df.loc[0, "avg_sub_metering_1_wh"],
        df.loc[0, "avg_sub_metering_2_wh"],
        df.loc[0, "avg_sub_metering_3_wh"],
    ]
    if all(pd.isna(value) for value in values):
        raise ValueError("No sub metering data available for plotting.")

    plt.figure(figsize=(7.5, 4.8))
    plt.bar(["Sub 1", "Sub 2", "Sub 3"], values)
    plt.title("Perbandingan Rata-rata Sub Metering")
    plt.xlabel("Sub metering")
    plt.ylabel("Rata-rata sub metering (Wh)")
    plt.grid(axis="y", alpha=0.3)
    return _save_figure(output_path)
