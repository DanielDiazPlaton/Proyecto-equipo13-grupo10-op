"""Analisis exploratorio (EDA): estadisticas descriptivas y figuras del reporte."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from mibici.config import FIGURES_DIR

BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
CATEGORICAL = [BLUE, ORANGE, AQUA, YELLOW]
MARKERS = ["o", "s", "^", "D"]
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
DAYS = ["Lun", "Mar", "Mié", "Jue", "Vie", "Sáb", "Dom"]

plt.rcParams.update(
    {
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.dpi": 160,
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.edgecolor": GRID,
        "axes.labelcolor": MUTED,
        "xtick.color": MUTED,
        "ytick.color": MUTED,
        "axes.titlesize": 12,
        "axes.titleweight": "bold",
        "axes.titlecolor": TEXT,
        "axes.titlelocation": "left",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": True,
        "grid.color": GRID,
        "grid.linewidth": 0.8,
        "axes.axisbelow": True,
        "legend.frameon": False,
    }
)


class EDAReport:
    """Genera las estadisticas y figuras del EDA a partir de los viajes limpios."""

    def __init__(self, trips: pd.DataFrame, stations: pd.DataFrame, out_dir: Path = FIGURES_DIR):
        self.trips = trips
        self.stations = stations
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.files: list[Path] = []

    def _save(self, fig, name: str) -> Path:
        path = self.out_dir / name
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        self.files.append(path)
        return path

    # Estadisticas descriptivas
    def summary(self) -> dict:
        t = self.trips
        daily = t.groupby(t["start_time"].dt.date).size()
        weekday = daily[pd.to_datetime(daily.index).dayofweek < 5]
        weekend = daily[pd.to_datetime(daily.index).dayofweek >= 5]
        return {
            "viajes": len(t),
            "usuarios_unicos": t["user_id"].nunique(),
            "estaciones_origen": t["origin_id"].nunique(),
            "viajes_dia_promedio": daily.mean(),
            "viajes_dia_habil": weekday.mean(),
            "viajes_dia_fds": weekend.mean(),
            "dia_max": str(daily.idxmax()),
            "viajes_dia_max": daily.max(),
            "dia_min": str(daily.idxmin()),
            "viajes_dia_min": daily.min(),
            "duracion_mediana": t["duration_min"].median(),
            "duracion_media": t["duration_min"].mean(),
            "duracion_p95": t["duration_min"].quantile(0.95),
            "pct_mujeres": (t["gender"] == "F").mean() * 100,
            "pct_hombres": (t["gender"] == "M").mean() * 100,
            "edad_mediana": t["age"].median(),
            "pct_viajes_circulares": t["is_round_trip"].mean() * 100,
            "viajes_por_usuario_mediana": t.groupby("user_id").size().median(),
        }

    def duration_table(self) -> pd.DataFrame:
        return (
            self.trips["duration_min"]
            .describe(percentiles=[0.25, 0.5, 0.75, 0.95, 0.99])
            .round(2)
            .to_frame("duracion_min")
        )

    # Figuras
    def plot_daily(self) -> Path:
        t = self.trips
        daily = t.groupby(t["start_time"].dt.floor("D")).size()
        fig, ax = plt.subplots(figsize=(9, 3.6))
        ax.plot(daily.index, daily.values, color=BLUE, lw=2)
        weekend = daily[daily.index.dayofweek >= 5]
        ax.scatter(
            weekend.index,
            weekend.values,
            color=BLUE,
            s=22,
            zorder=3,
            edgecolor="white",
            linewidth=1.5,
            label="Sábado o domingo",
        )
        hol = pd.Timestamp("2026-09-16")
        if hol in daily.index:
            ax.annotate(
                "16 sep (día festivo)",
                (hol, daily[hol]),
                xytext=(0, -28),
                textcoords="offset points",
                ha="center",
                color=MUTED,
                fontsize=9,
                arrowprops={"arrowstyle": "-", "color": MUTED},
            )
        ax.set_title("Viajes por día, julio a septiembre de 2026")
        ax.set_ylabel("Viajes")
        ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
        ax.legend(loc="upper left")
        return self._save(fig, "fig01_viajes_diarios.png")

    def plot_hourly_profile(self) -> Path:
        t = self.trips
        day = t["start_time"].dt.floor("D")
        kind = np.where(t["start_time"].dt.dayofweek >= 5, "Fin de semana", "Día hábil")
        counts = t.groupby([kind, day, t["start_time"].dt.hour]).size()
        prof = counts.groupby(level=[0, 2]).mean().unstack(0)
        fig, ax = plt.subplots(figsize=(9, 3.6))
        for col, color in zip(["Día hábil", "Fin de semana"], [BLUE, ORANGE]):
            ax.plot(prof.index, prof[col], color=color, lw=2, label=col)
        ax.set_xticks(range(0, 24, 2))
        ax.set_title("Viajes promedio por hora de inicio")
        ax.set_xlabel("Hora del día")
        ax.set_ylabel("Viajes por hora")
        ax.legend(loc="upper left")
        return self._save(fig, "fig02_perfil_horario.png")

    def plot_duration(self, raw_duration: pd.Series) -> Path:
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
        bins = np.arange(0, 61, 1)
        axes[0].hist(raw_duration.clip(upper=60), bins=bins, color=MUTED, rwidth=0.85)
        axes[0].set_title("Antes de limpiar")
        axes[1].hist(self.trips["duration_min"].clip(upper=60), bins=bins, color=BLUE, rwidth=0.85)
        axes[1].set_title("Después de limpiar")
        med = self.trips["duration_min"].median()
        axes[1].axvline(med, color=TEXT, lw=1, ls="--")
        axes[1].text(
            med + 1, axes[1].get_ylim()[1] * 0.9, f"mediana {med:.1f} min", color=TEXT, fontsize=9
        )
        for ax in axes:
            ax.set_xlabel("Duración (min, recortada en 60)")
            ax.yaxis.set_major_formatter(matplotlib.ticker.StrMethodFormatter("{x:,.0f}"))
        axes[0].set_ylabel("Viajes")
        return self._save(fig, "fig03_duracion.png")

    def plot_heatmap(self) -> Path:
        t = self.trips
        n_days = t.groupby(t["start_time"].dt.dayofweek)["start_time"].apply(
            lambda s: s.dt.date.nunique()
        )
        mat = t.groupby([t["start_time"].dt.dayofweek, t["start_time"].dt.hour]).size()
        mat = mat.unstack(fill_value=0).div(n_days, axis=0)
        fig, ax = plt.subplots(figsize=(9, 3.2))
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
            "azul", ["#f0efec", "#9ec5f4", "#3987e5", "#1c5cab", "#0d366b"]
        )
        im = ax.imshow(mat.values, aspect="auto", cmap=cmap)
        ax.set_yticks(range(7), DAYS)
        ax.set_xticks(range(0, 24, 2))
        ax.set_xlabel("Hora del día")
        ax.grid(False)
        ax.set_title("Viajes promedio por día de la semana y hora")
        cb = fig.colorbar(im, ax=ax, pad=0.01)
        cb.outline.set_visible(False)
        cb.set_label("Viajes por hora", color=MUTED)
        return self._save(fig, "fig04_heatmap_dia_hora.png")

    def plot_top_stations(self, n: int = 15) -> Path:
        top = self.trips["origin_id"].value_counts().head(n)
        names = self.stations.set_index("station_id")["station_name"]
        labels = [str(names.get(i, i))[:38] for i in top.index]
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.barh(labels[::-1], top.values[::-1], color=BLUE, height=0.7)
        for y, v in enumerate(top.values[::-1]):
            ax.text(v, y, f" {v:,}", va="center", fontsize=8, color=MUTED)
        ax.set_title(f"Las {n} estaciones con más salidas")
        ax.set_xlabel("Viajes iniciados")
        ax.grid(axis="y", visible=False)
        return self._save(fig, "fig05_top_estaciones.png")

    def plot_demographics(self) -> Path:
        t = self.trips[self.trips["gender"].isin(["M", "F"])]
        tab = pd.crosstab(t["age_group"], t["gender"], normalize="columns") * 100
        x = np.arange(len(tab))
        fig, ax = plt.subplots(figsize=(9, 3.4))
        ax.bar(x - 0.19, tab["M"], width=0.36, color=BLUE, label="Hombres")
        ax.bar(x + 0.19, tab["F"], width=0.36, color=ORANGE, label="Mujeres")
        ax.set_xticks(x, tab.index.astype(str))
        ax.set_xlabel("Grupo de edad")
        ax.set_ylabel("Porcentaje de viajes")
        ax.set_title("Distribución de viajes por edad y género")
        ax.legend()
        return self._save(fig, "fig06_demografia.png")

    def plot_net_flow(self, n: int = 10) -> Path:
        t = self.trips
        net = t["destination_id"].value_counts().sub(t["origin_id"].value_counts(), fill_value=0)
        n_days = t["start_time"].dt.date.nunique()
        net = (net / n_days).sort_values()
        sel = pd.concat([net.head(n), net.tail(n)])
        names = self.stations.set_index("station_id")["station_name"]
        labels = [str(names.get(i, i))[:34] for i in sel.index]
        colors = ["#e34948" if v < 0 else BLUE for v in sel.values]
        fig, ax = plt.subplots(figsize=(9, 5.4))
        ax.barh(labels, sel.values, color=colors, height=0.7)
        ax.axvline(0, color=MUTED, lw=1)
        ax.set_title("Balance neto diario por estación (llegadas menos salidas)")
        ax.set_xlabel("Bicicletas por día. Negativo: la estación se vacía. Positivo: se llena")
        ax.grid(axis="y", visible=False)
        return self._save(fig, "fig07_balance_neto.png")

    def plot_station_map(self, clusters: pd.DataFrame) -> Path:
        dep = self.trips["origin_id"].value_counts().rename("salidas")
        st = self.stations.merge(clusters, on="station_id", how="inner").merge(
            dep, left_on="station_id", right_index=True
        )
        fig, ax = plt.subplots(figsize=(7.5, 6.5))
        for k in sorted(st["station_cluster"].unique()):
            s = st[st["station_cluster"] == k]
            ax.scatter(
                s["lon"],
                s["lat"],
                s=s["salidas"] / st["salidas"].max() * 160 + 8,
                color=CATEGORICAL[k],
                marker=MARKERS[k],
                alpha=0.85,
                edgecolor="white",
                linewidth=0.8,
                label=f"Grupo {k}",
            )
        ax.set_title("Estaciones por grupo de perfil horario (tamaño = salidas)")
        ax.set_xlabel("Longitud")
        ax.set_ylabel("Latitud")
        ax.set_aspect("equal")
        ax.legend(loc="upper right", markerscale=0.8)
        return self._save(fig, "fig08_mapa_grupos.png")

    def plot_cluster_profiles(self, profiles: pd.DataFrame, clusters: pd.DataFrame) -> Path:
        df = profiles.join(clusters.set_index("station_id")["station_cluster"], how="inner")
        mean = df.groupby("station_cluster").mean()
        k_values = list(mean.index)
        fig, axes = plt.subplots(1, len(k_values), figsize=(10, 2.8), sharey=True)
        hours = np.arange(24)
        for ax, k in zip(np.atleast_1d(axes), k_values):
            habil = mean.loc[k, [f"habil_{h:02d}" for h in hours]].values * 100
            fds = mean.loc[k, [f"fds_{h:02d}" for h in hours]].values * 100
            ax.plot(hours, habil, color=BLUE, lw=2, label="Hábil")
            ax.plot(hours, fds, color=ORANGE, lw=2, label="Fin de semana")
            n = int((df["station_cluster"] == k).sum())
            ax.set_title(f"Grupo {k} ({n} est.)", fontsize=10)
            ax.set_xticks([0, 6, 12, 18])
            ax.set_xlabel("Hora")
        np.atleast_1d(axes)[0].set_ylabel("% de salidas del día")
        np.atleast_1d(axes)[0].legend(fontsize=8, loc="upper left")
        return self._save(fig, "fig09_perfiles_grupos.png")

    def plot_correlation(self, panel: pd.DataFrame, columns: list[str]) -> Path:
        corr = panel[columns].corr()
        fig, ax = plt.subplots(figsize=(7.5, 6))
        cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
            "div", ["#e34948", "#f0efec", "#2a78d6"]
        )
        im = ax.imshow(corr.values, cmap=cmap, vmin=-1, vmax=1)
        ax.set_xticks(range(len(columns)), columns, rotation=60, ha="right", fontsize=8)
        ax.set_yticks(range(len(columns)), columns, fontsize=8)
        for i in range(len(columns)):
            for j in range(len(columns)):
                if abs(corr.values[i, j]) >= 0.3 and i != j:
                    ax.text(
                        j,
                        i,
                        f"{corr.values[i, j]:.2f}",
                        ha="center",
                        va="center",
                        fontsize=6.5,
                        color=TEXT,
                    )
        ax.grid(False)
        ax.set_title("Correlación entre variables del panel estación-hora")
        cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
        cb.outline.set_visible(False)
        return self._save(fig, "fig10_correlacion.png")

    def plot_pca_variance(self, ratios: np.ndarray, silhouettes: dict) -> Path:
        fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
        x = np.arange(1, len(ratios) + 1)
        axes[0].bar(x, ratios * 100, color=BLUE, width=0.6)
        axes[0].plot(x, np.cumsum(ratios) * 100, color=TEXT, lw=1.5, marker="o", ms=5)
        axes[0].set_xticks(x, [f"PC{i}" for i in x])
        axes[0].set_ylabel("% varianza explicada")
        axes[0].set_title("PCA del perfil horario")
        ks = list(silhouettes)
        axes[1].plot(ks, list(silhouettes.values()), color=BLUE, lw=2, marker="o", ms=6)
        axes[1].set_xlabel("Número de grupos (k)")
        axes[1].set_ylabel("Coeficiente de silueta")
        axes[1].set_title("Selección de k en KMeans")
        return self._save(fig, "fig11_pca_silueta.png")
