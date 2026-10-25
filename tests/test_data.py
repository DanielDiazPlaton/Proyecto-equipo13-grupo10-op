"""Pruebas unitarias de las reglas de limpieza y del versionado."""

import pandas as pd

from mibici.cleaning import TripCleaner
from mibici.config import CleaningConfig
from mibici.versioning import dataframe_fingerprint

STATIONS = pd.DataFrame({"station_id": [1, 2, 3]})


def make_trips() -> pd.DataFrame:
    rows = [
        # id, usuario, genero, nacimiento, inicio, fin, origen, destino
        (1, 10, "M", 1990, "2026-09-01 08:00:00", "2026-09-01 08:10:00", 1, 2),   # valido
        (2, 11, "F", 1985, "2026-09-01 08:00:00", "2026-09-01 08:00:30", 1, 2),   # < 1 min
        (3, 12, "M", 1995, "2026-09-01 09:00:00", "2026-09-01 09:01:30", 3, 3),   # misma est.
        (4, 13, "M", 2000, "2026-09-01 10:00:00", "2026-09-01 15:00:00", 1, 3),   # > 180 min
        (5, 14, "F", 199, "2026-09-01 11:00:00", "2026-09-01 11:20:00", 2, 99),   # est. fuera
        (6, 15, None, 199, "2026-09-01 12:00:00", "2026-09-01 12:15:00", 2, 3),   # demografia
        (1, 10, "M", 1990, "2026-09-01 08:00:00", "2026-09-01 08:10:00", 1, 2),   # duplicado
    ]
    cols = ["trip_id", "user_id", "gender", "birth_year", "start_time", "end_time",
            "origin_id", "destination_id"]
    df = pd.DataFrame(rows, columns=cols)
    df["gender"] = df["gender"].astype("string")
    df["birth_year"] = df["birth_year"].astype(float)
    df["source_month"] = "2026_09"
    return df


def test_clean_removes_invalid_trips():
    cleaner = TripCleaner(STATIONS, CleaningConfig())
    clean = cleaner.clean(make_trips())
    assert sorted(clean["trip_id"]) == [1, 6]


def test_demographic_errors_do_not_drop_rows():
    clean = TripCleaner(STATIONS).clean(make_trips())
    row = clean[clean["trip_id"] == 6].iloc[0]
    assert row["gender"] == "NE"
    assert pd.isna(row["birth_year"])


def test_report_counts_match():
    cleaner = TripCleaner(STATIONS)
    trips = make_trips()
    clean = cleaner.clean(trips)
    report = cleaner.report_frame()
    assert report["filas_eliminadas"].sum() == len(trips) - len(clean)


def test_fingerprint_changes_with_data():
    df = make_trips()
    other = df.copy()
    other.loc[0, "origin_id"] = 3
    assert dataframe_fingerprint(df) == dataframe_fingerprint(df.copy())
    assert dataframe_fingerprint(df) != dataframe_fingerprint(other)
