"""Validity: parse a generated CSV against its trip skeleton into a trip vector of delays, or raise Invalid.

Invalid if the row count, field count or station names differ from the skeleton, a needed time is not HH:MM, or the
train departs before it arrives. Delay is actual minus scheduled time, wrapped across midnight.
"""
import csv
import io
import re

import numpy as np
import pandas as pd

from pipeline.data import trip_vector

HHMM = re.compile(r"^([01]\d|2[0-3]):[0-5]\d$")


class Invalid(Exception):
    pass


def minutes(t: str) -> int:
    return int(t[:2]) * 60 + int(t[3:])


def wrap(d: int) -> int:
    """Minute difference across midnight, in [-720, 720)."""
    return (d + 720) % 1440 - 720


def parse_time(value: str, what: str) -> int:
    value = value.strip()
    if not HHMM.match(value):
        raise Invalid(f"bad {what} time {value!r}")
    return minutes(value)


def parse(text: str, stops: pd.DataFrame) -> np.ndarray:
    """Trip vector of generated delays. stops: the trip's rows from stops.parquet in order."""
    lines = [l for l in text.strip().splitlines() if l.strip() and not l.strip().startswith("```")]
    rows = list(csv.reader(io.StringIO("\n".join(lines))))
    if rows and rows[0] and rows[0][0].strip().lower() == "station":
        rows = rows[1:]
    if len(rows) != len(stops):
        raise Invalid(f"{len(rows)} rows for {len(stops)} stops")
    dep_delay, arr_delay = np.zeros(len(stops), int), np.zeros(len(stops), int)
    for i, (row, s) in enumerate(zip(rows, stops.itertuples())):
        if len(row) != 3:
            raise Invalid(f"row {i} has {len(row)} fields")
        if row[0].strip() != s.station:
            raise Invalid(f"row {i} is {row[0]!r}, expected {s.station!r}")
        arr = parse_time(row[1], "arrival") if pd.notna(s.arr) else None
        dep = parse_time(row[2], "departure") if pd.notna(s.dep) else None
        if arr is not None:
            arr_delay[i] = wrap(arr - minutes(s.arr))
        if dep is not None:
            dep_delay[i] = wrap(dep - minutes(s.dep))
        if arr is not None and dep is not None and wrap(dep - arr) < 0:
            raise Invalid(f"row {i} departs before it arrives")
    return trip_vector(dep_delay, arr_delay)
