"""Real data: download the Rijden de Treinen archive (NS, June 2025) and sample the trips to generate from.

Writes data/real/trips.parquet (330 trips, Intercity and Sprinter alternating by sample_rank; ranks 0-29 pilot,
30-329 main) and data/real/stops.parquet (their stops with scheduled times and real delays).
"""
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

URL = "https://opendata.rijdendetreinen.nl/public/services/services-2025-06.csv.gz"
RAW = Path("data/raw/services-2025-06.csv.gz")
REAL = Path("data/real")
TYPES = ["Intercity", "Sprinter"]
MIN_STOPS = 5
N_SAMPLE = 330  # ranks 0-29 pilot, 30-329 main
SEED = 0

COLS = {
    "Service:RDT-ID": "trip_id", "Service:Date": "date", "Service:Type": "type", "Service:Train number": "train",
    "Stop:Station code": "code", "Stop:Station name": "station", "Stop:Arrival time": "arr_ts",
    "Stop:Arrival delay": "arr_delay", "Stop:Departure time": "dep_ts", "Stop:Departure delay": "dep_delay",
}


def load_stops(path: Path = RAW) -> pd.DataFrame:
    """Stops of uncancelled NS Intercity and Sprinter trips, ordered by scheduled time within each trip."""
    df = pd.read_csv(path)
    df = df[(df["Service:Company"] == "NS") & df["Service:Type"].isin(TYPES)
            & ~df["Service:Completely cancelled"] & ~df["Service:Partly cancelled"]]
    return assemble(df.rename(columns=COLS)[list(COLS.values())])


def assemble(df: pd.DataFrame) -> pd.DataFrame:
    """Order stops by scheduled timestamp (stop IDs are not always in order) and add seq and HH:MM times."""
    df = df.copy()
    df["t"] = pd.to_datetime(df.dep_ts.fillna(df.arr_ts), utc=True)
    df = df.sort_values(["trip_id", "t"]).drop(columns="t")
    df["seq"] = df.groupby("trip_id").cumcount()
    df["arr"] = df.arr_ts.str[11:16]
    df["dep"] = df.dep_ts.str[11:16]
    return df.reset_index(drop=True)


def trip_vector(dep_delay: np.ndarray, arr_delay: np.ndarray) -> np.ndarray:
    """Departure delay at every stop but the last, arrival delay at the last."""
    return np.append(dep_delay[:-1], arr_delay[-1])


def complete_trips(stops: pd.DataFrame) -> pd.Index:
    """Trips with >= MIN_STOPS stops, times and delays present wherever the trip vector needs them."""
    last = stops.seq == stops.groupby("trip_id").seq.transform("max")
    first = stops.seq == 0
    ok = (last | stops.dep_delay.notna() & stops.dep.notna()) & (~last | stops.arr_delay.notna() & stops.arr.notna())
    ok &= first | stops.arr.notna()
    g = ok.groupby(stops.trip_id)
    return g.all().index[g.all() & (g.size() >= MIN_STOPS)]


def archive_vectors() -> dict:
    """Trip vectors of every complete trip in the archive."""
    stops = load_stops()
    stops = stops[stops.trip_id.isin(complete_trips(stops))]
    return {t: trip_vector(s.dep_delay.to_numpy(), s.arr_delay.to_numpy()).astype(int)
            for t, s in stops.groupby("trip_id")}


def sample(stops: pd.DataFrame) -> pd.DataFrame:
    """N_SAMPLE trips, types alternating by rank, so every prefix is stratified."""
    trips = stops[stops.trip_id.isin(complete_trips(stops))].groupby("trip_id").agg(
        date=("date", "first"), type=("type", "first"), train=("train", "first"), n_stops=("seq", "size"))
    rng = np.random.default_rng(SEED)
    per_type = [trips[trips.type == t].sample(N_SAMPLE // 2, random_state=rng).reset_index() for t in TYPES]
    for i, d in enumerate(per_type):
        d["sample_rank"] = 2 * d.index + i
    return pd.concat(per_type).sort_values("sample_rank").reset_index(drop=True)


def main():
    if not RAW.exists():
        RAW.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(URL, RAW)
    stops = load_stops()
    trips = sample(stops)
    stops = stops[stops.trip_id.isin(trips.trip_id)]
    REAL.mkdir(parents=True, exist_ok=True)
    trips.to_parquet(REAL / "trips.parquet")
    stops[["trip_id", "seq", "code", "station", "arr", "dep", "arr_delay", "dep_delay"]].to_parquet(
        REAL / "stops.parquet")
    print(f"{len(trips)} trips, {len(stops)} stops")
    print(trips.groupby("type").n_stops.describe())


if __name__ == "__main__":
    main()
