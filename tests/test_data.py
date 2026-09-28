import numpy as np
import pandas as pd
import pytest

from pipeline.data import assemble, complete_trips, trip_vector
from pipeline.validate import Invalid, parse


def raw(rows):
    cols = ["trip_id", "date", "type", "train", "code", "station", "arr_ts", "arr_delay", "dep_ts", "dep_delay"]
    return pd.DataFrame(rows, columns=cols)


def stop(code, arr, dep, arr_delay=0, dep_delay=0, trip=1, day="2025-06-01"):
    ts = lambda t: None if t is None else f"{t}:00+02:00"
    return [trip, day, "Sprinter", 1234, code, code, ts(arr), arr_delay if arr else None, ts(dep),
            dep_delay if dep else None]


def test_orders_by_scheduled_time_not_file_order():
    s = assemble(raw([stop("A", None, "2025-06-01T10:00"), stop("C", "2025-06-01T10:20", None),
                      stop("B", "2025-06-01T10:09", "2025-06-01T10:10")]))
    assert s.code.tolist() == ["A", "B", "C"]
    assert s.seq.tolist() == [0, 1, 2]
    assert s.dep.tolist()[:2] == ["10:00", "10:10"]


def test_overnight_service_keeps_order():
    s = assemble(raw([stop("B", "2025-06-02T00:05", "2025-06-02T00:06"), stop("A", None, "2025-06-01T23:55"),
                      stop("C", "2025-06-02T00:15", None)]))
    assert s.code.tolist() == ["A", "B", "C"]


def test_trip_vector_uses_arrival_at_last_stop():
    assert trip_vector(np.array([1, 2, 3, np.nan]), np.array([np.nan, 2, 4, 5])).tolist() == [1, 2, 3, 5]


def test_complete_trips_needs_five_stops_and_delays():
    times = [(None, "10:00"), ("10:05", "10:06"), ("10:10", "10:11"), ("10:15", "10:16"), ("10:20", None)]
    rows = [stop(str(i), a and f"2025-06-01T{a}", d and f"2025-06-01T{d}") for i, (a, d) in enumerate(times)]
    short = [stop(str(i), a and f"2025-06-01T{a}", d and f"2025-06-01T{d}", trip=2) for i, (a, d) in
             enumerate(times[1:])]
    missing = [r[:] for r in rows]
    for r in missing:
        r[0] = 3
    missing[2][9] = None
    s = assemble(raw(rows + short + missing))
    assert complete_trips(s).tolist() == [1]


def skeleton(times):
    return pd.DataFrame({"station": [f"S{i}" for i in range(len(times))],
                         "arr": [a for a, _ in times], "dep": [d for _, d in times]})


def test_parse_delays_across_midnight():
    stops = skeleton([(None, "23:58"), ("00:03", "00:04"), ("00:10", None)])
    text = "station,actual_arrival,actual_departure\nS0,,23:59\nS1,00:05,00:06\nS2,00:09,\n"
    assert parse(text, stops).tolist() == [1, 2, -1]


@pytest.mark.parametrize("text", [
    "S0,,10:00\nS1,10:05,10:06",               # missing row
    "S0,,10:00\nS2,10:05,10:06\nS1,10:10,",    # wrong order
    "S0,,10:00\nS1,10:05:30,10:06\nS2,10:10,", # seconds
    "S0,,10:00\nS1,10:07,10:06\nS2,10:10,",    # departs before arrival
])
def test_parse_rejects(text):
    with pytest.raises(Invalid):
        parse(text, skeleton([(None, "10:00"), ("10:05", "10:06"), ("10:10", None)]))
