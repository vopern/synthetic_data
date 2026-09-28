"""Generated data: ask each model, under each prompt, for actual times of every sampled trip.

The model gets the trip skeleton (type, train number, date, stations with scheduled times) and returns CSV. One call
per trip and arm (model x prompt), appended to data/synthetic/generations.jsonl; reruns skip calls already made.
"""
import argparse
import asyncio
import json
from pathlib import Path

import pandas as pd

from pipeline.llm import complete

OUT = Path("data/synthetic/generations.jsonl")
SYSTEM = "You write operational records. Output only the requested CSV: no commentary, no code fences."
NEUTRAL = "Generate realistic operational records for this train service."
PROMPTS = {"neutral": NEUTRAL, "instructed": NEUTRAL + " Delays carry over from one stop to the next."}
FORMAT = ("Output CSV with the header station,actual_arrival,actual_departure and one row per stop in the same "
          "order. Times as HH:MM. Leave a field empty where the schedule has no time.")


def render(trip: pd.Series, stops: pd.DataFrame, prompt: str) -> str:
    """User prompt: instruction, skeleton, output format."""
    rows = "\n".join(f"{s.station},{s.arr or ''},{s.dep or ''}" for s in stops.itertuples())
    return (f"{PROMPTS[prompt]}\n\nService: {trip.type} {trip.train}, {trip.date}\n"
            f"Stops with scheduled times:\nstation,scheduled_arrival,scheduled_departure\n{rows}\n\n{FORMAT}")


def done_keys() -> set:
    if not OUT.exists():
        return set()
    return {(r["trip_id"], r["model"], r["prompt"]) for r in map(json.loads, OUT.open())}


async def run(start: int, n: int, models: list[str], prompts: list[str], concurrency: int):
    trips = pd.read_parquet("data/real/trips.parquet")
    trips = trips[(trips.sample_rank >= start) & (trips.sample_rank < start + n)]
    stops = pd.read_parquet("data/real/stops.parquet").replace({float("nan"): None})
    by_trip = dict(list(stops.groupby("trip_id")))
    done = done_keys()
    jobs = [(t, m, p) for t in trips.itertuples() for m in models for p in prompts
            if (int(t.trip_id), m, p) not in done]
    print(f"{len(jobs)} calls to make")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    sem = asyncio.Semaphore(concurrency)

    async def one(trip, model, prompt):
        async with sem:
            c = await complete(render(trip, by_trip[trip.trip_id], prompt), SYSTEM, model)
        if c.is_error:
            print("error:", trip.trip_id, model, prompt)
            return
        with OUT.open("a") as f:
            f.write(json.dumps({"trip_id": int(trip.trip_id), "model": model, "prompt": prompt,
                                "served_by": c.model, "text": c.text, "input_tokens": c.input_tokens,
                                "output_tokens": c.output_tokens, "duration_s": c.duration_s}) + "\n")

    await asyncio.gather(*[one(*j) for j in jobs])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", type=int, default=30, help="first sample_rank; ranks < 30 are the pilot")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--models", nargs="+", default=["haiku", "sonnet"])
    ap.add_argument("--prompts", nargs="+", default=["neutral", "instructed"])
    ap.add_argument("--concurrency", type=int, default=6)
    a = ap.parse_args()
    asyncio.run(run(a.start, a.n, a.models, a.prompts, a.concurrency))
