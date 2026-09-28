# Notes on Train delays: real vs generated

Do LLM-generated train trips carry delays over between stops as strongly as real ones, and does telling the model that
delays carry over also teach it that trains recover? We take 300 NS trips from the Rijden de Treinen archive (June
2025), give Haiku and Sonnet each trip's skeleton (stops and scheduled times), ask for actual times, and compare delay
correlation between stops and recovery with the real trips.

Paper: `paper/main.pdf`.

## Run

```
uv sync
uv run python -m pytest                          # trip assembly and CSV validation
uv run python -m pipeline.data                   # download archive -> data/real/{trips,stops}.parquet
uv run python -m pipeline.generate --start 0 --n 30     # pilot
uv run python -m pipeline.generate               # main run, ranks 30-329, resumable
uv run python -m pipeline.analyze --start 0 --n 30 --tag pilot
uv run python -m pipeline.analyze                # tables, verdicts, figures -> results/
```

The generations are committed in `data/synthetic/generations.jsonl`, so `pipeline.data` and `pipeline.analyze` reproduce
all results without Claude.

The archive comes from https://opendata.rijdendetreinen.nl (CC BY 4.0). Generation goes through the Claude Agent SDK, so
it needs a logged-in Claude Code install, not an API key.

## Decisions

### Use Case and Data Set: Train delays
- Goal: surface shortcomings in LLMs trained on token prediction in reproducing sequential operational data with complex characteristics
- Dataset of the right shape and size is available
- An agent trained to optimize train schedules is a plausible usecase
- In comparison: evaluating sampling of random numbers from Enron Emails or word distribution in wildchat data
may produce differences, but with less clear implications on operations.
  
### Implementation
- Claude Agent SDK allows to generate train traces using Claude Code subscription without additional cost
- Hypotheses defined before the full generation run
- Bootstrapping for confidence intervals
- Code and most text written by Claude Code
  - Research project execution outlined and planned first
  - Implementation in steps: generation of some pilot examples, full run, analysis and paper writing

### Experiment

Expected story: LLMs miss the obvious structure of delays unless told, and when told, they fix exactly what they're told
and still miss the rest. That would show real data holds structure a prompt writer can't name in advance.

- H1: with a neutral prompt, generated trips have lower delay correlation than real trips. Delay carry-over is physical
  (a late train stays late), and we expected a model writing times stop by stop to lose it.
- H2: with a neutral prompt, generated trips recover less often than real ones. Recovery depends on schedule slack,
  which is invisible in the timetable, so we expected the model to miss it.
- H3: the instructed prompt ("delays carry over") closes the correlation gap but not the recovery gap. Naming one
  property should fix that property. Recovery stands in for structure nobody names, so it should stay wrong.

## Results

- H1 (falsified): without being told, the models already carry delays over from stop to stop, as real trains do.
- H2 (supported): generated trains lose delay less often than real ones.
- H3 (falsified): telling the model "delays carry over" didn't fix one thing and leave the other. It changed both, and
  in different ways for Haiku and Sonnet.

The main point still stands: real data has structure the generators miss. Real delays jump up suddenly and come down
slowly. Without instruction, generated delays barely move in either direction. We didn't expect this, and didn't
prompt for it. Only the real data showed it.

A second point: fixing a known gap with a prompt doesn't work as expected. One added sentence shifted several properties
of the data at once.

Numbers and intervals are in `paper/main.pdf` (Results).

## Limits
- One operator, one month, two models from one family, one prompt pair, default sampling, no experiments with temperature.
- We show the generated data differs, not that the difference matters for a downstream model or agent.
- Fresh run with larger sample size for sharper tests as some metrics were calculated after main run.
- Pooled lag-1 correlation also rewards trips that are late throughout. A within-trip measure might have decided H1
  differently.
- NS live data may record the last forecast, not a measured time, so some sudden rises may be forecast updates.

## Next Steps

- Delay prediction: train a predictor of final delay from the delay at stop k, once on generated and once on real trips,
  and evaluate both on held-out real trips. This is the direct test of whether the gaps matter; a small error gap would
  mean real data only seems to add something here.
- Real data as seed: give the model a few real trips as examples, or fit a simple stochastic delay model (incident jumps
  plus slack-limited recovery) on the archive, and compare both to pure generation.
- Breadth: other operators and months, other model families, and other sequential records (maintenance logs, ticket
  histories) to see whether "sudden up, slow down" dynamics are missed in general.
