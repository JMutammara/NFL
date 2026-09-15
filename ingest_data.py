#!/usr/bin/env python
"""Stage 1 — Data ingestion.

Downloads (or refreshes) every nflverse asset the pipeline needs and prints a
verification report: row counts, season coverage, key-column integrity and a
few sanity checks on the play-by-play data.

    python ingest_data.py                # cached where possible
    python ingest_data.py --refresh      # force re-download of current season
    python ingest_data.py --seasons 2022 2023 2024
"""
from __future__ import annotations

import warnings

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
try:
    import pandas as _pd

    warnings.filterwarnings("ignore", category=_pd.errors.PerformanceWarning)
except Exception:  # pragma: no cover
    pass

import argparse

import numpy as np
import pandas as pd

from nfl_pipeline.config import load_config
from nfl_pipeline.ingest import ingest_all, ingestion_report
from nfl_pipeline.utils import log


def verify(data: dict[str, pd.DataFrame], cfg) -> bool:
    ok = True
    pbp, sch = data["pbp"], data["schedules"]
    print("\n=== INGESTION REPORT ===")
    print(ingestion_report(data).to_string(index=False))

    print("\n=== PLAY-BY-PLAY SANITY ===")
    games_pbp = pbp.groupby("season")["game_id"].nunique()
    done = sch[sch["home_score"].notna() & sch["season"].isin(pbp["season"].unique())]
    games_sch = done.groupby("season")["game_id"].nunique()
    cmp = pd.DataFrame({"pbp_games": games_pbp, "sched_completed": games_sch}).fillna(0).astype(int)
    cmp["diff"] = cmp["pbp_games"] - cmp["sched_completed"]
    print(cmp.to_string())
    if (cmp["diff"].abs() > 2).any():
        log.warning("pbp / schedule game counts disagree by >2 in at least one season")
        ok = False

    plays = pbp[(pbp["pass"] == 1) | (pbp["rush"] == 1)]
    print(f"\nscrimmage plays: {len(plays):,}   epa null: {plays['epa'].isna().mean():.3%}   "
          f"success null: {plays['success'].isna().mean():.3%}")
    print(f"pass plays with cpoe: {plays.loc[plays['pass'] == 1, 'cpoe'].notna().mean():.1%}   "
          f"epa/play mean: {plays['epa'].mean():+.4f} (should be ~0)")
    if abs(plays["epa"].mean()) > 0.05:
        log.warning("league-average EPA/play is far from zero; check play filtering")
        ok = False
    dup = pbp.duplicated(["game_id", "play_id"]).sum()
    print(f"duplicate (game_id, play_id): {dup}")
    ok &= dup == 0

    print("\n=== SCHEDULE / CURRENT SEASON ===")
    cur = sch[sch["season"] == cfg.current_season]
    done = cur[cur["home_score"].notna()]
    upcoming = cur[cur["home_score"].isna()]
    print(f"season {cfg.current_season}: {len(done)} games played (weeks {sorted(done['week'].unique().tolist())}), "
          f"{len(upcoming)} scheduled; next week = {int(upcoming['week'].min()) if len(upcoming) else 'n/a'}")
    print(f"lines available for next week: spread {upcoming[upcoming['week'] == upcoming['week'].min()]['spread_line'].notna().mean():.0%}, "
          f"total {upcoming[upcoming['week'] == upcoming['week'].min()]['total_line'].notna().mean():.0%}")

    print("\n=== KEY INTEGRITY ===")
    ws = data["weekly_stats"]
    print(f"weekly_stats seasons {ws['season'].min()}-{ws['season'].max()}, players {ws['player_id'].nunique():,}, "
          f"skill rows {ws['position'].isin(cfg.get('features.player_positions')).sum():,}")
    print(f"weekly_stats target_share null (WR/TE/RB): "
          f"{ws.loc[ws['position'].isin(['WR', 'TE', 'RB']), 'target_share'].isna().mean():.1%}")
    ro = data["rosters"]
    print(f"rosters gsis_id null: {ro['gsis_id'].isna().mean():.1%}  pfr_id null: {ro['pfr_id'].isna().mean():.1%}")
    part = data["participation"]
    if len(part):
        print(f"participation: was_pressure null {part['was_pressure'].isna().mean():.1%}, "
              f"time_to_throw null {part['time_to_throw'].isna().mean():.1%}")
    print("\nVERDICT:", "OK" if ok else "CHECK WARNINGS ABOVE")
    return ok


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refresh", action="store_true", help="re-download the current season even if cached")
    ap.add_argument("--seasons", nargs="*", type=int, default=None)
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    data = ingest_all(cfg, refresh=args.refresh, seasons=args.seasons)
    verify(data, cfg)


if __name__ == "__main__":
    main()
