#!/usr/bin/env python
"""Stage 2 — Feature engineering.

Builds the team-game efficiency table, rolls it into leakage-safe windows,
assembles the game-level and player-level feature matrices, and prints a
summary of feature families and data dimensions for approval.

    python build_features.py             # build + summary (no approval)
    python build_features.py --approve   # build + write the APPROVED marker
    python build_features.py --summary   # re-print the summary from cached artifacts

Training (train_models.py) refuses to run until the APPROVED marker exists
or it is invoked with --approve-features.
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
import json
import time
from datetime import datetime

import pandas as pd

from nfl_pipeline.config import APPROVAL_FILE, FEATURES_DIR, PROCESSED_DIR, load_config
from nfl_pipeline.features.game_features import build_game_features, roll_team_games, save_game_features
from nfl_pipeline.features.player_features import build_player_features, save_player_features
from nfl_pipeline.features.players import build_player_games
from nfl_pipeline.features.team_efficiency import build_team_game_stats
from nfl_pipeline.ingest import ingest_all, normalize_depth_charts
from nfl_pipeline.utils import log, timed


def build(cfg, refresh: bool = False) -> tuple[pd.DataFrame, dict, pd.DataFrame, dict]:
    data = ingest_all(cfg, refresh=refresh)
    sch = data["schedules"]
    sch = sch[sch["season"].isin(cfg.seasons)].reset_index(drop=True)
    with timed("team-game efficiency table"):
        tg = build_team_game_stats(data["pbp"], sch, cfg, data["participation"], data["ftn"], data["pfr_pass"], data["pfr_def"])
        tg.to_parquet(PROCESSED_DIR / "team_game_stats.parquet", index=False)
    with timed("team rolling / adjusted / QB features"):
        tgr, families = roll_team_games(tg, cfg)
        tgr.to_parquet(PROCESSED_DIR / "team_game_rolled.parquet", index=False)
    with timed("game feature matrix"):
        gf, gman = build_game_features(tgr, sch, cfg, families)
        save_game_features(gf, gman)
    with timed("player-game table"):
        depth = normalize_depth_charts(data["depth_charts"], sch)
        pg, dvp = build_player_games(data["weekly_stats"], data["pbp"], sch, data["rosters"], data["players"],
                                     data["snap_counts"], data["injuries"], depth, cfg)
        pg.to_parquet(PROCESSED_DIR / "player_games.parquet", index=False)
        dvp.to_parquet(PROCESSED_DIR / "defense_vs_position.parquet", index=False)
    with timed("player feature matrix"):
        pf, pman = build_player_features(pg, dvp, tgr, gf, cfg)
        save_player_features(pf, pman)
    return gf, gman, pf, pman


def summarize(gf: pd.DataFrame, gman: dict, pf: pd.DataFrame, pman: dict, cfg) -> str:
    lines = []
    p = lines.append
    p("=" * 78)
    p("FEATURE ENGINEERING SUMMARY  (generated %s)" % datetime.now().strftime("%Y-%m-%d %H:%M"))
    p("=" * 78)
    played = gf[gf["played"] == 1]
    upcoming = gf[gf["played"] == 0]
    p("\n[GAME-LEVEL MATRIX]  data/features/game_features.parquet")
    p(f"  rows            : {len(gf):,} games  ({len(played):,} played for training, {len(upcoming):,} upcoming)")
    p(f"  seasons         : {int(gf['season'].min())}-{int(gf['season'].max())}   (postseason included: {cfg.get('data.include_postseason')})")
    p(f"  features        : {len(gman['features']):,}")
    p(f"  targets         : {', '.join(gman['target_cols'])}")
    p(f"  market coverage : spread {played['spread_line'].notna().mean():.1%}, total {played['total_line'].notna().mean():.1%}, "
      f"moneyline {played['home_ml_prob_novig'].notna().mean():.1%}")
    p(f"  target stats    : margin mean {played['home_margin'].mean():+.2f} sd {played['home_margin'].std():.2f} | "
      f"total mean {played['total_points'].mean():.1f} sd {played['total_points'].std():.1f} | home win {played['home_win'].mean():.1%}")
    p(f"  market baseline : spread MAE {(played['home_margin'] - played['spread_line']).abs().mean():.2f}, "
      f"total MAE {(played['total_points'] - played['total_line']).abs().mean():.2f}")
    p("  families:")
    for fam, cols in gman["families"].items():
        miss = gf.loc[gf['played'] == 1, cols].isna().mean().mean() if cols else 0
        ex = ", ".join(cols[:4]) + (" ..." if len(cols) > 4 else "")
        p(f"    {fam:<14} {len(cols):>4}  (avg missing {miss:5.1%})  e.g. {ex}")
    nxt = upcoming[upcoming["season"] == cfg.current_season]
    if len(nxt):
        p(f"  next slate      : season {cfg.current_season} week {int(nxt['week'].min())} -> {int((nxt['week'] == nxt['week'].min()).sum())} games")

    pp = pf[pf["is_projection"] == 0]
    p("\n[PLAYER-LEVEL MATRIX]  data/features/player_features.parquet")
    p(f"  rows            : {len(pf):,} player-games  ({len(pp):,} played, {int((pf['is_projection'] == 1).sum()):,} projection rows)")
    p(f"  players         : {pf['player_id'].nunique():,}   by position: " + ", ".join(f"{k} {v:,}" for k, v in pp['position'].value_counts().items()))
    p(f"  features        : {len(pman['features']):,}")
    p(f"  targets         : {', '.join(pman['target_cols'])}  (aux: {', '.join(pman['aux_targets'])})")
    for t in pman["target_cols"]:
        s = pp[t]
        p(f"    {t:<16} mean {s.mean():6.2f}  sd {s.std():6.2f}  zero-share {(s == 0).mean():5.1%}")
    p("  families:")
    for fam, cols in pman["families"].items():
        miss = pp[cols].isna().mean().mean() if cols else 0
        ex = ", ".join(cols[:4]) + (" ..." if len(cols) > 4 else "")
        p(f"    {fam:<14} {len(cols):>4}  (avg missing {miss:5.1%})  e.g. {ex}")
    p("\n[LEAKAGE GUARDS]")
    p("  * every rolling / EWM feature is shifted one game before windowing (game never sees its own stats)")
    p("  * opponent adjustment uses the opponent's prior EWM and an expanding league mean")
    p("  * season-start EWM state = last season's final EWM shrunk toward last season's league/position mean")
    p("  * rolling-origin CV in train_models.py trains only on (season, week) strictly before each validation block")
    p("\nNEXT STEP: review the families above. To approve and train:")
    p("    python build_features.py --approve        # writes data/features/APPROVED")
    p("    python train_models.py                    # or: python train_models.py --approve-features")
    p("=" * 78)
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refresh", action="store_true", help="re-download current-season raw data first")
    ap.add_argument("--approve", action="store_true", help="write the APPROVED marker after building")
    ap.add_argument("--summary", action="store_true", help="print the summary from cached feature files (no rebuild)")
    ap.add_argument("--config", default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    t0 = time.time()
    if args.summary:
        gf = pd.read_parquet(FEATURES_DIR / "game_features.parquet")
        pf = pd.read_parquet(FEATURES_DIR / "player_features.parquet")
        gman = json.load(open(FEATURES_DIR / "game_manifest.json"))
        pman = json.load(open(FEATURES_DIR / "player_manifest.json"))
    else:
        gf, gman, pf, pman = build(cfg, refresh=args.refresh)
        log.info("feature build complete in %.1fs", time.time() - t0)
    text = summarize(gf, gman, pf, pman, cfg)
    print(text)
    (FEATURES_DIR / "feature_summary.md").write_text("```\n" + text + "\n```\n")
    if args.approve:
        APPROVAL_FILE.write_text(datetime.now().isoformat())
        print(f"\nAPPROVED marker written -> {APPROVAL_FILE}")


if __name__ == "__main__":
    main()
