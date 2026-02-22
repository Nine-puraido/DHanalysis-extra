"""CLI entry point for modeling pipeline.

Usage:
    python -m dhx.modeling train --feature-set-version 5 --test-cutoff 2026-01-15
    python -m dhx.modeling train --feature-set-version 5 --model-type xgboost
    python -m dhx.modeling train --feature-set-version 5 --model-type ensemble
    python -m dhx.modeling predict --fixture-ids 3,4,504
    python -m dhx.modeling predict-upcoming
    python -m dhx.modeling predict-upcoming --fixture-ids 1156,1157,1158
    python -m dhx.modeling activate --market 1x2 --model-version-id 5
    python -m dhx.modeling list-models
    python -m dhx.modeling strategy-sim --test-cutoff 2026-01-15 --stake 100
    python -m dhx.modeling consensus-sim --test-cutoff 2026-01-15
"""

from __future__ import annotations

import io
import logging
from datetime import UTC, datetime

import polars as pl
import typer
from rich.console import Console
from rich.table import Table

from dhx.db import get_client, insert_rows, select_rows, update_rows

app = typer.Typer(help="DHextra modeling CLI", invoke_without_command=True)
console = Console()


@app.callback()
def _callback() -> None:
    """Modeling pipeline for DHextra (poisson / xgboost / ensemble / contrarian)."""


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _start_pipeline_run(pipeline_name: str, details: dict | None = None) -> int:
    row = {
        "pipeline_name": pipeline_name,
        "started_at": datetime.now(UTC).isoformat(),
        "status": "running",
        "details_json": details or {},
    }
    inserted = insert_rows("pipeline_runs", [row])
    return inserted[0]["id"]


def _end_pipeline_run(run_id: int, status: str, rows_written: int, details: dict) -> None:
    update_rows(
        "pipeline_runs",
        {
            "ended_at": datetime.now(UTC).isoformat(),
            "status": status,
            "rows_written": rows_written,
            "details_json": details,
        },
        {"id": run_id},
    )


def _load_parquet_from_storage(feature_set_version_id: int) -> pl.DataFrame:
    """Load Parquet from Supabase Storage by feature_set_version_id."""
    rows = select_rows(
        "feature_set_versions",
        "id,parquet_path,row_count,training_window",
        filters={"id": feature_set_version_id},
    )
    if not rows:
        raise ValueError(f"feature_set_versions id={feature_set_version_id} not found")

    parquet_path = rows[0]["parquet_path"]
    # parquet_path is "features/feature_matrix/.../file.parquet" — split bucket/path
    parts = parquet_path.split("/", 1)
    bucket = parts[0]
    path = parts[1]

    console.print(f"[dim]Downloading {parquet_path}...[/dim]")
    data = get_client().storage.from_(bucket).download(path)
    df = pl.read_parquet(io.BytesIO(data))
    console.print(f"[dim]Loaded {len(df)} rows x {len(df.columns)} cols[/dim]")
    return df


# ======================================================================
# Commands
# ======================================================================


@app.command()
def train(
    feature_set_version: int = typer.Option(
        ..., "--feature-set-version", help="feature_set_versions.id to use"
    ),
    test_cutoff: str | None = typer.Option(
        None, "--test-cutoff", help="Train/test split date (YYYY-MM-DD). Default: last 20%%"
    ),
    model_type: str = typer.Option(
        "poisson", "--model-type", help="Model type: poisson, xgboost, ensemble, or contrarian"
    ),
    log_level: str = typer.Option("INFO", "--log-level", help="Logging level"),
) -> None:
    """Train a model on the feature matrix."""
    _setup_logging(log_level)

    if model_type not in ("poisson", "xgboost", "ensemble", "contrarian"):
        console.print(f"[red]Unknown model type: {model_type}[/red]")
        raise typer.Exit(1)

    console.print(f"[bold]=== DHextra Model Training ({model_type}) ===[/bold]")
    console.print()

    run_id = _start_pipeline_run(
        "modeling_train",
        {
            "feature_set_version_id": feature_set_version,
            "test_cutoff": test_cutoff,
            "model_type": model_type,
        },
    )
    console.print(f"[dim]Pipeline run id={run_id}[/dim]")

    try:
        # 1. Load data
        console.print("[cyan]1. Loading feature matrix...[/cyan]")
        df = _load_parquet_from_storage(feature_set_version)

        # Ensure kickoff_at is datetime for splitting
        if df["kickoff_at"].dtype == pl.String:
            df = df.with_columns(pl.col("kickoff_at").str.to_datetime(time_zone="UTC"))

        # Sort by date
        df = df.sort("kickoff_at")

        # 2. Train/test split
        console.print("[cyan]2. Splitting train/test...[/cyan]")
        if test_cutoff:
            cutoff_dt = datetime.strptime(test_cutoff, "%Y-%m-%d").replace(
                tzinfo=UTC
            )
            train_df = df.filter(pl.col("kickoff_at") < cutoff_dt)
            test_df = df.filter(pl.col("kickoff_at") >= cutoff_dt)
        else:
            # Default: last 20% by date
            n = len(df)
            split_idx = int(n * 0.8)
            train_df = df.head(split_idx)
            test_df = df.tail(n - split_idx)

        console.print(f"  Train: {len(train_df)} fixtures")
        console.print(f"  Test:  {len(test_df)} fixtures")

        if len(train_df) < 200:
            raise ValueError(f"Need >= 200 train fixtures, got {len(train_df)}")
        if len(test_df) < 50:
            console.print(
                f"[yellow]  Warning: only {len(test_df)} test fixtures"
                f" (recommend >= 50)[/yellow]"
            )

        # 3. Fit model
        console.print(f"[cyan]3. Fitting {model_type} model...[/cyan]")
        from dhx.modeling.registry import create_model

        model = create_model(model_type)

        # Get training window from the data
        fsv_rows = select_rows(
            "feature_set_versions", "training_window", {"id": feature_set_version}
        )
        model.training_window = fsv_rows[0]["training_window"] if fsv_rows else ""

        model.fit(train_df)
        console.print(f"  [green]Fitted on {model.training_samples} fixtures[/green]")
        _print_convergence_info(model_type, model.convergence_info)

        # 4. Evaluate on test set
        console.print("[cyan]4. Evaluating on test set...[/cyan]")
        from dhx.modeling.evaluate import evaluate_all

        metrics = evaluate_all(model, test_df)

        # 5. Save model + register versions
        console.print("[cyan]5. Saving model artifact...[/cyan]")
        from dhx.modeling.store import save_model

        version_ids = save_model(model, metrics, feature_set_version, model_type)
        console.print(f"  [green]Model version IDs: {version_ids}[/green]")

        # 6. Mark pipeline success
        details = {
            "feature_set_version_id": feature_set_version,
            "model_type": model_type,
            "train_size": len(train_df),
            "test_size": len(test_df),
            "model_version_ids": version_ids,
            "metrics": metrics,
        }
        _end_pipeline_run(run_id, "success", len(train_df), details)

        # Summary tables
        console.print()
        _print_metrics_table(metrics)
        _print_version_table(version_ids)

        console.print()
        console.print("[green]Training complete.[/green]")
        console.print(
            f"[dim]Activate with: python -m dhx.modeling activate "
            f"--market 1x2 --model-version-id {version_ids['1x2']}[/dim]"
        )

    except Exception as e:
        _end_pipeline_run(run_id, "failed", 0, {"error": str(e)})
        console.print(f"\n[red]Training failed: {e}[/red]")
        console.print(f"[dim]Pipeline run id={run_id} marked as failed[/dim]")
        raise typer.Exit(1)


@app.command()
def predict(
    fixture_ids: str = typer.Option(
        ..., "--fixture-ids", help="Comma-separated fixture IDs"
    ),
    log_level: str = typer.Option("INFO", "--log-level", help="Logging level"),
) -> None:
    """Generate predictions for specific fixtures using the active model."""
    _setup_logging(log_level)

    fids = [int(x.strip()) for x in fixture_ids.split(",")]
    console.print(f"[bold]=== Predicting {len(fids)} fixtures ===[/bold]")
    console.print()

    from dhx.modeling.markets import (
        ah_probability,
        find_fair_ah_line,
        margin_distribution,
        totals_probability,
    )
    from dhx.modeling.store import load_active_model

    # Load active models — all markets share the same artifact, so just load 1x2
    try:
        model, _ = load_active_model("1x2")
    except ValueError:
        console.print("[red]No active model. Run 'activate' first.[/red]")
        raise typer.Exit(1)

    # Get model_version_ids for all markets
    mv_ids: dict[str, int] = {}
    for market in ("1x2", "totals", "btts", "ah"):
        rows = select_rows(
            "model_versions", "id", filters={"market": market, "is_active": True}
        )
        if rows:
            mv_ids[market] = rows[0]["id"]

    if len(mv_ids) < 4:
        console.print("[yellow]Warning: not all markets have active models[/yellow]")

    # Load feature data for the fixtures
    df = _load_feature_data_for_fixtures(fids)
    if df is None or len(df) == 0:
        console.print("[red]No feature data found for the specified fixtures[/red]")
        raise typer.Exit(1)

    console.print(f"[dim]Loaded features for {len(df)} fixtures[/dim]")

    # Fetch market lines from odds data
    market_lines = _fetch_market_lines(fids)

    # Generate predictions
    lambdas = model.predict_lambdas(df)
    now = datetime.now(UTC).isoformat()
    prediction_rows: list[dict] = []

    results_table = Table(title="Predictions")
    results_table.add_column("Fixture")
    results_table.add_column("Lambda H", justify="right")
    results_table.add_column("Lambda A", justify="right")
    results_table.add_column("P(Home)", justify="right")
    results_table.add_column("P(Draw)", justify="right")
    results_table.add_column("P(Away)", justify="right")
    results_table.add_column("P(O2.5)", justify="right")
    results_table.add_column("P(BTTS)", justify="right")
    results_table.add_column("AH Line", justify="right")

    for row in lambdas.iter_rows(named=True):
        fid = int(row["fixture_id"])
        lh = float(row["lambda_home"])
        la = float(row["lambda_away"])
        mp = model.market_probabilities(lh, la)

        # Compute extended markets
        ah_lines = market_lines.get(fid, {}).get("ah", [])
        totals_lines_for_fixture = market_lines.get(fid, {}).get("totals", [])
        fair_ah_line = find_fair_ah_line(lh, la)
        margin_dist = margin_distribution(lh, la)

        base_context = {"lambda_home": round(lh, 4), "lambda_away": round(la, 4)}

        # Build base 7 prediction rows (1x2 + totals 2.5 + btts)
        preds: list[tuple[str, str, float | None, float, dict]] = [
            ("1x2", "home", None, mp["1x2"]["home"], base_context),
            ("1x2", "draw", None, mp["1x2"]["draw"], base_context),
            ("1x2", "away", None, mp["1x2"]["away"], base_context),
            ("totals", "over", 2.5, mp["totals"]["over"], base_context),
            ("totals", "under", 2.5, mp["totals"]["under"], base_context),
            ("btts", "yes", None, mp["btts"]["yes"], base_context),
            ("btts", "no", None, mp["btts"]["no"], base_context),
        ]

        # AH rows — for each line from odds + model fair line
        ah_context = {
            **base_context,
            "fair_ah_line": fair_ah_line,
            "margin_distribution": margin_dist,
        }
        all_ah_lines = sorted(set(ah_lines) | {fair_ah_line})
        for line in all_ah_lines:
            for side in ("home", "away"):
                r = ah_probability(lh, la, line, side)
                ctx = {**ah_context, "win": r["win"], "push": r["push"], "loss": r["loss"]}
                preds.append(("ah", side, line, r["effective_prob"], ctx))

        # Multi-line totals rows (skip 2.5 which is already in base)
        for line in sorted(set(totals_lines_for_fixture) - {2.5}):
            tp = totals_probability(lh, la, line)
            preds.append(("totals", "over", line, tp["over"], base_context))
            preds.append(("totals", "under", line, tp["under"], base_context))

        for market, selection, line, prob, context in preds:
            if market not in mv_ids:
                continue
            prob_f = float(prob)
            prob_f = max(0.001, min(0.999, prob_f))
            fair = round(1.0 / prob_f, 4) if prob_f > 0 else None
            prediction_rows.append(
                {
                    "fixture_id": fid,
                    "model_version_id": mv_ids[market],
                    "market": market,
                    "selection": selection,
                    "line": line,
                    "probability": round(prob_f, 6),
                    "fair_odds": fair,
                    "predicted_at": now,
                    "context_json": context,
                }
            )

        results_table.add_row(
            str(fid),
            f"{lh:.3f}",
            f"{la:.3f}",
            f"{mp['1x2']['home']:.3f}",
            f"{mp['1x2']['draw']:.3f}",
            f"{mp['1x2']['away']:.3f}",
            f"{mp['totals']['over']:.3f}",
            f"{mp['btts']['yes']:.3f}",
            f"{fair_ah_line:+.2f}",
        )

    # Insert into predictions table
    if prediction_rows:
        inserted = insert_rows("predictions", prediction_rows)
        console.print(f"[green]Inserted {len(inserted)} prediction rows[/green]")

    console.print()
    console.print(results_table)


@app.command("predict-upcoming")
def predict_upcoming(
    fixture_ids: str | None = typer.Option(
        None, "--fixture-ids", help="Comma-separated fixture IDs (default: all scheduled)"
    ),
    log_level: str = typer.Option("INFO", "--log-level", help="Logging level"),
) -> None:
    """Generate predictions for scheduled (upcoming) fixtures."""
    _setup_logging(log_level)

    console.print("[bold]=== Predict Upcoming Fixtures ===[/bold]")
    console.print()

    # 1. Discover scheduled fixtures
    if fixture_ids:
        fids = [int(x.strip()) for x in fixture_ids.split(",")]
        console.print(f"[dim]Using specified fixture IDs: {fids}[/dim]")
    else:
        console.print("[cyan]1. Discovering scheduled fixtures...[/cyan]")
        client = get_client()
        rows = (
            client.schema("dhx").table("fixtures")
            .select("id")
            .eq("status", "scheduled")
            .execute()
            .data
        )
        fids = [r["id"] for r in rows]
        if not fids:
            console.print("[yellow]No scheduled fixtures found.[/yellow]")
            return
        console.print(f"  Found {len(fids)} scheduled fixtures")

    # 2. Compute features for upcoming fixtures
    console.print("[cyan]2. Computing features for upcoming fixtures...[/cyan]")
    from dhx.features.upcoming import compute_features_for_upcoming

    df = compute_features_for_upcoming(fids)
    console.print(f"  [green]Feature matrix: {len(df)} rows x {len(df.columns)} cols[/green]")

    # 3. Generate predictions using the shared helper
    console.print("[cyan]3. Generating predictions...[/cyan]")
    n_inserted, results_table = _generate_and_insert_predictions(df, fids)

    console.print(f"[green]Inserted {n_inserted} prediction rows[/green]")
    console.print()
    console.print(results_table)


def _generate_and_insert_predictions(
    df: pl.DataFrame, fixture_ids: list[int]
) -> tuple[int, Table]:
    """Shared helper: load active model, predict, insert, return (count, table).

    Used by both `predict` and `predict-upcoming` commands.
    """
    from dhx.modeling.markets import (
        ah_probability,
        find_fair_ah_line,
        margin_distribution,
        totals_probability,
    )
    from dhx.modeling.store import load_active_model

    # Load active model
    try:
        model, _ = load_active_model("1x2")
    except ValueError:
        console.print("[red]No active model. Run 'activate' first.[/red]")
        raise typer.Exit(1)

    # Get model_version_ids for all markets
    mv_ids: dict[str, int] = {}
    for market in ("1x2", "totals", "btts", "ah"):
        rows = select_rows(
            "model_versions", "id", filters={"market": market, "is_active": True}
        )
        if rows:
            mv_ids[market] = rows[0]["id"]

    if len(mv_ids) < 4:
        console.print("[yellow]Warning: not all markets have active models[/yellow]")

    # Fetch market lines from odds data
    market_lines = _fetch_market_lines(fixture_ids)

    # Generate predictions
    lambdas = model.predict_lambdas(df)
    now = datetime.now(UTC).isoformat()
    prediction_rows: list[dict] = []

    results_table = Table(title="Predictions")
    results_table.add_column("Fixture")
    results_table.add_column("Lambda H", justify="right")
    results_table.add_column("Lambda A", justify="right")
    results_table.add_column("P(Home)", justify="right")
    results_table.add_column("P(Draw)", justify="right")
    results_table.add_column("P(Away)", justify="right")
    results_table.add_column("P(O2.5)", justify="right")
    results_table.add_column("P(BTTS)", justify="right")
    results_table.add_column("AH Line", justify="right")

    for row in lambdas.iter_rows(named=True):
        fid = int(row["fixture_id"])
        lh = float(row["lambda_home"])
        la = float(row["lambda_away"])
        mp = model.market_probabilities(lh, la)

        # Compute extended markets
        ah_lines = market_lines.get(fid, {}).get("ah", [])
        totals_lines_for_fixture = market_lines.get(fid, {}).get("totals", [])
        fair_ah_line = find_fair_ah_line(lh, la)
        margin_dist = margin_distribution(lh, la)

        base_context = {"lambda_home": round(lh, 4), "lambda_away": round(la, 4)}

        # Build base 7 prediction rows (1x2 + totals 2.5 + btts)
        preds: list[tuple[str, str, float | None, float, dict]] = [
            ("1x2", "home", None, mp["1x2"]["home"], base_context),
            ("1x2", "draw", None, mp["1x2"]["draw"], base_context),
            ("1x2", "away", None, mp["1x2"]["away"], base_context),
            ("totals", "over", 2.5, mp["totals"]["over"], base_context),
            ("totals", "under", 2.5, mp["totals"]["under"], base_context),
            ("btts", "yes", None, mp["btts"]["yes"], base_context),
            ("btts", "no", None, mp["btts"]["no"], base_context),
        ]

        # AH rows
        ah_context = {
            **base_context,
            "fair_ah_line": fair_ah_line,
            "margin_distribution": margin_dist,
        }
        all_ah_lines = sorted(set(ah_lines) | {fair_ah_line})
        for line in all_ah_lines:
            for side in ("home", "away"):
                r = ah_probability(lh, la, line, side)
                ctx = {**ah_context, "win": r["win"], "push": r["push"], "loss": r["loss"]}
                preds.append(("ah", side, line, r["effective_prob"], ctx))

        # Multi-line totals rows (skip 2.5 which is already in base)
        for line in sorted(set(totals_lines_for_fixture) - {2.5}):
            tp = totals_probability(lh, la, line)
            preds.append(("totals", "over", line, tp["over"], base_context))
            preds.append(("totals", "under", line, tp["under"], base_context))

        for market, selection, line, prob, context in preds:
            if market not in mv_ids:
                continue
            prob_f = float(prob)
            prob_f = max(0.001, min(0.999, prob_f))
            fair = round(1.0 / prob_f, 4) if prob_f > 0 else None
            prediction_rows.append(
                {
                    "fixture_id": fid,
                    "model_version_id": mv_ids[market],
                    "market": market,
                    "selection": selection,
                    "line": line,
                    "probability": round(prob_f, 6),
                    "fair_odds": fair,
                    "predicted_at": now,
                    "context_json": context,
                }
            )

        results_table.add_row(
            str(fid),
            f"{lh:.3f}",
            f"{la:.3f}",
            f"{mp['1x2']['home']:.3f}",
            f"{mp['1x2']['draw']:.3f}",
            f"{mp['1x2']['away']:.3f}",
            f"{mp['totals']['over']:.3f}",
            f"{mp['btts']['yes']:.3f}",
            f"{fair_ah_line:+.2f}",
        )

    # Insert into predictions table
    n_inserted = 0
    if prediction_rows:
        inserted = insert_rows("predictions", prediction_rows)
        n_inserted = len(inserted)

    return n_inserted, results_table


@app.command()
def activate(
    market: str = typer.Option(..., "--market", help="Market to activate (1x2, totals, btts, ah)"),
    model_version_id: int = typer.Option(
        ..., "--model-version-id", help="model_versions.id to activate"
    ),
) -> None:
    """Activate a model version for a specific market."""
    from dhx.modeling.store import activate_model as _activate

    _activate(market, model_version_id)
    console.print(
        f"[green]Activated model_versions id={model_version_id}"
        f" for market={market}[/green]"
    )


@app.command()
def backtest(
    feature_set_version: int = typer.Option(
        ..., "--feature-set-version", help="feature_set_versions.id to use"
    ),
    test_cutoff: str | None = typer.Option(
        None, "--test-cutoff", help="Train/test split date (YYYY-MM-DD). Default: last 20%%"
    ),
    model_type: str = typer.Option(
        "xgboost", "--model-type", help="Model type: poisson, xgboost, ensemble, or contrarian"
    ),
    kelly: float = typer.Option(
        0.25, "--kelly", help="Kelly multiplier (0.25 = quarter Kelly)"
    ),
    min_edge: float = typer.Option(
        0.03, "--min-edge", help="Minimum edge threshold to place a bet"
    ),
    max_stake_pct: float = typer.Option(
        0.05, "--max-stake-pct", help="Maximum stake as fraction of bankroll"
    ),
    bankroll: float = typer.Option(
        1000.0, "--bankroll", help="Initial bankroll"
    ),
    strategy: str = typer.Option(
        "edge", "--strategy", help="Betting strategy: edge (positive EV) or disagreement (AH line gap)"
    ),
    flat_stake: float = typer.Option(
        0.02, "--flat-stake", help="Flat stake as fraction of bankroll (disagreement strategy)"
    ),
    log_level: str = typer.Option("INFO", "--log-level", help="Logging level"),
) -> None:
    """Backtest value betting strategy using historical odds and model predictions."""
    _setup_logging(log_level)

    if model_type not in ("poisson", "xgboost", "ensemble", "contrarian"):
        console.print(f"[red]Unknown model type: {model_type}[/red]")
        raise typer.Exit(1)

    if strategy not in ("edge", "disagreement"):
        console.print(f"[red]Unknown strategy: {strategy}. Use 'edge' or 'disagreement'.[/red]")
        raise typer.Exit(1)

    label = f"{model_type}, strategy={strategy}"
    console.print(f"[bold]=== Value Betting Backtest ({label}) ===[/bold]")
    console.print()

    # 1. Load feature matrix
    console.print("[cyan]1. Loading feature matrix...[/cyan]")
    df = _load_parquet_from_storage(feature_set_version)

    if df["kickoff_at"].dtype == pl.String:
        df = df.with_columns(pl.col("kickoff_at").str.to_datetime(time_zone="UTC"))
    df = df.sort("kickoff_at")

    # 2. Train/test split
    console.print("[cyan]2. Splitting train/test...[/cyan]")
    if test_cutoff:
        cutoff_dt = datetime.strptime(test_cutoff, "%Y-%m-%d").replace(tzinfo=UTC)
        train_df = df.filter(pl.col("kickoff_at") < cutoff_dt)
        test_df = df.filter(pl.col("kickoff_at") >= cutoff_dt)
    else:
        n = len(df)
        split_idx = int(n * 0.8)
        train_df = df.head(split_idx)
        test_df = df.tail(n - split_idx)

    console.print(f"  Train: {len(train_df)} fixtures")
    console.print(f"  Test:  {len(test_df)} fixtures")

    if len(train_df) < 200:
        console.print(f"[red]Need >= 200 train fixtures, got {len(train_df)}[/red]")
        raise typer.Exit(1)

    # 3. Fit model
    console.print(f"[cyan]3. Fitting {model_type} model...[/cyan]")
    from dhx.modeling.registry import create_model

    model = create_model(model_type)
    model.training_window = ""
    model.fit(train_df)
    console.print(f"  [green]Fitted on {model.training_samples} fixtures[/green]")

    if strategy == "edge":
        # 4. Run edge backtest (existing behavior)
        console.print("[cyan]4. Running backtest...[/cyan]")
        from dhx.modeling.backtest import run_backtest

        result, sweep, edges_df = run_backtest(
            model,
            test_df,
            model_type=model_type,
            kelly_multiplier=kelly,
            min_edge=min_edge,
            max_stake_pct=max_stake_pct,
            initial_bankroll=bankroll,
        )

        # 5. Print results
        console.print()
        _print_backtest_results(result, sweep, edges_df)

    else:
        # 4. Run disagreement backtest (AH only)
        console.print("[cyan]4. Running AH disagreement backtest...[/cyan]")
        from dhx.modeling.backtest import run_disagreement_backtest

        result, edges_df = run_disagreement_backtest(
            model,
            test_df,
            model_type=model_type,
            flat_stake_pct=flat_stake,
            max_stake_pct=max_stake_pct,
            initial_bankroll=bankroll,
        )

        # 5. Print results
        console.print()
        _print_disagreement_results(result, edges_df)


@app.command("list-models")
def list_models() -> None:
    """List all model versions."""
    client = get_client()
    rows = (
        client.schema("dhx").table("model_versions")
        .select("id,model_name,market,is_active,training_window,metrics_json,created_at")
        .order("created_at", desc=True)
        .limit(50)
        .execute()
        .data
    )

    if not rows:
        console.print("[dim]No model versions found[/dim]")
        return

    table = Table(title="Model Versions")
    table.add_column("ID", justify="right")
    table.add_column("Name")
    table.add_column("Market")
    table.add_column("Active")
    table.add_column("Window")
    table.add_column("Brier", justify="right")
    table.add_column("Log-loss", justify="right")
    table.add_column("Created")

    for r in rows:
        metrics = r.get("metrics_json") or {}
        brier = str(metrics.get("brier", "-"))
        logloss = str(metrics.get("log_loss", "-"))
        active = "[green]YES[/green]" if r["is_active"] else "no"
        created = r["created_at"][:19] if r["created_at"] else "-"
        table.add_row(
            str(r["id"]),
            r["model_name"],
            r["market"],
            active,
            r.get("training_window", "-"),
            brier,
            logloss,
            created,
        )

    console.print(table)


@app.command("trend-sim")
def trend_sim(
    test_cutoff: str | None = typer.Option(
        None, "--test-cutoff", help="Only simulate on fixtures after this date (YYYY-MM-DD)"
    ),
    stake: float = typer.Option(
        100.0, "--stake", help="Flat stake per bet (e.g. 100)"
    ),
    min_trend_matches: int = typer.Option(
        3, "--min-trend-matches", help="Minimum decided matches per team before trend signal is valid"
    ),
    only_leagues: str | None = typer.Option(
        None, "--only-leagues", help="Comma-separated league keys to include (e.g. ELC,BL2,TSL)"
    ),
    exclude_leagues: str | None = typer.Option(
        None, "--exclude-leagues", help="Comma-separated league keys to exclude (e.g. EPL,LL)"
    ),
    log_level: str = typer.Option("INFO", "--log-level", help="Logging level"),
) -> None:
    """Run team-trend-only strategy simulation (trend decides model vs market → flat AH bet)."""
    _setup_logging(log_level)

    console.print("[bold]=== Team Trend Strategy Simulation ===[/bold]")
    console.print()

    from dhx.modeling.trend_sim import run_trend_sim

    only = [x.strip() for x in only_leagues.split(",")] if only_leagues else None
    exclude = [x.strip() for x in exclude_leagues.split(",")] if exclude_leagues else None

    result = run_trend_sim(
        test_cutoff=test_cutoff,
        stake=stake,
        min_trend_matches=min_trend_matches,
        only_leagues=only,
        exclude_leagues=exclude,
    )

    _print_trend_results(result)


@app.command("consensus-sim")
def consensus_sim(
    test_cutoff: str | None = typer.Option(
        None, "--test-cutoff", help="Only analyze fixtures after this date (YYYY-MM-DD)"
    ),
    only_leagues: str | None = typer.Option(
        None, "--only-leagues", help="Comma-separated league keys to include (e.g. EPL,LL)"
    ),
    exclude_leagues: str | None = typer.Option(
        None, "--exclude-leagues", help="Comma-separated league keys to exclude (e.g. EPL)"
    ),
    log_level: str = typer.Option("INFO", "--log-level", help="Logging level"),
) -> None:
    """Run consensus simulation: model-market AH agreement analysis."""
    _setup_logging(log_level)

    console.print("[bold]=== Consensus Simulation: Model-Market Agreement ===[/bold]")
    console.print()

    from dhx.modeling.consensus_sim import run_consensus_sim

    only = [x.strip() for x in only_leagues.split(",")] if only_leagues else None
    exclude = [x.strip() for x in exclude_leagues.split(",")] if exclude_leagues else None

    result = run_consensus_sim(
        test_cutoff=test_cutoff,
        only_leagues=only,
        exclude_leagues=exclude,
    )

    _print_consensus_results(result)


@app.command("strategy-sim")
def strategy_sim(
    test_cutoff: str | None = typer.Option(
        None, "--test-cutoff", help="Only simulate on fixtures after this date (YYYY-MM-DD)"
    ),
    stake: float = typer.Option(
        100.0, "--stake", help="Flat stake per bet (e.g. 100)"
    ),
    min_signals: int = typer.Option(
        1, "--min-signals", help="Minimum active signals to bet (1-4; S1 league, S2 chain, S3 trend, S4 totals)"
    ),
    only_leagues: str | None = typer.Option(
        None, "--only-leagues", help="Comma-separated league keys to include (e.g. ELC,BL2,TSL)"
    ),
    exclude_leagues: str | None = typer.Option(
        None, "--exclude-leagues", help="Comma-separated league keys to exclude (e.g. EPL,LL)"
    ),
    min_edge: float = typer.Option(
        0.0, "--min-edge", help="Minimum EV edge (model_prob × odds - 1) to place bet"
    ),
    log_level: str = typer.Option("INFO", "--log-level", help="Logging level"),
) -> None:
    """Run checklist strategy simulation (4 signals -> majority vote -> flat AH bet).

    Signals: S1 league-conditional, S2 price chain (>=3 proxies),
    S3 team trends (last 15), S4 totals agreement.
    """
    _setup_logging(log_level)

    console.print("[bold]=== Checklist Strategy Simulation ===[/bold]")
    console.print()

    from dhx.modeling.strategy_sim import run_strategy_sim

    only = [x.strip() for x in only_leagues.split(",")] if only_leagues else None
    exclude = [x.strip() for x in exclude_leagues.split(",")] if exclude_leagues else None

    result = run_strategy_sim(
        test_cutoff=test_cutoff,
        stake=stake,
        min_signals=min_signals,
        min_edge=min_edge,
        only_leagues=only,
        exclude_leagues=exclude,
    )

    _print_strategy_results(result)


# ======================================================================
# Internal helpers
# ======================================================================


def _fetch_market_lines(fixture_ids: list[int]) -> dict[int, dict[str, list[float]]]:
    """Query vw_odds_latest_pre_kickoff for distinct AH and totals lines per fixture.

    Returns {fixture_id: {"ah": [lines], "totals": [lines]}}.
    """
    client = get_client()
    result: dict[int, dict[str, list[float]]] = {}

    # Fetch in batches to respect Supabase query limits
    for market in ("ah", "totals"):
        data = (
            client.schema("dhx").table("vw_odds_latest_pre_kickoff")
            .select("fixture_id,line")
            .in_("fixture_id", fixture_ids)
            .eq("market", market)
            .execute()
            .data
        )
        if not data:
            continue
        for row in data:
            fid = row["fixture_id"]
            line = row["line"]
            if line is None:
                continue
            if fid not in result:
                result[fid] = {"ah": [], "totals": []}
            if market not in result[fid]:
                result[fid][market] = []
            line_f = float(line)
            if line_f not in result[fid][market]:
                result[fid][market].append(line_f)

    return result


def _load_feature_data_for_fixtures(fixture_ids: list[int]) -> pl.DataFrame | None:
    """Load feature data for specific fixtures from the latest feature set."""
    # Find the latest feature_set_version
    client = get_client()
    fsv_rows = (
        client.schema("dhx").table("feature_set_versions")
        .select("id,parquet_path")
        .order("created_at", desc=True)
        .limit(1)
        .execute()
        .data
    )
    if not fsv_rows:
        return None

    parquet_path = fsv_rows[0]["parquet_path"]
    parts = parquet_path.split("/", 1)
    bucket = parts[0]
    path = parts[1]

    data = client.storage.from_(bucket).download(path)
    df = pl.read_parquet(io.BytesIO(data))

    # Filter to requested fixtures
    return df.filter(pl.col("fixture_id").is_in(fixture_ids))


def _print_metrics_table(metrics: dict[str, dict]) -> None:
    table = Table(title="Evaluation Metrics (Test Set)")
    table.add_column("Market")
    table.add_column("N", justify="right")
    table.add_column("Brier", justify="right")
    table.add_column("Log-loss", justify="right")
    table.add_column("Accuracy", justify="right")
    table.add_column("Cal. Error", justify="right")

    for market in ("1x2", "totals", "btts", "ah"):
        m = metrics.get(market, {})
        table.add_row(
            market,
            str(m.get("n", "-")),
            f"{m['brier']:.4f}" if "brier" in m else "-",
            f"{m['log_loss']:.4f}" if "log_loss" in m else "-",
            f"{m['accuracy']:.3f}" if "accuracy" in m else "-",
            f"{m['calibration_error']:.4f}" if "calibration_error" in m else "-",
        )

    console.print(table)


def _print_version_table(version_ids: dict[str, int]) -> None:
    table = Table(title="Registered Model Versions")
    table.add_column("Market")
    table.add_column("Version ID", justify="right")
    table.add_column("Status")

    for market, vid in version_ids.items():
        table.add_row(market, str(vid), "[yellow]inactive[/yellow] (activate to use)")

    console.print(table)


def _print_convergence_info(model_type: str, conv: dict) -> None:
    """Print convergence info adapted to model type."""
    if model_type == "poisson":
        if "home" in conv:
            console.print(
                f"  Home: {conv['home']['iterations']} iters, "
                f"deviance={conv['home']['deviance']:.4f}"
            )
        if "away" in conv:
            console.print(
                f"  Away: {conv['away']['iterations']} iters, "
                f"deviance={conv['away']['deviance']:.4f}"
            )
        if "rho" in conv:
            console.print(f"  Dixon-Coles rho={conv['rho']:.6f}")
    elif model_type == "xgboost":
        console.print(
            f"  Features: {conv.get('n_features', '?')}, "
            f"Home best iter: {conv.get('home_best_iteration', '?')}, "
            f"Away best iter: {conv.get('away_best_iteration', '?')}"
        )
        if "rho" in conv:
            console.print(f"  Dixon-Coles rho={conv['rho']:.6f}")
    elif model_type == "ensemble":
        w_p = conv.get("weight_poisson", "?")
        w_x = conv.get("weight_xgboost", "?")
        console.print(f"  Blend weights: poisson={w_p}, xgboost={w_x}")
        if "rho" in conv:
            console.print(f"  Dixon-Coles rho={conv['rho']:.6f}")
    elif model_type == "contrarian":
        base = conv.get("base_xgboost", {})
        console.print(
            f"  Base XGBoost: {base.get('n_features', '?')} features, "
            f"Home iter: {base.get('home_best_iteration', '?')}, "
            f"Away iter: {base.get('away_best_iteration', '?')}"
        )
        console.print(
            f"  Reversion: {conv.get('n_reversion_features', '?')} features, "
            f"ridge_alpha={conv.get('ridge_alpha', '?')}"
        )
        console.print(
            f"  Correction std: home={conv.get('correction_home_std', '?')}, "
            f"away={conv.get('correction_away_std', '?')}"
        )
        if "rho" in conv:
            console.print(f"  Dixon-Coles rho={conv['rho']:.6f}")
    else:
        console.print(f"  Convergence: {conv}")


def _print_strategy_results(result) -> None:
    """Print Rich tables for checklist strategy simulation output (4-signal)."""
    console.print(
        f"[bold]{'=' * 60}[/bold]\n"
        f"[bold]  Checklist Strategy Simulation (4 Signals)[/bold]\n"
        f"[bold]{'=' * 60}[/bold]"
    )
    console.print()

    # Summary
    pnl_color = "green" if result.total_pnl >= 0 else "red"
    roi_color = "green" if result.roi >= 0 else "red"

    console.print(f"  Fixtures with odds:         {result.fixtures_with_odds}")
    console.print(f"  With line disagreement:     {result.fixtures_with_disagreement}")
    console.print(f"  Total bets:                 {result.total_bets}")
    if result.skipped > 0:
        console.print(f"  Skipped (low signals/tie):  {result.skipped}")
    console.print(f"  Stake per bet:              {result.stake_per_bet:.0f}")
    console.print(f"  Wins / Losses / Pushes:     {result.wins} / {result.losses} / {result.pushes}")
    console.print(f"  Hit rate:                   [{roi_color}]{result.hit_rate:.1%}[/{roi_color}]")
    console.print(f"  Total P&L:                  [{pnl_color}]{result.total_pnl:+.0f}[/{pnl_color}]")
    console.print(f"  ROI:                        [{roi_color}]{result.roi:.1%}[/{roi_color}]")
    console.print()

    # By decision
    table = Table(title="By Decision (Follow Model vs Follow Market)")
    table.add_column("Decision")
    table.add_column("Bets", justify="right")
    table.add_column("W/L", justify="right")
    table.add_column("Hit Rate", justify="right")
    table.add_column("P&L", justify="right")

    for dec in ("model", "market"):
        stats = result.by_decision.get(dec, {})
        bets = stats.get("bets", 0)
        wins = stats.get("wins", 0)
        losses = stats.get("losses", 0)
        hr = stats.get("hit_rate", 0.0)
        pnl = stats.get("pnl", 0.0)
        pc = "green" if pnl >= 0 else "red"
        table.add_row(
            dec.title(),
            str(bets),
            f"{wins}/{losses}",
            f"{hr:.1%}",
            f"[{pc}]{pnl:+.0f}[/{pc}]",
        )

    console.print(table)
    console.print()

    # By signal count (4-signal buckets)
    table2 = Table(title="By Signal Count (How Many Signals Active)")
    table2.add_column("Signals")
    table2.add_column("Bets", justify="right")
    table2.add_column("W/L", justify="right")
    table2.add_column("Hit Rate", justify="right")
    table2.add_column("P&L", justify="right")

    labels = {
        "4_signals": "All 4",
        "3_signals": "3 of 4",
        "2_signals": "2 of 4",
        "1_signal": "1 only",
    }
    for key in ("4_signals", "3_signals", "2_signals", "1_signal"):
        stats = result.by_signal_count.get(key, {})
        bets = stats.get("bets", 0)
        wins = stats.get("wins", 0)
        losses = stats.get("losses", 0)
        hr = stats.get("hit_rate", 0.0)
        pnl = stats.get("pnl", 0.0)
        pc = "green" if pnl >= 0 else "red"
        table2.add_row(
            labels[key],
            str(bets),
            f"{wins}/{losses}",
            f"{hr:.1%}",
            f"[{pc}]{pnl:+.0f}[/{pc}]",
        )

    console.print(table2)
    console.print()

    # Per-signal accuracy
    if hasattr(result, "by_signal") and result.by_signal:
        table_sig = Table(title="Per-Signal Accuracy")
        table_sig.add_column("Signal")
        table_sig.add_column("Fired", justify="right")
        table_sig.add_column("Correct", justify="right")
        table_sig.add_column("Accuracy", justify="right")

        sig_labels = {
            "s1": "S1 League",
            "s2": "S2 Chain",
            "s3": "S3 Trend",
            "s4": "S4 Totals",
        }
        for sig_key in ("s1", "s2", "s3", "s4"):
            stats = result.by_signal.get(sig_key, {})
            fired = stats.get("fired", 0)
            correct = stats.get("correct", 0)
            pct = stats.get("pct", 0.0)
            if fired == 0:
                table_sig.add_row(sig_labels[sig_key], "0", "-", "-")
            else:
                pc = "green" if pct >= 50 else "red"
                table_sig.add_row(
                    sig_labels[sig_key],
                    str(fired),
                    str(correct),
                    f"[{pc}]{pct:.1f}%[/{pc}]",
                )

        console.print(table_sig)
        console.print()

    # By signal combination (top combos by volume)
    if hasattr(result, "by_combination") and result.by_combination:
        # Sort by bet count descending
        sorted_combos = sorted(
            result.by_combination.items(),
            key=lambda x: x[1]["bets"],
            reverse=True,
        )
        # Show top 15
        top_combos = sorted_combos[:15]

        table_combo = Table(title="By Signal Combination (top patterns)")
        table_combo.add_column("Pattern (S1,S2,S3,S4)")
        table_combo.add_column("Bets", justify="right")
        table_combo.add_column("W/L", justify="right")
        table_combo.add_column("Hit Rate", justify="right")
        table_combo.add_column("P&L", justify="right")

        for pattern, stats in top_combos:
            bets = stats["bets"]
            wins = stats["wins"]
            losses = stats["losses"]
            hr = stats["hit_rate"]
            pnl = stats["pnl"]
            pc = "green" if pnl >= 0 else "red"
            # Make pattern more readable
            parts = pattern.split(",")
            display = " ".join(
                f"{'mdl' if p == 'model' else 'mkt' if p == 'market' else ' - '}"
                for p in parts
            )
            table_combo.add_row(
                display,
                str(bets),
                f"{wins}/{losses}",
                f"{hr:.1%}",
                f"[{pc}]{pnl:+.0f}[/{pc}]",
            )

        console.print(table_combo)
        console.print()

    # By league
    table3 = Table(title="By League")
    table3.add_column("League")
    table3.add_column("Bets", justify="right")
    table3.add_column("W/L", justify="right")
    table3.add_column("Hit Rate", justify="right")
    table3.add_column("P&L", justify="right")

    for lk in sorted(result.by_league.keys()):
        stats = result.by_league[lk]
        bets = stats.get("bets", 0)
        wins = stats.get("wins", 0)
        losses = stats.get("losses", 0)
        hr = stats.get("hit_rate", 0.0)
        pnl = stats.get("pnl", 0.0)
        pc = "green" if pnl >= 0 else "red"
        table3.add_row(
            lk,
            str(bets),
            f"{wins}/{losses}",
            f"{hr:.1%}",
            f"[{pc}]{pnl:+.0f}[/{pc}]",
        )

    console.print(table3)
    console.print()

    # Recent bets (last 20)
    recent = result.bets[-20:] if len(result.bets) > 20 else result.bets
    if recent:
        table4 = Table(title=f"Recent Bets (last {len(recent)} of {result.total_bets})")
        table4.add_column("Date")
        table4.add_column("Lge")
        table4.add_column("Match")
        table4.add_column("Model", justify="right")
        table4.add_column("Market", justify="right")
        table4.add_column("S1", justify="center")
        table4.add_column("S2", justify="center")
        table4.add_column("S3", justify="center")
        table4.add_column("S4", justify="center")
        table4.add_column("Dec")
        table4.add_column("Bet")
        table4.add_column("Score", justify="center")
        table4.add_column("P&L", justify="right")

        for b in recent:
            date_str = b.kickoff_at[:10] if len(b.kickoff_at) >= 10 else b.kickoff_at
            s1_str = (b.signal_s1 or "-")[:3]
            s2_str = b.signal_chain or "-"
            if b.signal_chain and b.chain_implied_ah is not None:
                s2_str = f"{b.signal_chain[:3]}({b.proxy_count})"
            elif b.signal_chain:
                s2_str = b.signal_chain[:3]
            s3_str = b.signal_trend or "-"
            if b.signal_trend and b.home_team_model_pct is not None:
                s3_str = f"{b.signal_trend[:3]}({b.home_team_model_pct:.0f}/{b.away_team_model_pct:.0f})"
            elif b.signal_trend:
                s3_str = b.signal_trend[:3]
            s4_str = (b.signal_totals or "-")[:3]
            pnl_c = "green" if b.pnl > 0 else ("red" if b.pnl < 0 else "dim")
            dec_c = "blue" if b.decision == "model" else "yellow"

            table4.add_row(
                date_str,
                b.league_key,
                f"{b.home_team} v {b.away_team}",
                f"{b.model_line:+.2f}",
                f"{b.market_line:+.2f}",
                s1_str,
                s2_str,
                s3_str,
                s4_str,
                f"[{dec_c}]{b.decision[:3]}[/{dec_c}]",
                b.bet_side.title(),
                f"{b.home_score}-{b.away_score}",
                f"[{pnl_c}]{b.pnl:+.0f}[/{pnl_c}]",
            )

        console.print(table4)


def _print_trend_results(result) -> None:
    """Print Rich tables for trend strategy simulation output."""
    console.print(
        f"[bold]{'=' * 60}[/bold]\n"
        f"[bold]  Team Trend + Implied Odds Strategy Simulation[/bold]\n"
        f"[bold]{'=' * 60}[/bold]"
    )
    console.print()

    # Summary
    pnl_color = "green" if result.total_pnl >= 0 else "red"
    roi_color = "green" if result.roi >= 0 else "red"

    console.print(f"  Fixtures with odds:         {result.fixtures_with_odds}")
    console.print(f"  With line disagreement:     {result.fixtures_with_disagreement}")
    console.print(f"  Skipped (implied→market):   {result.skipped_chain_market}")
    console.print(f"  Skipped (no trend data):    {result.skipped_no_trend}")
    console.print(f"  Total bets:                 {result.total_bets}")
    console.print(f"  Stake per bet:              {result.stake_per_bet:.0f}")
    console.print(f"  Wins / Losses / Pushes:     {result.wins} / {result.losses} / {result.pushes}")
    console.print(f"  Hit rate:                   [{roi_color}]{result.hit_rate:.1%}[/{roi_color}]")
    console.print(f"  Total P&L:                  [{pnl_color}]{result.total_pnl:+.0f}[/{pnl_color}]")
    console.print(f"  ROI:                        [{roi_color}]{result.roi:.1%}[/{roi_color}]")
    console.print()

    # By reason
    table = Table(title="By Reason (What Triggered the Bet)")
    table.add_column("Reason")
    table.add_column("Bets", justify="right")
    table.add_column("W/L", justify="right")
    table.add_column("Hit Rate", justify="right")
    table.add_column("P&L", justify="right")

    reason_labels = {
        "chain→model": "Implied→Model, bet Model",
        "trend→model": "No chain, trend→Model",
        "trend→market": "No chain, trend→Market",
    }
    for reason_key in ("chain→model", "trend→model", "trend→market"):
        stats = result.by_reason.get(reason_key, {})
        bets = stats.get("bets", 0)
        if bets == 0:
            continue
        wins = stats.get("wins", 0)
        losses = stats.get("losses", 0)
        hr = stats.get("hit_rate", 0.0)
        pnl = stats.get("pnl", 0.0)
        pc = "green" if pnl >= 0 else "red"
        table.add_row(
            reason_labels.get(reason_key, reason_key),
            str(bets),
            f"{wins}/{losses}",
            f"{hr:.1%}",
            f"[{pc}]{pnl:+.0f}[/{pc}]",
        )

    console.print(table)
    console.print()

    # By league
    table2 = Table(title="By League")
    table2.add_column("League")
    table2.add_column("Bets", justify="right")
    table2.add_column("W/L", justify="right")
    table2.add_column("Hit Rate", justify="right")
    table2.add_column("P&L", justify="right")

    for lk in sorted(result.by_league.keys()):
        stats = result.by_league[lk]
        bets = stats.get("bets", 0)
        wins = stats.get("wins", 0)
        losses = stats.get("losses", 0)
        hr = stats.get("hit_rate", 0.0)
        pnl = stats.get("pnl", 0.0)
        pc = "green" if pnl >= 0 else "red"
        table2.add_row(
            lk,
            str(bets),
            f"{wins}/{losses}",
            f"{hr:.1%}",
            f"[{pc}]{pnl:+.0f}[/{pc}]",
        )

    console.print(table2)
    console.print()

    # Per-league breakdown by reason
    if hasattr(result, "by_league_reason") and result.by_league_reason:
        table_lr = Table(title="Per-League Breakdown by Reason")
        table_lr.add_column("League")
        table_lr.add_column("Reason")
        table_lr.add_column("Bets", justify="right")
        table_lr.add_column("W/L", justify="right")
        table_lr.add_column("Hit Rate", justify="right")
        table_lr.add_column("P&L", justify="right")

        reason_short = {
            "chain→model": "chain→mdl",
            "trend→model": "trend→mdl",
            "trend→market": "trend→mkt",
        }
        for lk in sorted(result.by_league_reason.keys()):
            reasons = result.by_league_reason[lk]
            for rk in ("chain→model", "trend→model", "trend→market"):
                stats = reasons.get(rk)
                if not stats or stats["bets"] == 0:
                    continue
                bets = stats["bets"]
                wins = stats["wins"]
                losses = stats["losses"]
                hr = stats["hit_rate"]
                pnl = stats["pnl"]
                pc = "green" if pnl >= 0 else "red"
                table_lr.add_row(
                    lk,
                    reason_short.get(rk, rk),
                    str(bets),
                    f"{wins}/{losses}",
                    f"{hr:.1%}",
                    f"[{pc}]{pnl:+.0f}[/{pc}]",
                )

        console.print(table_lr)
        console.print()

    # Recent bets (last 30)
    recent = result.bets[-30:] if len(result.bets) > 30 else result.bets
    if recent:
        table3 = Table(title=f"Recent Bets (last {len(recent)} of {result.total_bets})")
        table3.add_column("Date")
        table3.add_column("League")
        table3.add_column("Match")
        table3.add_column("Model", justify="right")
        table3.add_column("Market", justify="right")
        table3.add_column("Implied", justify="right")
        table3.add_column("Reason")
        table3.add_column("Bet")
        table3.add_column("Score", justify="center")
        table3.add_column("P&L", justify="right")

        for b in recent:
            date_str = b.kickoff_at[:10] if len(b.kickoff_at) >= 10 else b.kickoff_at
            implied_str = f"{b.chain_implied:+.2f}" if b.chain_implied is not None else "-"
            reason_short = b.reason.replace("chain→model", "chain→mdl").replace("trend→", "trend→")
            pnl_c = "green" if b.pnl > 0 else ("red" if b.pnl < 0 else "dim")

            table3.add_row(
                date_str,
                b.league_key,
                f"{b.home_team} v {b.away_team}",
                f"{b.model_line:+.2f}",
                f"{b.market_line:+.2f}",
                implied_str,
                reason_short,
                f"{b.bet_team} ({b.bet_side})",
                f"{b.home_score}-{b.away_score}",
                f"[{pnl_c}]{b.pnl:+.0f}[/{pnl_c}]",
            )

        console.print(table3)


def _print_consensus_results(result) -> None:
    """Print Rich tables for consensus simulation output."""
    console.print(
        f"[bold]{'=' * 60}[/bold]\n"
        f"[bold]  Consensus Simulation: Model-Market Agreement[/bold]\n"
        f"[bold]{'=' * 60}[/bold]"
    )
    console.print()

    # Summary
    agree_pct = (
        result.total_agreed / result.total_with_ah * 100
        if result.total_with_ah > 0
        else 0.0
    )
    console.print(f"  Total finished fixtures:    {result.total_finished}")
    console.print(f"  With AH lines (both):       {result.total_with_ah}")
    console.print(f"  AH agreed (same direction): {result.total_agreed} ({agree_pct:.1f}%)")
    console.print()

    # AH Coverage table
    fav_color = "green" if result.fav_cover_rate >= 0.5 else "red"
    decided = result.fav_covers + result.dog_covers

    table = Table(title="AH Coverage: Favorite vs Underdog (Agreed Matches)")
    table.add_column("Segment")
    table.add_column("Total", justify="right")
    table.add_column("Fav Covers", justify="right")
    table.add_column("Dog Covers", justify="right")
    table.add_column("Push", justify="right")
    table.add_column("Fav %", justify="right")

    table.add_row(
        "[bold]Overall[/bold]",
        str(result.total_agreed),
        str(result.fav_covers),
        str(result.dog_covers),
        str(result.pushes),
        f"[{fav_color}]{result.fav_cover_rate:.1%}[/{fav_color}]",
    )

    for lk in sorted(result.ah_by_league.keys()):
        stats = result.ah_by_league[lk]
        fc = "green" if stats.fav_cover_rate >= 0.5 else "red"
        table.add_row(
            lk,
            str(stats.total),
            str(stats.fav_covers),
            str(stats.dog_covers),
            str(stats.pushes),
            f"[{fc}]{stats.fav_cover_rate:.1%}[/{fc}]",
        )

    console.print(table)
    console.print()

    # O/U Disagreement table
    console.print(
        f"  Agreed matches with O/U data: {result.total_with_ou}\n"
        f"  O/U disagreements:            {result.disagree_count}"
    )
    console.print()

    if result.disagree_count > 0:
        mw_color = "green" if result.model_win_rate >= 0.5 else "red"

        table2 = Table(title="O/U Disagreement: Model vs Market (Among AH-Agreed)")
        table2.add_column("Segment")
        table2.add_column("Disagree", justify="right")
        table2.add_column("Model Wins", justify="right")
        table2.add_column("Market Wins", justify="right")
        table2.add_column("Model %", justify="right")

        table2.add_row(
            "[bold]Overall[/bold]",
            str(result.disagree_count),
            str(result.model_correct),
            str(result.market_correct),
            f"[{mw_color}]{result.model_win_rate:.1%}[/{mw_color}]",
        )

        for lk in sorted(result.ou_by_league.keys()):
            stats = result.ou_by_league[lk]
            if stats.total == 0:
                continue
            mc = "green" if stats.model_win_rate >= 0.5 else "red"
            table2.add_row(
                lk,
                str(stats.total),
                str(stats.model_correct),
                str(stats.market_correct),
                f"[{mc}]{stats.model_win_rate:.1%}[/{mc}]",
            )

        console.print(table2)
    else:
        console.print("[dim]No O/U disagreements found.[/dim]")

    console.print()


def _print_disagreement_results(result, edges_df) -> None:
    """Print Rich tables for disagreement backtest output."""
    # Header
    console.print(
        f"[bold]{'=' * 60}[/bold]\n"
        f"[bold]  AH Disagreement Backtest ({result.model_type})[/bold]\n"
        f"[bold]{'=' * 60}[/bold]"
    )
    console.print()

    # Summary
    pnl_color = "green" if result.total_pnl >= 0 else "red"
    roi_color = "green" if result.roi >= 0 else "red"
    console.print("[bold]Summary:[/bold]")
    console.print(
        f"  Bankroll: {result.initial_bankroll:.2f} -> "
        f"[{pnl_color}]{result.final_bankroll:.2f}[/{pnl_color}]"
    )
    console.print(f"  Total bets: {result.total_bets}")
    console.print(f"  Total staked: {result.total_staked:.2f}")
    console.print(
        f"  P&L: [{pnl_color}]{result.total_pnl:+.2f}[/{pnl_color}]"
    )
    console.print(
        f"  ROI: [{roi_color}]{result.roi:+.2%}[/{roi_color}]"
    )
    console.print(f"  Hit rate: {result.hit_rate:.2%}")
    console.print(f"  Max drawdown: {result.max_drawdown_pct:.1f}%")
    console.print(f"  Staking: flat")
    console.print()

    # Gap breakdown table
    if len(edges_df) > 0 and "gap" in edges_df.columns:
        # Build bet lookup for settlement results
        bet_lookup: dict[tuple[int, str], dict] = {}
        for b in result.bets:
            bet_lookup[(b.fixture_id, b.selection)] = {
                "pnl": b.pnl,
                "stake": b.stake,
                "result": b.result,
            }

        # Group edges by gap scenario
        scenarios = [
            ("Model more aggressive (gap < 0)", edges_df.filter(pl.col("gap") < 0), "underdog"),
            ("Agree (gap = 0)", edges_df.filter(pl.col("gap") == 0), "favorite"),
            ("Market more aggressive (gap > 0)", edges_df.filter(pl.col("gap") > 0), "underdog"),
        ]

        table = Table(title="Gap Breakdown")
        table.add_column("Scenario")
        table.add_column("Bets", justify="right")
        table.add_column("Side")
        table.add_column("Hit Rate", justify="right")
        table.add_column("P&L", justify="right")
        table.add_column("ROI", justify="right")

        for label, subset, side_label in scenarios:
            if len(subset) == 0:
                table.add_row(label, "0", side_label, "-", "-", "-")
                continue

            # Match to actual bets
            n_bets = 0
            n_hits = 0
            total_pnl = 0.0
            total_staked = 0.0
            for row in subset.iter_rows(named=True):
                fid = int(row["fixture_id"])
                sel = row["selection"]
                key = (fid, sel)
                if key in bet_lookup:
                    b = bet_lookup[key]
                    n_bets += 1
                    total_pnl += b["pnl"]
                    total_staked += b["stake"]
                    if b["result"] in ("win", "half_win"):
                        n_hits += 1

            hit_rate = n_hits / n_bets if n_bets > 0 else 0.0
            roi = total_pnl / total_staked if total_staked > 0 else 0.0
            pnl_c = "green" if total_pnl >= 0 else "red"
            roi_c = "green" if roi >= 0 else "red"

            table.add_row(
                label,
                str(n_bets),
                side_label,
                f"{hit_rate:.1%}",
                f"[{pnl_c}]{total_pnl:+.2f}[/{pnl_c}]",
                f"[{roi_c}]{roi:+.2%}[/{roi_c}]",
            )

        console.print(table)
        console.print()

        # Show some example bets
        if len(edges_df) > 0:
            table = Table(title="Sample Bets (first 10)")
            table.add_column("Fixture", justify="right")
            table.add_column("Fair Line", justify="right")
            table.add_column("Market Line", justify="right")
            table.add_column("Gap", justify="right")
            table.add_column("Side")
            table.add_column("Odds", justify="right")
            table.add_column("Result")
            table.add_column("P&L", justify="right")

            sample = edges_df.sort("kickoff_at").head(10)
            for row in sample.iter_rows(named=True):
                fid = int(row["fixture_id"])
                sel = row["selection"]
                key = (fid, sel)
                bet = bet_lookup.get(key)
                result_str = bet["result"] if bet else "-"
                pnl_str = f"{bet['pnl']:+.2f}" if bet else "-"
                pnl_c = ""
                if bet:
                    pnl_c = "green" if bet["pnl"] >= 0 else "red"
                    pnl_str = f"[{pnl_c}]{pnl_str}[/{pnl_c}]"

                table.add_row(
                    str(fid),
                    f"{row['fair_line']:+.2f}",
                    f"{row['market_line']:+.2f}",
                    f"{row['gap']:+.2f}",
                    sel,
                    f"{row['market_odds']:.3f}",
                    result_str,
                    pnl_str,
                )

            console.print(table)


def _print_backtest_results(result, sweep: list[dict], edges_df) -> None:
    """Print Rich tables for backtest output."""
    from dhx.modeling.backtest import BacktestResult

    # Header
    console.print(
        f"[bold]{'=' * 55}[/bold]\n"
        f"[bold]  Backtest Results ({result.model_type}, "
        f"min_edge={result.min_edge:.0%})[/bold]\n"
        f"[bold]{'=' * 55}[/bold]"
    )
    console.print()

    # Summary
    pnl_color = "green" if result.total_pnl >= 0 else "red"
    roi_color = "green" if result.roi >= 0 else "red"
    console.print("[bold]Summary:[/bold]")
    console.print(
        f"  Bankroll: {result.initial_bankroll:.2f} -> "
        f"[{pnl_color}]{result.final_bankroll:.2f}[/{pnl_color}]"
    )
    console.print(f"  Total bets: {result.total_bets}")
    console.print(f"  Total staked: {result.total_staked:.2f}")
    console.print(
        f"  P&L: [{pnl_color}]{result.total_pnl:+.2f}[/{pnl_color}]"
    )
    console.print(
        f"  ROI: [{roi_color}]{result.roi:+.2%}[/{roi_color}]"
    )
    console.print(f"  Hit rate: {result.hit_rate:.2%}")
    console.print(f"  Avg edge: {result.avg_edge:.2%}")
    console.print(f"  Max drawdown: {result.max_drawdown_pct:.1f}%")
    console.print(
        f"  Kelly: {result.kelly_multiplier}x | "
        f"Max stake: {5:.0f}% of bankroll"
    )
    console.print()

    # By-market table
    if result.bets_by_market:
        table = Table(title="By Market")
        table.add_column("Market")
        table.add_column("Bets", justify="right")
        table.add_column("Hit Rate", justify="right")
        table.add_column("Avg Edge", justify="right")
        table.add_column("Staked", justify="right")
        table.add_column("P&L", justify="right")
        table.add_column("ROI", justify="right")

        for mkt in ("1x2", "totals", "btts", "ah"):
            m = result.bets_by_market.get(mkt)
            if not m:
                continue
            pnl_c = "green" if m["pnl"] >= 0 else "red"
            roi_c = "green" if m["roi"] >= 0 else "red"
            table.add_row(
                mkt,
                str(m["bets"]),
                f"{m['hit_rate']:.2%}",
                f"{m['avg_edge']:.2%}",
                f"{m['staked']:.2f}",
                f"[{pnl_c}]{m['pnl']:+.2f}[/{pnl_c}]",
                f"[{roi_c}]{m['roi']:+.2%}[/{roi_c}]",
            )

        console.print(table)
        console.print()

    # Edge threshold sweep
    if sweep:
        table = Table(title="Edge Threshold Sweep")
        table.add_column("Min Edge", justify="right")
        table.add_column("Bets", justify="right")
        table.add_column("Staked", justify="right")
        table.add_column("P&L", justify="right")
        table.add_column("ROI", justify="right")
        table.add_column("Hit Rate", justify="right")
        table.add_column("Drawdown", justify="right")
        table.add_column("Final", justify="right")

        for s in sweep:
            pnl_c = "green" if s["pnl"] >= 0 else "red"
            roi_c = "green" if s["roi"] >= 0 else "red"
            table.add_row(
                f"{s['min_edge']:.0%}",
                str(s["bets"]),
                f"{s['staked']:.2f}",
                f"[{pnl_c}]{s['pnl']:+.2f}[/{pnl_c}]",
                f"[{roi_c}]{s['roi']:+.2%}[/{roi_c}]",
                f"{s['hit_rate']:.2%}",
                f"{s['max_drawdown']:.1f}%",
                f"{s['final_bankroll']:.2f}",
            )

        console.print(table)
        console.print()

    # Edge distribution
    if len(edges_df) > 0:
        import polars as pl

        table = Table(title="Edge Distribution (All Computed Edges)")
        table.add_column("Range", justify="right")
        table.add_column("Count", justify="right")
        table.add_column("% of Total", justify="right")

        edges = edges_df["edge"]
        total = len(edges)
        buckets = [
            ("< -10%", edges.filter(edges < -0.10).len()),
            ("-10% to -5%", edges.filter((edges >= -0.10) & (edges < -0.05)).len()),
            ("-5% to 0%", edges.filter((edges >= -0.05) & (edges < 0.0)).len()),
            ("0% to 3%", edges.filter((edges >= 0.0) & (edges < 0.03)).len()),
            ("3% to 5%", edges.filter((edges >= 0.03) & (edges < 0.05)).len()),
            ("5% to 10%", edges.filter((edges >= 0.05) & (edges < 0.10)).len()),
            ("10% to 20%", edges.filter((edges >= 0.10) & (edges < 0.20)).len()),
            ("> 20%", edges.filter(edges >= 0.20).len()),
        ]
        for label, count in buckets:
            count_val = int(count)
            pct = count_val / total * 100 if total > 0 else 0
            table.add_row(label, str(count_val), f"{pct:.1f}%")

        console.print(table)


if __name__ == "__main__":
    app()
