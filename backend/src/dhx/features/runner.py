"""CLI entry point for feature engineering pipeline.

Usage:
    python -m dhx.features build --start-date 2025-08-15 --end-date 2026-02-16
"""

from __future__ import annotations

import logging
from datetime import date, datetime, timezone
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

from dhx.db import insert_rows, update_rows

app = typer.Typer(help="DHextra feature engineering CLI", invoke_without_command=True)
console = Console()


@app.callback()
def _callback() -> None:
    """Feature engineering pipeline for DHextra."""


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def _start_pipeline_run() -> int:
    """Create a pipeline_runs row with status=running. Returns the run ID."""
    row = {
        "pipeline_name": "feature_engineering",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "status": "running",
        "details_json": {},
    }
    inserted = insert_rows("pipeline_runs", [row])
    return inserted[0]["id"]


def _end_pipeline_run(run_id: int, status: str, rows_written: int, details: dict) -> None:
    """Update a pipeline_runs row with final status."""
    update_rows(
        "pipeline_runs",
        {
            "ended_at": datetime.now(timezone.utc).isoformat(),
            "status": status,
            "rows_written": rows_written,
            "details_json": details,
        },
        {"id": run_id},
    )


@app.command()
def build(
    start_date: str = typer.Option(..., "--start-date", help="Start date YYYY-MM-DD (inclusive)"),
    end_date: Optional[str] = typer.Option(None, "--end-date", help="End date YYYY-MM-DD (inclusive, default: today)"),
    log_level: str = typer.Option("INFO", "--log-level", help="Logging level"),
) -> None:
    """Build the feature matrix from database data and upload to Storage."""
    _setup_logging(log_level)

    if end_date is None:
        end_date_val = date.today().isoformat()
    else:
        end_date_val = end_date

    # Validate dates
    try:
        start_dt = datetime.strptime(start_date, "%Y-%m-%d").date()
        end_dt = datetime.strptime(end_date_val, "%Y-%m-%d").date()
    except ValueError as e:
        console.print(f"[red]Invalid date format: {e}. Use YYYY-MM-DD[/red]")
        raise typer.Exit(1)

    console.print("[bold]═══ DHextra Feature Engineering ═══[/bold]")
    console.print(f"[bold]Range:[/bold] {start_date} → {end_date_val}")
    console.print()

    # Start pipeline tracking
    run_id = _start_pipeline_run()
    console.print(f"[dim]Pipeline run id={run_id}[/dim]")
    console.print()

    try:
        # Step 1: Load data
        console.print("[cyan]▶ Step 1/6: Loading data from Supabase...[/cyan]")
        from dhx.features.loader import load_feature_data

        data = load_feature_data()
        console.print(
            f"  [green]✓ Loaded {len(data['fixtures'])} fixtures, "
            f"{len(data['results'])} results, "
            f"{len(data['stats'])} stats, "
            f"{len(data['odds'])} odds[/green]"
        )

        # Step 2: Unpivot to team perspective
        console.print("[cyan]▶ Step 2/6: Unpivoting to team perspective...[/cyan]")
        from dhx.features.unpivot import unpivot_to_team_perspective

        team_df = unpivot_to_team_perspective(
            data["fixtures"], data["results"], data["stats"],
            player_ratings=data["player_ratings"],
        )
        console.print(f"  [green]✓ {len(team_df)} team-perspective rows[/green]")

        # Step 3: Compute rolling features
        console.print("[cyan]▶ Step 3/6: Computing rolling features...[/cyan]")
        from dhx.features.rolling import compute_all_rolling_features

        team_features = compute_all_rolling_features(team_df)
        n_rolling = len([c for c in team_features.columns if c not in team_df.columns])
        console.print(f"  [green]✓ {n_rolling} rolling features added[/green]")

        # Step 4: Compute market features
        console.print("[cyan]▶ Step 4/6: Computing market features...[/cyan]")
        from dhx.features.market import compute_market_features

        market_features = compute_market_features(data["odds"])
        n_mkt = len(market_features.columns) - 1  # exclude fixture_id
        console.print(f"  [green]✓ {n_mkt} market features for {len(market_features)} fixtures[/green]")

        # Step 5: Compute labels
        console.print("[cyan]▶ Step 5/6: Computing labels...[/cyan]")
        from dhx.features.labels import compute_labels

        labels = compute_labels(data["fixtures"], data["results"])
        n_labels = len(labels.columns) - 1
        console.print(f"  [green]✓ {n_labels} labels for {len(labels)} fixtures[/green]")

        # Step 6: Assemble and store
        console.print("[cyan]▶ Step 6/6: Assembling feature matrix and uploading...[/cyan]")
        from dhx.features.assemble import assemble_feature_matrix
        from dhx.features.store import save_feature_set

        matrix = assemble_feature_matrix(
            data["fixtures"], team_features, market_features, labels,
            start_dt, end_dt,
        )
        console.print(f"  [green]✓ Matrix: {len(matrix)} rows × {len(matrix.columns)} columns[/green]")

        version_id = save_feature_set(matrix, start_date, end_date_val)
        console.print(f"  [green]✓ Saved as feature_set_versions id={version_id}[/green]")

        # Mark pipeline as success
        details = {
            "rows": len(matrix),
            "columns": len(matrix.columns),
            "feature_set_version_id": version_id,
            "start_date": start_date,
            "end_date": end_date_val,
        }
        _end_pipeline_run(run_id, "success", len(matrix), details)

        # Summary table
        console.print()
        table = Table(title="Feature Matrix Summary")
        table.add_column("Metric")
        table.add_column("Value", justify="right")
        table.add_row("Rows (fixtures)", str(len(matrix)))
        table.add_row("Total columns", str(len(matrix.columns)))
        table.add_row("Feature columns", str(len([c for c in matrix.columns if not c.startswith("label_") and c not in ("fixture_id", "home_team_id", "away_team_id", "kickoff_at", "league_id", "league_key")])))
        table.add_row("Label columns", str(n_labels))
        table.add_row("Delta columns", str(len([c for c in matrix.columns if c.startswith("delta_")])))
        table.add_row("Market columns", str(len([c for c in matrix.columns if c.startswith("mkt_")])))
        table.add_row("Feature set version", str(version_id))
        table.add_row("Pipeline run", str(run_id))
        console.print(table)

        # Null analysis
        console.print()
        null_counts = matrix.null_count()
        high_nulls = [
            c for c in matrix.columns
            if null_counts[c][0] > len(matrix) * 0.5
        ]
        if high_nulls:
            console.print(f"[yellow]Columns with >50% nulls: {len(high_nulls)}[/yellow]")
        else:
            console.print("[green]✓ All columns have <50% nulls[/green]")

        console.print()
        console.print("[green]✓ Feature engineering complete.[/green]")

    except Exception as e:
        _end_pipeline_run(run_id, "failed", 0, {"error": str(e)})
        console.print(f"\n[red]✗ Pipeline failed: {e}[/red]")
        console.print(f"[dim]Pipeline run id={run_id} marked as failed[/dim]")
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
