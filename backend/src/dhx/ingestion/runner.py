"""CLI entry point for ingestion workers.

Usage:
    python -m dhx.ingestion.runner --source sofascore --date 2026-02-17
    python -m dhx.ingestion.runner --source sofascore --date 2026-02-17 --days 3
"""

import logging
from datetime import date, datetime, timedelta
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(help="DHextra data ingestion CLI")
console = Console()


def _setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


@app.command()
def ingest(
    source: str = typer.Option("sofascore", help="Data source key"),
    date_str: Optional[str] = typer.Option(None, "--date", help="Start date YYYY-MM-DD (default: today UTC)"),
    days: int = typer.Option(1, help="Number of days to ingest from start date"),
    log_level: str = typer.Option("INFO", "--log-level", help="Logging level"),
) -> None:
    """Ingest data from a source for one or more dates."""
    _setup_logging(log_level)

    if date_str is None:
        start = date.today()
    else:
        try:
            start = datetime.strptime(date_str, "%Y-%m-%d").date()
        except ValueError:
            console.print(f"[red]Invalid date format: {date_str}. Use YYYY-MM-DD[/red]")
            raise typer.Exit(1)

    dates = [start + timedelta(days=i) for i in range(days)]
    console.print(f"[bold]Source:[/bold] {source}")
    console.print(f"[bold]Dates:[/bold]  {dates[0]} → {dates[-1]} ({len(dates)} day(s))")
    console.print()

    if source == "sofascore":
        from dhx.ingestion.sofascore import run as run_sofascore

        all_stats = []
        for d in dates:
            d_str = d.isoformat()
            console.print(f"[cyan]▶ Ingesting {d_str}...[/cyan]")
            stats = run_sofascore(d_str)
            all_stats.append((d_str, stats))

            # Print per-date summary
            color = "green" if stats["errors"] == 0 else "yellow"
            console.print(
                f"  [{color}]✓ events={stats['events_fetched']} "
                f"tracked={stats['events_tracked']} "
                f"fixtures={stats['fixtures_upserted']} "
                f"odds={stats['odds_snapshots_inserted']} "
                f"errors={stats['errors']}[/{color}]"
            )

        # Print summary table
        console.print()
        table = Table(title="Ingestion Summary")
        table.add_column("Date")
        table.add_column("Events", justify="right")
        table.add_column("Tracked", justify="right")
        table.add_column("Fixtures", justify="right")
        table.add_column("Odds", justify="right")
        table.add_column("Errors", justify="right")

        total_errors = 0
        for d_str, s in all_stats:
            total_errors += s["errors"]
            table.add_row(
                d_str,
                str(s["events_fetched"]),
                str(s["events_tracked"]),
                str(s["fixtures_upserted"]),
                str(s["odds_snapshots_inserted"]),
                str(s["errors"]),
            )
        console.print(table)

        if total_errors > 0:
            console.print(f"[yellow]⚠ {total_errors} total error(s). Check ingestion_errors table.[/yellow]")
        else:
            console.print("[green]✓ All done, no errors.[/green]")
    else:
        console.print(f"[red]Unknown source: {source}. Available: sofascore[/red]")
        raise typer.Exit(1)


@app.command()
def settle(
    source: str = typer.Option("sofascore", help="Data source key"),
    date_str: Optional[str] = typer.Option(None, "--date", help="Settle only fixtures from this date YYYY-MM-DD"),
    log_level: str = typer.Option("INFO", "--log-level", help="Logging level"),
) -> None:
    """Fetch results for finished matches and settle fixtures."""
    _setup_logging(log_level)

    label = date_str or "all pending"
    console.print(f"[bold]Source:[/bold]  {source}")
    console.print(f"[bold]Scope:[/bold]   {label}")
    console.print()

    if source == "sofascore":
        if date_str:
            from dhx.ingestion.results import run_bulk

            console.print("[cyan]▶ Settling results (bulk mode)...[/cyan]")
            stats = run_bulk(date_str)
        else:
            from dhx.ingestion.results import run as run_results

            console.print("[cyan]▶ Checking for results...[/cyan]")
            stats = run_results(date_str)

        color = "green" if stats["errors"] == 0 else "yellow"
        console.print(
            f"  [{color}]✓ checked={stats['fixtures_checked']} "
            f"settled={stats['results_inserted']} "
            f"updated={stats['fixtures_updated']} "
            f"pending={stats['still_pending']} "
            f"errors={stats['errors']}[/{color}]"
        )

        # Summary table
        console.print()
        table = Table(title="Settlement Summary")
        table.add_column("Metric")
        table.add_column("Count", justify="right")
        table.add_row("Fixtures checked", str(stats["fixtures_checked"]))
        table.add_row("Results inserted", str(stats["results_inserted"]))
        table.add_row("Fixtures updated", str(stats["fixtures_updated"]))
        table.add_row("Still pending", str(stats["still_pending"]))
        table.add_row("Errors", str(stats["errors"]))
        console.print(table)

        if stats["errors"] > 0:
            console.print(f"[yellow]⚠ {stats['errors']} error(s). Check ingestion_errors table.[/yellow]")
        elif stats["results_inserted"] == 0 and stats["fixtures_checked"] == 0:
            console.print("[dim]No pending fixtures found to settle.[/dim]")
        else:
            console.print("[green]✓ Settlement complete.[/green]")
    else:
        console.print(f"[red]Unknown source: {source}. Available: sofascore[/red]")
        raise typer.Exit(1)


@app.command()
def backfill(
    start_date: str = typer.Option(..., "--start-date", help="Start date YYYY-MM-DD (inclusive)"),
    end_date: Optional[str] = typer.Option(None, "--end-date", help="End date YYYY-MM-DD (inclusive, default: today UTC)"),
    league: Optional[str] = typer.Option(None, "--league", help="Only backfill this league key (e.g. BL2, CH, EPL)"),
    skip_odds: bool = typer.Option(False, "--skip-odds", help="Skip fetching odds"),
    skip_stats: bool = typer.Option(False, "--skip-stats", help="Skip fetching match statistics"),
    skip_lineups: bool = typer.Option(False, "--skip-lineups", help="Skip fetching player lineups/ratings"),
    log_level: str = typer.Option("INFO", "--log-level", help="Logging level"),
) -> None:
    """Backfill historical data from SofaScore for a date range.

    Iterates through all 5 tracked leagues, fetches past events from
    SofaScore tournament endpoints, and populates fixtures, results,
    odds, match statistics, and player ratings.

    Idempotent — safe to re-run. Skips data that already exists.
    """
    _setup_logging(log_level)

    if end_date is None:
        end_date_val = date.today().isoformat()
    else:
        end_date_val = end_date

    # Validate dates
    try:
        datetime.strptime(start_date, "%Y-%m-%d")
        datetime.strptime(end_date_val, "%Y-%m-%d")
    except ValueError as e:
        console.print(f"[red]Invalid date format: {e}. Use YYYY-MM-DD[/red]")
        raise typer.Exit(1)

    console.print("[bold]═══ DHextra Historical Backfill ═══[/bold]")
    console.print(f"[bold]Range:[/bold]   {start_date} → {end_date_val}")
    console.print(f"[bold]Odds:[/bold]    {'skip' if skip_odds else 'fetch'}")
    console.print(f"[bold]Stats:[/bold]   {'skip' if skip_stats else 'fetch'}")
    console.print(f"[bold]Lineups:[/bold] {'skip' if skip_lineups else 'fetch'}")
    console.print()

    from dhx.ingestion.backfill import run as run_backfill

    # Progress callback for live updates
    def on_progress(league_key: str, idx: int, total: int, event: dict) -> None:
        home = event.get("homeTeam", {}).get("shortName", "?")
        away = event.get("awayTeam", {}).get("shortName", "?")
        console.print(
            f"  [dim]{league_key}[/dim] [{idx}/{total}] {home} vs {away}",
        )

    if league:
        console.print(f"[bold]League:[/bold]  {league}")
    console.print("[cyan]▶ Starting backfill...[/cyan]")
    console.print()
    stats = run_backfill(
        start_date,
        end_date_val,
        skip_odds=skip_odds,
        skip_stats=skip_stats,
        skip_lineups=skip_lineups,
        progress_callback=on_progress,
        only_league=league,
    )

    # Summary table
    console.print()
    table = Table(title="Backfill Summary")
    table.add_column("Metric")
    table.add_column("Count", justify="right")

    table.add_row("Leagues processed", str(stats["leagues_processed"]))
    table.add_row("Events found", str(stats["events_found"]))
    table.add_row("Fixtures upserted", str(stats["fixtures_upserted"]))
    table.add_row("Results upserted", str(stats["results_upserted"]))
    table.add_row("Odds snapshots", str(stats["odds_snapshots_inserted"]))
    table.add_row("Match statistics", str(stats["statistics_upserted"]))
    table.add_row("Player ratings", str(stats["player_ratings_upserted"]))
    table.add_row("Skipped (odds)", str(stats["skipped_odds"]))
    table.add_row("Skipped (stats)", str(stats["skipped_stats"]))
    table.add_row("Skipped (lineups)", str(stats["skipped_lineups"]))
    table.add_row("Errors", str(stats["errors"]))
    console.print(table)

    # Per-league breakdown
    if stats.get("per_league"):
        console.print()
        league_table = Table(title="Per-League Breakdown")
        league_table.add_column("League")
        league_table.add_column("Events", justify="right")
        league_table.add_column("Fixtures", justify="right")
        league_table.add_column("Results", justify="right")
        league_table.add_column("Odds", justify="right")
        league_table.add_column("Stats", justify="right")
        league_table.add_column("Lineups", justify="right")
        league_table.add_column("Errors", justify="right")

        for league_key, ls in stats["per_league"].items():
            league_table.add_row(
                league_key,
                str(ls.get("events_found", 0)),
                str(ls.get("fixtures", 0)),
                str(ls.get("results", 0)),
                str(ls.get("odds", 0)),
                str(ls.get("statistics", 0)),
                str(ls.get("lineups", 0)),
                str(ls.get("errors", 0)),
            )
        console.print(league_table)

    if stats["errors"] > 0:
        console.print(f"\n[yellow]⚠ {stats['errors']} error(s). Check ingestion_errors table.[/yellow]")
    else:
        console.print("\n[green]✓ Backfill complete, no errors.[/green]")


if __name__ == "__main__":
    app()
