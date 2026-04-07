#!/usr/bin/env python3
"""
Aggregate extracted Citi Bike trip files into tidy hourly ride counts.

The script expects extracted monthly Citi Bike trip files in CSV or parquet
format. It scans them lazily with Polars, counts rides by trip start date and
hour, fills in missing hourly buckets with zeroes, and writes a compact output
file that is small enough to keep after the raw trip files are deleted.

Example
-------
python3 grf/data_cleaning.py citi_bike/extracted_files \\
    --output citi_bike/citibike_hourly_rides_2022_2025.parquet
"""

from __future__ import annotations

import argparse
import gzip
import io
from datetime import date
from pathlib import Path
from typing import Iterable
from urllib.error import HTTPError, URLError
from urllib.request import urlopen

import polars as pl

DEFAULT_START_DATE = date(2022, 1, 1)
DEFAULT_END_DATE = date(2025, 12, 31)
DEFAULT_INPUT_PATH = '/Users/mm/Documents/Data Science/data/citi_bike/extracted_files/'
#DEFAULT_INPUT_PATH = Path(__file__).resolve().parent.parent / "citi_bike" / "extracted_files"
DEFAULT_WEATHER_STATION = "72503"
METEOSTAT_DAILY_URL = "https://data.meteostat.net/daily/{year}/{station}.csv.gz"
SUPPORTED_EXTENSIONS = {".csv", ".parquet"}


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description=(
            "Aggregate extracted Citi Bike trip files into hourly ride counts."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "input_path",
        nargs="?",
        type=Path,
        default=DEFAULT_INPUT_PATH,
        help=(
            "Directory containing extracted Citi Bike files, or a single file. "
            f"Defaults to {DEFAULT_INPUT_PATH}."
        ),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output file path. Use .parquet or .csv. Defaults next to the input.",
    )
    parser.add_argument(
        "--timestamp-column",
        default="started_at",
        help="Trip start timestamp column to aggregate from.",
    )
    parser.add_argument(
        "--start-date",
        type=date.fromisoformat,
        default=DEFAULT_START_DATE,
        help="Inclusive start date for the analysis window.",
    )
    parser.add_argument(
        "--end-date",
        type=date.fromisoformat,
        default=DEFAULT_END_DATE,
        help="Inclusive end date for the analysis window.",
    )
    parser.add_argument(
        "--weather-station",
        default=DEFAULT_WEATHER_STATION,
        help="Meteostat weather station ID used for daily weather enrichment.",
    )
    parser.add_argument(
        "--weather-cache",
        type=Path,
        default=None,
        help=(
            "Local weather cache path. If the file exists it is reused; "
            "otherwise Meteostat data is downloaded once and saved here."
        ),
    )
    return parser.parse_args()


def default_artifact_dir(input_path: Path) -> Path:
    """Choose a default output/cache directory near the raw files, not inside them."""
    if input_path.exists():
        base_dir = input_path if input_path.is_dir() else input_path.parent
    else:
        base_dir = input_path if not input_path.suffix else input_path.parent

    if base_dir.name == "extracted_files":
        return base_dir.parent

    return base_dir


def output_year_label(start_date: date, end_date: date) -> str:
    """Build a compact year label for default output file names."""
    if start_date.year == end_date.year:
        return str(start_date.year)

    return f"{start_date.year}_{end_date.year}"


def default_output_path(
    input_path: Path,
    start_date: date,
    end_date: date,
) -> Path:
    """Choose a default output path near the raw files, not inside them."""
    base_dir = default_artifact_dir(input_path)
    return base_dir / f"citibike_hourly_rides_{output_year_label(start_date, end_date)}.parquet"


def default_weather_cache_path(
    input_path: Path,
    station_id: str,
    start_date: date,
    end_date: date,
) -> Path:
    """Choose a default cache location for Meteostat daily weather data."""
    base_dir = default_artifact_dir(input_path)
    return (
        base_dir
        / f"meteostat_daily_{station_id}_{start_date.isoformat()}_{end_date.isoformat()}.parquet"
    )


def discover_input_files(
    input_path: Path,
    exclude_paths: Iterable[Path] | None = None,
) -> list[Path]:
    """Return supported source files under the provided path."""
    excluded = {
        path.expanduser().resolve()
        for path in (exclude_paths or [])
    }

    if not input_path.exists():
        raise FileNotFoundError(f"Input path does not exist: {input_path}")

    if input_path.is_file():
        resolved_input = input_path.expanduser().resolve()
        if resolved_input in excluded:
            raise FileNotFoundError(
                f"Input file is excluded from discovery: {input_path}"
            )
        if input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported file type: {input_path.name}. "
                f"Supported extensions: {sorted(SUPPORTED_EXTENSIONS)}"
            )
        return [input_path]

    source_files = sorted(
        path
        for path in input_path.rglob("*")
        if (
            path.is_file()
            and path.suffix.lower() in SUPPORTED_EXTENSIONS
            and path.expanduser().resolve() not in excluded
        )
    )

    if source_files:
        return source_files

    zip_files = sorted(path for path in input_path.rglob("*.zip") if path.is_file())
    if zip_files:
        raise FileNotFoundError(
            "No extracted CSV/parquet files were found. The directory still "
            "contains zip archives, so unzip the Citi Bike downloads first."
        )

    raise FileNotFoundError(
        "No supported input files were found. Expected extracted .csv or "
        ".parquet Citi Bike trip files."
    )


def scan_trip_file(file_path: Path, timestamp_column: str) -> pl.LazyFrame:
    """Read one trip file and normalize the trip start timestamp."""
    if file_path.suffix.lower() == ".csv":
        trip_frame = pl.scan_csv(file_path, try_parse_dates=False)
    elif file_path.suffix.lower() == ".parquet":
        trip_frame = pl.scan_parquet(file_path)
    else:
        raise ValueError(f"Unsupported file type: {file_path}")

    return (
        trip_frame.select(
            pl.col(timestamp_column)
            .cast(pl.Utf8, strict=False)
            .str.to_datetime(strict=False)
            .alias("started_at")
        )
        .drop_nulls()
        .with_columns(
            pl.col("started_at").dt.date().alias("ride_date"),
            pl.col("started_at").dt.hour().alias("ride_hour"),
        )
        .select("started_at", "ride_date", "ride_hour")
    )


def aggregate_hourly_rides(
    file_paths: Iterable[Path],
    timestamp_column: str,
    start_date: date,
    end_date: date,
) -> pl.DataFrame:
    """Aggregate all rides into date x hour counts for the requested window."""
    ride_frames = [scan_trip_file(path, timestamp_column) for path in file_paths]
    combined_trips = pl.concat(ride_frames, how="vertical_relaxed")

    hourly_counts = (
        combined_trips.filter(
            pl.col("ride_date").is_between(start_date, end_date, closed="both")
        )
        .group_by("ride_date", "ride_hour")
        .agg(pl.col("started_at").count().alias("ride_count"))
        .collect()
    )

    return build_complete_hourly_grid(hourly_counts, start_date, end_date)


def build_complete_hourly_grid(
    hourly_counts: pl.DataFrame,
    start_date: date,
    end_date: date,
) -> pl.DataFrame:
    """Fill missing hours with zero rides so the output is analysis-ready."""
    date_frame = pl.date_range(
        start=start_date,
        end=end_date,
        interval="1d",
        eager=True,
    ).alias("ride_date").to_frame()
    hour_frame = pl.DataFrame(
        {"ride_hour": list(range(24))},
        schema={"ride_hour": pl.Int64},
    )
    hourly_counts = hourly_counts.with_columns(pl.col('ride_hour').cast(pl.Int64))


    return (
        date_frame.join(hour_frame, how="cross")
        .join(hourly_counts, on=["ride_date", "ride_hour"], how="left")
        .with_columns(
            pl.col("ride_count").fill_null(0).cast(pl.UInt32),
            (
                pl.col("ride_date").cast(pl.Datetime)
                + pl.duration(hours=pl.col("ride_hour"))
            ).alias("hour_start")
        )
        .select("hour_start", "ride_date", "ride_hour", "ride_count")
        .sort("hour_start")
    )


def read_weather_cache(weather_cache_path: Path) -> pl.DataFrame:
    """Read a cached weather file from disk."""
    if weather_cache_path.suffix.lower() == ".parquet":
        return pl.read_parquet(weather_cache_path)

    if weather_cache_path.suffix.lower() == ".csv":
        return pl.read_csv(weather_cache_path, try_parse_dates=True)

    raise ValueError("Weather cache path must end in .parquet or .csv.")


def write_weather_cache(weather_data: pl.DataFrame, weather_cache_path: Path) -> None:
    """Persist the downloaded weather data for reuse."""
    weather_cache_path.parent.mkdir(parents=True, exist_ok=True)

    if weather_cache_path.suffix.lower() == ".parquet":
        weather_data.write_parquet(weather_cache_path, compression="zstd")
        return

    if weather_cache_path.suffix.lower() == ".csv":
        weather_data.write_csv(weather_cache_path)
        return

    raise ValueError("Weather cache path must end in .parquet or .csv.")


def weather_date_expr(raw_weather: pl.DataFrame) -> pl.Expr:
    """Infer the weather date column across cached and raw Meteostat schemas."""
    weather_columns = set(raw_weather.columns)

    for column_name in ("ride_date", "date", "time"):
        if column_name in weather_columns:
            return pl.coalesce(
                [
                    pl.col(column_name).cast(pl.Date, strict=False),
                    pl.col(column_name)
                    .cast(pl.Utf8, strict=False)
                    .str.to_date(strict=False),
                ]
            )

    if {"year", "month", "day"}.issubset(weather_columns):
        return pl.date(
            year=pl.col("year").cast(pl.Int32, strict=False),
            month=pl.col("month").cast(pl.Int32, strict=False),
            day=pl.col("day").cast(pl.Int32, strict=False),
        )

    raise ValueError(
        "Unable to infer a weather date column. Expected one of "
        "`ride_date`, `date`, `time`, or the `year`/`month`/`day` triplet."
    )


def normalize_weather_data(
    raw_weather: pl.DataFrame,
    start_date: date,
    end_date: date,
) -> pl.DataFrame:
    """Keep the weather columns used in the Citi Bike output."""
    return (
        raw_weather.select(
            weather_date_expr(raw_weather).alias("ride_date"),
            pl.col("tmin").cast(pl.Float64, strict=False).alias("weather_tmin_c"),
            pl.col("tmax").cast(pl.Float64, strict=False).alias("weather_tmax_c"),
            pl.col("prcp").cast(pl.Float64, strict=False).alias("weather_prcp_mm"),
        )
        .drop_nulls(subset=["ride_date"])
        .filter(pl.col("ride_date").is_between(start_date, end_date, closed="both"))
        .sort("ride_date")
        .unique(subset=["ride_date"], keep="last")
    )


def download_meteostat_daily_weather(
    station_id: str,
    start_date: date,
    end_date: date,
) -> pl.DataFrame:
    """Download daily Meteostat weather data for the requested station and years."""
    yearly_frames: list[pl.DataFrame] = []

    for year in range(start_date.year, end_date.year + 1):
        url = METEOSTAT_DAILY_URL.format(year=year, station=station_id)
        try:
            with urlopen(url, timeout=60) as response:
                compressed_bytes = response.read()
        except HTTPError as exc:
            raise RuntimeError(
                f"Failed to download Meteostat weather data for station {station_id} "
                f"and year {year}: HTTP {exc.code}."
            ) from exc
        except URLError as exc:
            raise RuntimeError(
                f"Failed to download Meteostat weather data for station {station_id} "
                f"and year {year}: {exc.reason}."
            ) from exc

        try:
            csv_bytes = gzip.decompress(compressed_bytes)
        except OSError:
            csv_bytes = compressed_bytes

        yearly_frames.append(pl.read_csv(io.BytesIO(csv_bytes), try_parse_dates=True))

    combined_weather = pl.concat(yearly_frames, how="vertical_relaxed")
    return normalize_weather_data(combined_weather, start_date, end_date)


def load_or_create_weather_cache(
    weather_cache_path: Path,
    station_id: str,
    start_date: date,
    end_date: date,
) -> pl.DataFrame:
    """Reuse cached weather data when available, otherwise download it once."""
    if weather_cache_path.exists():
        return normalize_weather_data(
            read_weather_cache(weather_cache_path),
            start_date,
            end_date,
        )

    weather_data = download_meteostat_daily_weather(
        station_id=station_id,
        start_date=start_date,
        end_date=end_date,
    )
    write_weather_cache(weather_data, weather_cache_path)
    return weather_data


def attach_daily_weather(
    hourly_rides: pl.DataFrame,
    weather_data: pl.DataFrame,
) -> pl.DataFrame:
    """Join daily weather features onto the hourly ride grid."""
    return (
        hourly_rides.join(weather_data, on="ride_date", how="left")
        .select(
            "hour_start",
            "ride_date",
            "ride_hour",
            "ride_count",
            "weather_tmin_c",
            "weather_tmax_c",
            "weather_prcp_mm",
        )
        .sort("hour_start")
    )


def write_output(result: pl.DataFrame, output_path: Path) -> None:
    """Write the final aggregate in parquet or CSV format."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if output_path.suffix.lower() == ".parquet":
        result.write_parquet(output_path, compression="zstd")
        return

    if output_path.suffix.lower() == ".csv":
        result.write_csv(output_path)
        return

    raise ValueError("Output path must end in .parquet or .csv.")


def main() -> None:
    """Run the aggregation end to end."""
    args = parse_args()

    if args.start_date > args.end_date:
        raise ValueError("--start-date must be on or before --end-date.")

    output_path = args.output or default_output_path(
        input_path=args.input_path,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    weather_cache_path = args.weather_cache or default_weather_cache_path(
        input_path=args.input_path,
        station_id=args.weather_station,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    input_files = discover_input_files(
        args.input_path,
        exclude_paths=[output_path, weather_cache_path],
    )

    print(f"Found {len(input_files)} source files.")
    print(
        f"Aggregating rides between {args.start_date.isoformat()} and "
        f"{args.end_date.isoformat()} using `{args.timestamp_column}`."
    )

    result = aggregate_hourly_rides(
        file_paths=input_files,
        timestamp_column=args.timestamp_column,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    weather_data = load_or_create_weather_cache(
        weather_cache_path=weather_cache_path,
        station_id=args.weather_station,
        start_date=args.start_date,
        end_date=args.end_date,
    )
    result = attach_daily_weather(result, weather_data)
    write_output(result, output_path)

    print(
        f"Weather cache ready at {weather_cache_path} using Meteostat station "
        f"`{args.weather_station}`."
    )
    print(f"Wrote {result.height:,} hourly rows to {output_path}")


if __name__ == "__main__":
    main()
