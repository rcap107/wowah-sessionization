import re
from datetime import datetime
from pathlib import Path
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

import polars as pl

NUM_WORKERS = os.cpu_count() or 4

output_path = "../data/wowah_parsed_mp.csv"
root_dir = "~/work/sessionization/WoWAH"

# Define regex pattern parts
timestamp_pattern = r"\d+/\d+/\d+ \d+:\d+:\d+"
digits = r"\d+"
optional_digits = r"\d*"
text_field = r"[A-Za-z\' ]+"
optional_text_field = r"[A-Za-z\' ]*"

# Combine into full line regex
line_re = re.compile(
    rf"^.*\"{digits},\s({timestamp_pattern}),\s({digits}),({digits}),\s*({optional_digits}),\s({digits}),\s({text_field}),\s({text_field}),\s({text_field}),\s({optional_text_field}),\s{digits}\".*$"
)


def create_output_file(path):
    with open(path, "w") as f:
        header = "timestamp,avatar_id,guild,level,race,charclass,zone\n"
        f.write(header)


def process_single_file(file_path):
    matched_lines = []
    match_count = 0
    try:
        with open(file_path, "r") as f:
            for line in f:
                matched = re.match(line_re, line)

                if matched:
                    timestamp = matched.group(1)
                    avatar_id = matched.group(3)
                    guild = matched.group(4) if matched.group(4) else ""
                    level = matched.group(5)
                    race = matched.group(6)
                    charclass = matched.group(7)
                    zone = matched.group(8)
                    clean_elements = [
                        timestamp,
                        avatar_id,
                        guild,
                        level,
                        race,
                        charclass,
                        zone,
                    ]
                    new_line = ",".join(clean_elements) + "\n"
                    matched_lines.append(new_line)
                    match_count += 1
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

    return matched_lines, match_count


def worker_wrapper(file_path):
    """
    Wrapper to return matched lines and count for multiprocessing.
    """
    return process_single_file(file_path)


# Column names of a parsed row, in the order produced by `process_single_file`.
PARSED_COLUMNS = ["timestamp", "char", "guild", "level", "race", "charclass", "zone"]


def list_raw_files(root_dir, start_date=None, end_date=None):
    """
    List the raw WoWAH heartbeat ``.txt`` files under `root_dir`, optionally
    restricted to the inclusive [start_date, end_date] range.

    Each file lives in a "YYYY-MM-DD" directory (e.g.
    ``2008_10_12/2008-11-03/12-03-48.txt``), which is what filtering is based
    on: `root_dir` itself is organised into quarterly folders (e.g.
    ``2008_10_12``) that span several months, so filtering on the folder name
    alone would not be precise enough to carve out, say, "historical" vs.
    "held-out" files on a monthly boundary.
    """
    file_paths = []
    for file_path in Path(root_dir).expanduser().rglob("*.txt"):
        try:
            day = datetime.strptime(file_path.parent.name, "%Y-%m-%d")
        except ValueError:
            continue
        if start_date is not None and day < start_date:
            continue
        if end_date is not None and day > end_date:
            continue
        file_paths.append(str(file_path))
    return sorted(file_paths)


def parse_files(file_paths):
    """
    Parse a list of raw WoWAH heartbeat ``.txt`` files into a single typed
    Polars DataFrame.

    This does the same thing as the CLI entrypoint, but instead of writing to
    CSV, it returns the parsed table directly. This allows to use this function
    as a node in the DataOp pipeline, so that the raw-log parsing step is part
    of the pipeline (like it would be in a real-time scenario), rather than
    a one-off script run ahead of time.
    """
    rows = {col: [] for col in PARSED_COLUMNS}

    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        # executor.map preserves the order of `file_paths`, so the resulting
        # table is built in a deterministic order.
        for matched_lines, _match_count in executor.map(
            process_single_file, file_paths
        ):
            for line in matched_lines:
                for col, value in zip(PARSED_COLUMNS, line.rstrip("\n").split(",")):
                    rows[col].append(value)

    df = pl.DataFrame(rows, schema={col: pl.Utf8 for col in PARSED_COLUMNS})
    df = df.with_columns(
        pl.col("timestamp").str.strptime(pl.Datetime, format="%m/%d/%y %H:%M:%S"),
        pl.col("char").cast(pl.Int32, strict=False),
        pl.col("guild").cast(pl.Int32, strict=False),
        pl.col("level").cast(pl.Int32, strict=False),
    ).sort(["timestamp", "char"])
    return df


def main():
    create_output_file(output_path)
    unmatched_files_log = "../data/no_matches_files.log"

    file_paths = [str(fp) for fp in Path(root_dir).rglob("*.txt")]
    # Process in parallel and stream to output
    with ProcessPoolExecutor(max_workers=NUM_WORKERS) as executor:
        # Submit all tasks
        future_to_file = {executor.submit(worker_wrapper, fp): fp for fp in file_paths}

        with (
            open(output_path, "a", encoding="utf-8") as outfile,
            open(unmatched_files_log, "w", encoding="utf-8") as logfile,
        ):
            completed_count = 0
            files_with_no_matches = 0
            for future in as_completed(future_to_file):
                file_path = future_to_file[future]
                try:
                    results, match_count = future.result()
                    if results and match_count > 0:
                        # Write results immediately to disk
                        for line in results:
                            outfile.write(line)
                    else:
                        # Log files with no matches
                        logfile.write(f"{file_path}\n")
                        files_with_no_matches += 1

                    completed_count += 1
                    if completed_count % 1000 == 0:
                        print(f"Processed {completed_count}/{len(file_paths)} files...")
                except Exception as exc:
                    print(f"File {file_path} generated an exception: {exc}")

    print(f"Done! Merged file saved to {output_path}")
    print(f"Files with no matches: {files_with_no_matches}")
    print(f"Log file: {unmatched_files_log}")


if __name__ == "__main__":
    main()
