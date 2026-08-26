import re
from pathlib import Path
import os
from concurrent.futures import ProcessPoolExecutor, as_completed

NUM_WORKERS = os.cpu_count() or 4

output_path = "../data/wowah_parsed_mp.csv"
root_dir = "/Users/rcap/work/sessionization/WoWAH"

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
