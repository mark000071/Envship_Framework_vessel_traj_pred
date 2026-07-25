#!/usr/bin/env python3
"""Stage 01: field standardization and sentinel handling."""

from __future__ import annotations

from pathlib import Path

from pipeline_utils import (
    ensure_dir,
    list_raw_files,
    load_config,
    parse_stage_args,
    read_csv_chunks,
    standardize_dma_chunk,
    write_json,
    write_stage_csv,
)


def output_name_for_raw(raw_path: Path) -> str:
    name = raw_path.name
    if name.endswith(".csv.gz"):
        name = name[:-7]
    elif name.endswith(".zip"):
        name = name[:-4]
    elif name.endswith(".csv"):
        name = name[:-4]
    return f"{name}.csv.gz"


def main() -> None:
    args = parse_stage_args(__doc__ or "stage 01")
    config = load_config(args.config)
    ensure_dir(args.output)
    files = list_raw_files(args.input)
    chunksize = int(config["io"]["chunksize"])
    manifest = []

    for raw_path in files:
        out_path = args.output / output_name_for_raw(raw_path)
        wrote_header = False
        input_rows = 0
        output_rows = 0
        for chunk in read_csv_chunks(raw_path, chunksize):
            input_rows += len(chunk)
            standardized = standardize_dma_chunk(chunk, raw_path.name, config)
            output_rows += len(standardized)
            standardized.to_csv(
                out_path,
                mode="w" if not wrote_header else "a",
                header=not wrote_header,
                index=False,
                compression="gzip",
            )
            wrote_header = True
        manifest.append(
            {
                "input_file": str(raw_path),
                "output_file": str(out_path),
                "input_rows": input_rows,
                "output_rows": output_rows,
            }
        )

    manifest_path = args.output / ("manifest.json" if args.input.is_dir() else f"{args.input.stem}.manifest.json")
    write_json(manifest_path, {"files": manifest, "stage": "01_standardize_fields"})
    print(f"standardized_files={len(manifest)}")


if __name__ == "__main__":
    main()
