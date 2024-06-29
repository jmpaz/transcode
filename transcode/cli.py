import argparse
import concurrent.futures
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import List, Tuple

import ffmpeg
import psutil
from rich.console import Console
from rich.live import Live
from rich.logging import RichHandler
from rich.panel import Panel
from rich.progress import BarColumn, Progress, TextColumn
from rich.text import Text

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger("transcode")


def setup_file_logger(output_dir: Path):
    log_dir = output_dir / "logs"
    log_dir.mkdir(exist_ok=True)
    log_file = log_dir / f"transcode_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(
        logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    )
    logger.addHandler(file_handler)
    return log_file


def get_video_files(input_path: Path) -> List[Tuple[Path, int]]:
    if input_path.is_file():
        return [(input_path, input_path.stat().st_size)]
    return [
        (f, f.stat().st_size)
        for f in input_path.glob("*")
        if f.is_file() and f.suffix.lower() in [".mkv", ".mp4", ".avi", ".mov"]
    ]


def transcode_file(input_file: Path, output_dir: Path, format: str):
    output_file = output_dir / f"{input_file.stem}.{format}"

    if output_file.exists():
        logger.info(f"Skipped {input_file.name} (already exists)")
        return True, f"Skipped {input_file.name} (already exists)"

    logger.info(f"Starting transcoding of {input_file.name}")

    if format == "mov":
        output_params = {"vcodec": "prores_ks", "profile:v": "3", "acodec": "pcm_s16le"}
    elif format in ["mp4", "mkv"]:
        output_params = {
            "vcodec": "libx264",
            "crf": "18",
            "preset": "slow",
            "acodec": "aac",
            "b:a": "192k",
        }
    else:
        output_params = {}  # use default parameters for other formats

    try:
        process = (
            ffmpeg.input(str(input_file))
            .output(str(output_file), **output_params)
            .overwrite_output()
            .run_async(pipe_stdout=True, pipe_stderr=True)
        )

        # Capture stdout and stderr
        out, err = process.communicate()

        if process.returncode != 0:
            logger.error(f"FFmpeg error for {input_file.name}:")
            logger.error(err.decode())
            return False, f"Error transcoding {input_file.name}: ffmpeg error"

        logger.info(f"Completed transcoding of {input_file.name}")
        return True, f"Completed transcoding of {input_file.name}"
    except ffmpeg.Error as e:
        logger.exception(f"FFmpeg error for {input_file.name}:")
        logger.error(e.stderr.decode() if e.stderr else "No error output")
        return False, f"Error transcoding {input_file.name}: {str(e)}"
    except Exception as e:
        logger.exception(f"Unexpected error transcoding {input_file.name}: {str(e)}")
        return False, f"Unexpected error transcoding {input_file.name}: {str(e)}"


def cli():
    parser = argparse.ArgumentParser(description="Transcode video files using ffmpeg")
    parser.add_argument(
        "input", type=str, help="Input file or directory containing video files"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Output directory for transcoded files",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="mp4",
        choices=["mov", "mp4", "mkv"],
        help="Output format (default: mp4)",
    )
    parser.add_argument(
        "--max-workers",
        type=int,
        default=None,
        help="Maximum number of worker processes",
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose logging"
    )

    args = parser.parse_args()

    if args.verbose:
        logger.setLevel(logging.DEBUG)

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(exist_ok=True)

    log_file = setup_file_logger(output_dir)
    logger.info(f"Logging to {log_file}")

    video_files = get_video_files(input_path)
    if not video_files:
        logger.error(f"No video files found in {input_path}")
        return

    video_files.sort(key=lambda x: x[1])  # Sort by file size, smallest first

    console = Console()

    # Determine the number of workers
    if args.max_workers is None:
        max_workers = max(
            1, psutil.cpu_count(logical=False) - 1
        )  # Use physical cores - 1
    else:
        max_workers = args.max_workers

    logger.info(f"Starting transcoding process with {max_workers} workers")

    progress = Progress(
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TextColumn("[{task.completed}/{task.total}]"),
        console=console,
        transient=True,
    )

    with Live(progress, console=console, refresh_per_second=4) as live:
        overall_task = progress.add_task("Overall Progress", total=len(video_files))

        with concurrent.futures.ProcessPoolExecutor(
            max_workers=max_workers
        ) as executor:
            futures = [
                executor.submit(transcode_file, video_file, output_dir, args.format)
                for video_file, _ in video_files
            ]

            for future in concurrent.futures.as_completed(futures):
                try:
                    success, message = future.result()
                    progress.update(overall_task, advance=1)
                    if success:
                        console.print(f"[green]{message}")
                    else:
                        console.print(f"[red]{message}")
                except Exception as e:
                    logger.exception(f"An error occurred in a worker process: {str(e)}")
                    console.print(f"[red]Error: {str(e)}")

    logger.info("Transcoding process complete")
    console.print("[green]Transcoding complete!")
    console.print(f"Log file: {log_file}")
