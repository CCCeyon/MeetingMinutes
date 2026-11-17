import argparse
from pathlib import Path


def _speaker_name(value: str) -> tuple[str, str]:
    label, sep, name = value.partition("=")
    if not sep or not label or not name:
        raise argparse.ArgumentTypeError(f"expected LABEL=NAME, got '{value}'")
    return label.strip(), name.strip()


def _existing_file(value: str) -> str:
    if not Path(value).is_file():
        raise argparse.ArgumentTypeError(f"file not found: '{value}'")
    return value


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transcribe a meeting recording with speaker labels and create minutes.",
    )
    parser.add_argument("audio", type=_existing_file, help="path to the audio file, e.g. meeting.m4a")
    parser.add_argument(
        "-l", "--language", default="English",
        help="language of the minutes (default: English)",
    )
    parser.add_argument(
        "-n", "--name", dest="names", type=_speaker_name, action="append", default=[],
        metavar="LABEL=NAME",
        help="rename a speaker, e.g. -n SPEAKER_00=Anna (repeatable)",
    )
    parser.add_argument(
        "-m", "--members", type=_existing_file, default=None,
        help="file with full names of members, one per line (default: config/members.txt)",
    )
    parser.add_argument(
        "-o", "--output-dir", type=Path, default=Path("results"),
        help="folder for the result files (default: results)",
    )

    args = parser.parse_args(argv)
    args.names = dict(args.names)
    return args
