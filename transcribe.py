import os.path
from pathlib import Path

from dotenv import load_dotenv

from cli import parse_args
from modules import Summarizer, Transcriber


def save(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text + "\n", encoding="utf-8")
    print(f"Saved {path}")


def main():
    args = parse_args()
    load_dotenv()
    stem = Path(args.audio).stem

    # Models are unloaded when the block ends, freeing memory for the LLM.
    with Transcriber() as transcriber:
        segments = transcriber.transcribe(args.audio, names=args.names)
    transcript = "\n".join(str(seg) for seg in segments)
    #print(transcript)
    save(args.output_dir / f"{stem}_transcript.txt", transcript)

    summarizer = Summarizer(members_path=args.members)
    print(f"Loaded {len(summarizer.members)} member names.")
    minutes = summarizer.summarize(transcript, language=args.language)
    #print(minutes)
    save(args.output_dir / f"{stem}_minutes.txt", minutes)


if __name__ == "__main__":
    main()
