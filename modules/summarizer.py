from pathlib import Path

from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler
from tqdm import tqdm

CONFIG_DIR = Path(__file__).resolve().parent.parent / "config"
DEFAULT_PROMPT = CONFIG_DIR / "prompt.txt"
DEFAULT_MEMBERS = CONFIG_DIR / "members.txt"


def load_members(path: str | Path) -> list[str]:
    """Read full names, one per line; blank lines and lines starting with # are skipped."""
    path = Path(path)
    if not path.is_file():
        return []
    lines = (line.strip() for line in path.read_text(encoding="utf-8").splitlines())
    return [line for line in lines if line and not line.startswith("#")]

THOUGHT_START = "<|channel>"
THOUGHT_END = "<channel|>"


def strip_thinking(text: str) -> str:
    """Remove Gemma's <|channel>thought ... <channel|> blocks, including an unfinished one."""
    result = []
    for part in text.split(THOUGHT_END):
        result.append(part.split(THOUGHT_START)[0])
    return "".join(result).strip()


class Summarizer:
    """Creates meeting minutes from a transcript using a local MLX LLM.

    The instructions are read from a text file and sent as the system message;
    the placeholders {language} and {members} are filled in, any other text is
    kept as is. The transcript is sent as a separate user message.
    """

    def __init__(
        self,
        model: str = "mlx-community/gemma-4-e4b-it-4bit",
        temperature: float = 0.2,
        prompt_path: str | Path = DEFAULT_PROMPT,
        members_path: str | Path | None = None,
        max_tokens: int = 4096,
    ):
        self.max_tokens = max_tokens
        self.prompt_template = Path(prompt_path).read_text(encoding="utf-8")
        self.members = load_members(members_path or DEFAULT_MEMBERS)
        self.model, self.tokenizer = load(model)
        self.sampler = make_sampler(temp=temperature)

    def build_messages(self, transcript: str, language: str) -> list[dict[str, str]]:
        members = "\n".join(f"- {name}" for name in self.members) or "(no member list available)"
        instructions = (
            self.prompt_template
            .replace("{language}", language)
            .replace("{members}", members)
        )
        return [
            {"role": "system", "content": instructions},
            {"role": "user", "content": transcript},
        ]

    def summarize(self, transcript: str, language: str = "English") -> str:
        formatted = self.tokenizer.apply_chat_template(
            self.build_messages(transcript, language),
            add_generation_prompt=True,
            tokenize=False,
        )
        # The length of the answer is unknown, so the bar counts generated tokens.
        output = []
        with tqdm(desc="Writing minutes", unit=" tokens") as bar:
            for response in stream_generate(
                self.model, self.tokenizer, prompt=formatted,
                sampler=self.sampler, max_tokens=self.max_tokens,
            ):
                output.append(response.text)
                bar.update(1)
        return strip_thinking("".join(output))
