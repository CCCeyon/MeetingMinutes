import gc
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import numpy as np
import torch
from parakeet_mlx import from_pretrained
from parakeet_mlx.audio import load_audio
from pyannote.audio import Pipeline
from tqdm import tqdm

# Share of the speaker-detection bar per pyannote step, roughly by runtime
# (computing the speaker embeddings takes by far the longest).
DIARIZATION_WEIGHTS = {
    "segmentation": 10,
    "speaker_counting": 0,
    "embeddings": 88,
    "discrete_diarization": 2,
}


class SpeakerProgress:
    """pyannote hook that combines all its internal steps into one tqdm bar."""

    def __init__(self):
        self.bar = tqdm(desc="Detecting speakers", total=100, bar_format="{l_bar}{bar}| [{elapsed}<{remaining}]")
        self.done_before = {}  # step -> percentage of all steps before it
        offset = 0
        for step, weight in DIARIZATION_WEIGHTS.items():
            self.done_before[step] = offset
            offset += weight

    def __call__(self, step_name, step_artifact, file=None, total=None, completed=None):
        if step_name not in DIARIZATION_WEIGHTS:
            return
        fraction = 1.0 if not total or completed is None else completed / total
        self.bar.n = self.done_before[step_name] + DIARIZATION_WEIGHTS[step_name] * min(fraction, 1.0)
        self.bar.refresh()

    def close(self) -> None:
        self.bar.n = 100
        self.bar.refresh()
        self.bar.close()


@dataclass
class Segment:
    speaker: str
    start: float
    end: float
    text: str

    def __str__(self) -> str:
        m, s = divmod(int(self.start), 60)
        m2, s2 = divmod(int(self.end), 60)
        return f"{self.speaker} [{m:02d}:{s:02d} - {m2:02d}:{s2:02d}]: {self.text}"


class Transcriber:
    """Speech-to-text (Parakeet) + speaker diarization (pyannote)."""

    def __init__(
        self,
        asr_model: str = "mlx-community/parakeet-tdt-0.6b-v3",
        diarization_model: str = "pyannote/speaker-diarization-community-1",
        device: str | None = None,
        chunk_duration: float = 120.0,
        overlap_duration: float = 15.0,
    ):
        # Long recordings are transcribed in overlapping chunks; processing a whole
        # meeting at once needs far more GPU memory than is available.
        self.chunk_duration = chunk_duration
        self.overlap_duration = overlap_duration
        self.asr = from_pretrained(asr_model)
        self.diarization = Pipeline.from_pretrained(diarization_model)
        if device is None:
            device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.device = device
        self.diarization.to(torch.device(device))

    def __enter__(self) -> "Transcriber":
        return self

    def __exit__(self, *exc) -> None:
        self.unload()

    def unload(self) -> None:
        """Release both models and return their memory to the system."""
        self.asr = None
        self.diarization = None
        gc.collect()
        mx.clear_cache()
        if self.device == "mps":
            torch.mps.empty_cache()

    def transcribe(self, audio_path: str, names: dict[str, str] | None = None) -> list[Segment]:
        if self.asr is None or self.diarization is None:
            raise RuntimeError("Transcriber models have been unloaded.")
        names = names or {}
        result = self._run_asr(audio_path)

        progress = SpeakerProgress()
        try:
            diarization = self.diarization(self._load_waveform(audio_path), hook=progress)
        finally:
            progress.close()
        turns = [(t.start, t.end, spk) for t, spk in diarization.speaker_diarization]

        segments = []
        for sent in result.sentences:
            spk = self._speaker_for(turns, sent.start, sent.end)
            segments.append(Segment(names.get(spk, spk), sent.start, sent.end, sent.text))
        return segments

    @staticmethod
    def _load_waveform(audio_path: str, sample_rate: int = 16000) -> dict:
        """Decode the audio once (via ffmpeg) for pyannote.

        Passing a file path lets pyannote decode it itself, which fails on
        compressed formats like mp3/m4a when seeking returns a few samples too few.
        """
        audio = load_audio(Path(audio_path), sample_rate, dtype=mx.float32)
        waveform = torch.from_numpy(np.array(audio, copy=True)).unsqueeze(0)
        return {"waveform": waveform, "sample_rate": sample_rate}

    def _run_asr(self, audio_path: str):
        """Run Parakeet chunk by chunk with a progress bar in seconds of audio."""
        sample_rate = self.asr.preprocessor_config.sample_rate
        with tqdm(desc="Transcribing", unit="s", bar_format="{l_bar}{bar}| {n}/{total}s [{elapsed}<{remaining}]") as bar:
            previous_end = 0

            def on_chunk(chunk_end: int, total: int) -> None:
                # Called *before* each chunk is processed, so the finished audio
                # reaches up to the end of the previous chunk.
                nonlocal previous_end
                bar.total = round(total / sample_rate)
                bar.n = round(previous_end / sample_rate)
                bar.refresh()
                previous_end = chunk_end

            result = self.asr.transcribe(
                audio_path,
                chunk_duration=self.chunk_duration,
                overlap_duration=self.overlap_duration,
                chunk_callback=on_chunk,
            )
            # Short recordings are processed in one piece without any callback.
            if bar.total is None:
                bar.total = round(result.sentences[-1].end) if result.sentences else 0
            bar.n = bar.total
            bar.refresh()
        return result

    @staticmethod
    def _speaker_for(turns, start: float, end: float) -> str:
        best, best_overlap = "UNKNOWN", 0.0
        for s, e, spk in turns:
            overlap = min(end, e) - max(start, s)
            if overlap > best_overlap:
                best, best_overlap = spk, overlap
        return best
