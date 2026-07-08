"""
diversity_metrics.py
====================
Pairwise embedding diversity metric for music generation models.

Measures how diverse multiple outputs from the same prompt are,
using pretrained audio embeddings (MERT or EnCodec).

  Low mean distance  → model is collapsing (outputs sound the same)
  High mean distance → model is diverse

Backends
--------
  mert    : m-a-p/MERT-v1-330M  — music-semantic (melody, harmony, genre)
  encodec : facebook/encodec-24khz — acoustic/timbral, lighter & faster

Install
-------
  pip install torch transformers torchaudio
  pip install matplotlib soundfile   # optional: plots + extra audio formats

Usage
-----
  # 1. From ACE-Step GenerationResult (in-memory, fastest)
  from diversity_metrics import evaluate_acestep_result
  report = evaluate_acestep_result(result, backend="mert")
  print(report.summary())
  report.plot("diversity.png")

  # 2. From audio files
  from diversity_metrics import DiversityEvaluator
  ev = DiversityEvaluator(backend="mert")
  report = ev.evaluate(["out_1.wav", "out_2.wav", "out_3.wav"])

  # 3. CLI
  python diversity_metrics.py out_1.wav out_2.wav out_3.wav --backend mert --plot report.png
"""

from __future__ import annotations

import itertools
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Literal, Optional

import numpy as np
import torch
import torch.nn.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# DiversityReport
# ──────────────────────────────────────────────────────────────────────────────

@dataclass
class DiversityReport:
    """All diversity results for one batch of generated audios."""

    backend: str
    labels: List[str]

    pair_labels:      List[str]  = field(default_factory=list)
    pair_distances:   List[float] = field(default_factory=list)   # cosine dist [0, 2]
    pair_similarities: List[float] = field(default_factory=list)  # cosine sim  [-1, 1]

    mean_distance: float = 0.0
    std_distance:  float = 0.0
    min_distance:  float = 0.0
    max_distance:  float = 0.0

    # Mean distance below this threshold → model is likely collapsing
    COLLAPSE_THRESHOLD: float = 0.10

    @property
    def collapse_warning(self) -> bool:
        return self.mean_distance < self.COLLAPSE_THRESHOLD

    # ── text summary ────────────────────────────────────────────────────────

    def summary(self) -> str:
        w = 58
        lines = [
            "",
            "=" * w,
            f"  Diversity Report  [{self.backend.upper()} embeddings]",
            "=" * w,
            f"  Outputs evaluated : {len(self.labels)}",
            f"  Pairs compared    : {len(self.pair_distances)}",
            "",
            f"  Mean  cosine distance : {self.mean_distance:.4f}",
            f"  Std   cosine distance : {self.std_distance:.4f}",
            f"  Min   cosine distance : {self.min_distance:.4f}  ← most similar pair",
            f"  Max   cosine distance : {self.max_distance:.4f}  ← most different pair",
            "",
        ]

        if self.collapse_warning:
            lines += [
                f"  ⚠  COLLAPSE WARNING",
                f"     mean distance {self.mean_distance:.4f} < threshold {self.COLLAPSE_THRESHOLD:.2f}",
                f"     The model may be generating near-identical outputs.",
            ]
        else:
            lines.append("  ✓  No collapse detected — outputs are sufficiently diverse.")

        lines += ["", "  Per-pair breakdown:"]
        for lbl, dist, sim in zip(
            self.pair_labels, self.pair_distances, self.pair_similarities
        ):
            bar = "█" * int(dist * 40)
            lines.append(f"    {lbl:<32}  dist={dist:.4f}  sim={sim:+.4f}  {bar}")

        lines += ["=" * w, ""]
        return "\n".join(lines)

    # ── plot ────────────────────────────────────────────────────────────────

    def plot(self, save_path: Optional[str] = None) -> None:
        """Heatmap of pairwise similarity + distance histogram."""
        try:
            import matplotlib.pyplot as plt
            import matplotlib.gridspec as gridspec
            from matplotlib.colors import LinearSegmentedColormap
        except ImportError:
            warnings.warn("matplotlib not installed — skipping plot. pip install matplotlib")
            return

        n = len(self.labels)
        # Rebuild full N×N similarity matrix
        sim_mat = np.eye(n, dtype=np.float32)
        for (i, j), s in zip(itertools.combinations(range(n), 2), self.pair_similarities):
            sim_mat[i, j] = s
            sim_mat[j, i] = s

        short = [Path(lb).stem[:18] for lb in self.labels]

        # colour map: red (similar) → yellow → green (diverse)
        cmap = LinearSegmentedColormap.from_list(
            "div", ["#ef4444", "#fbbf24", "#22c55e"]
        )

        BG, PANEL, TEXT = "#0d0d0d", "#161616", "#e5e5e5"

        fig = plt.figure(figsize=(14, 5.5), facecolor=BG)
        gs  = gridspec.GridSpec(1, 2, width_ratios=[1.25, 1], wspace=0.38,
                                left=0.06, right=0.96, top=0.88, bottom=0.14)

        # ── left: heatmap ──────────────────────────────────────────────────
        ax1 = fig.add_subplot(gs[0])
        ax1.set_facecolor(PANEL)
        im = ax1.imshow(sim_mat, vmin=-1, vmax=1, cmap=cmap, aspect="auto")

        ax1.set_xticks(range(n));  ax1.set_yticks(range(n))
        ax1.set_xticklabels(short, rotation=40, ha="right", fontsize=8, color=TEXT)
        ax1.set_yticklabels(short, fontsize=8, color=TEXT)
        ax1.set_title("Cosine Similarity Matrix", color=TEXT, fontsize=11, pad=8)
        for sp in ax1.spines.values():
            sp.set_edgecolor("#333")
        ax1.tick_params(colors="#555")

        for i in range(n):
            for j in range(n):
                v = sim_mat[i, j]
                ax1.text(j, i, f"{v:.2f}", ha="center", va="center",
                         fontsize=7, color="black" if abs(v) > 0.5 else TEXT)

        cb = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.04)
        cb.ax.tick_params(colors=TEXT, labelsize=7)
        cb.outline.set_edgecolor("#333")

        # ── right: histogram ───────────────────────────────────────────────
        ax2 = fig.add_subplot(gs[1])
        ax2.set_facecolor(PANEL)

        dists = np.array(self.pair_distances)
        bins  = max(6, len(dists) // 2)
        ax2.hist(dists, bins=bins, color="#22c55e", edgecolor=BG, alpha=0.85, zorder=3)
        ax2.axvline(self.mean_distance,      color="#fbbf24", lw=1.8, ls="--",
                    label=f"mean = {self.mean_distance:.3f}", zorder=4)
        ax2.axvline(self.COLLAPSE_THRESHOLD, color="#ef4444", lw=1.4, ls=":",
                    label=f"collapse < {self.COLLAPSE_THRESHOLD}", zorder=4)

        ax2.set_xlabel("Cosine Distance", color=TEXT, fontsize=9)
        ax2.set_ylabel("Pair count",      color=TEXT, fontsize=9)
        ax2.set_title("Pairwise Distance Distribution", color=TEXT, fontsize=11, pad=8)
        ax2.tick_params(colors=TEXT)
        for sp in ax2.spines.values():
            sp.set_edgecolor("#333")
        ax2.legend(fontsize=8, facecolor="#222", edgecolor="#444",
                   labelcolor=TEXT, framealpha=0.9)
        ax2.grid(axis="y", color="#2a2a2a", zorder=0)

        status_txt   = "⚠  COLLAPSE DETECTED" if self.collapse_warning else "✓  DIVERSE"
        status_color = "#ef4444"               if self.collapse_warning else "#22c55e"
        fig.suptitle(
            f"Generation Diversity  [{self.backend.upper()}]   {status_txt}",
            color=status_color, fontsize=13, fontweight="bold",
        )

        plt.tight_layout()
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches="tight",
                        facecolor=fig.get_facecolor())
            print(f"[Diversity] Plot saved → {save_path}")
        else:
            plt.show()
        plt.close()


# ──────────────────────────────────────────────────────────────────────────────
# Embedding backends
# ──────────────────────────────────────────────────────────────────────────────

class _MERTBackend:
    """
    m-a-p/MERT-v1-330M
    25-layer music transformer trained with masked modelling.
    Takes last hidden state, mean-pools over time → 1024-dim vector.
    Best for musical semantics: melody, harmony, style, genre.
    """
    MODEL_ID  = "m-a-p/MERT-v1-330M"
    TARGET_SR = 24_000

    def __init__(self, device: str):
        from transformers import AutoModel, Wav2Vec2FeatureExtractor
        print(f"[MERT] Loading {self.MODEL_ID} …")
        self.processor = Wav2Vec2FeatureExtractor.from_pretrained(
            self.MODEL_ID, trust_remote_code=True
        )
        self.model = (
            AutoModel.from_pretrained(self.MODEL_ID, trust_remote_code=True)
            .to(device).eval()
        )
        self.device = device

    @torch.no_grad()
    def embed(self, waveform: np.ndarray, sr: int) -> torch.Tensor:
        wav = _resample(waveform, sr, self.TARGET_SR)
        inp = self.processor(wav, sampling_rate=self.TARGET_SR, return_tensors="pt")
        inp = {k: v.to(self.device) for k, v in inp.items()}
        out = self.model(**inp, output_hidden_states=True)
        # last hidden state → mean-pool time axis → (D,)
        return out.last_hidden_state.mean(dim=1).squeeze(0)


class _EnCodecBackend:
    """
    facebook/encodec_24khz
    Continuous encoder features before the RVQ codebooks.
    Faster and lighter than MERT; captures acoustic / timbral diversity.
    """
    MODEL_ID = "facebook/encodec_24khz"
    TARGET_SR = 24_000

    def __init__(self, device: str):
        from transformers import EncodecModel, AutoProcessor

        print(f"[EnCodec] Loading {self.MODEL_ID} …")
        self.processor = AutoProcessor.from_pretrained(self.MODEL_ID)
        self.model = EncodecModel.from_pretrained(self.MODEL_ID).to(device).eval()
        self.device = device

    @torch.no_grad()
    def embed(self, waveform: np.ndarray, sr: int) -> torch.Tensor:
        wav = _resample(waveform, sr, self.TARGET_SR)

        inp = self.processor(
            raw_audio=wav,
            sampling_rate=self.TARGET_SR,
            return_tensors="pt",
        )

        inp = {k: v.to(self.device) for k, v in inp.items()}

        enc = self.model.encoder(inp["input_values"])

        if hasattr(enc, "last_hidden_state"):
            hidden = enc.last_hidden_state
        else:
            hidden = enc[0]

        return hidden.mean(dim=1).squeeze(0)


# ──────────────────────────────────────────────────────────────────────────────
# Audio helpers
# ──────────────────────────────────────────────────────────────────────────────

def _load_audio(path: str) -> tuple[np.ndarray, int]:
    """Load any audio file → mono float32 numpy + sample rate."""
    try:
        import torchaudio
        wav, sr = torchaudio.load(path)
        if wav.shape[0] > 1:
            wav = wav.mean(dim=0, keepdim=True)
        return wav.squeeze(0).numpy(), sr
    except Exception:
        pass
    try:
        import soundfile as sf
        wav, sr = sf.read(path, always_2d=False, dtype="float32")
        if wav.ndim == 2:
            wav = wav.mean(axis=1)
        return wav, sr
    except Exception:
        pass
    raise RuntimeError(
        f"Cannot load: {path}\nInstall torchaudio or soundfile."
    )


def _resample(wav: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    if orig_sr == target_sr:
        return wav
    try:
        import torchaudio.functional as AF
        t = torch.from_numpy(wav).unsqueeze(0)
        return AF.resample(t, orig_sr, target_sr).squeeze(0).numpy()
    except Exception:
        pass
    try:
        import librosa
        return librosa.resample(wav, orig_sr=orig_sr, target_sr=target_sr)
    except Exception:
        pass
    raise RuntimeError("Resampling failed. Install torchaudio or librosa.")


# ──────────────────────────────────────────────────────────────────────────────
# DiversityEvaluator
# ──────────────────────────────────────────────────────────────────────────────

class DiversityEvaluator:
    """
    Computes pairwise cosine-distance diversity across generated audio outputs.

    Parameters
    ----------
    backend : "mert" | "encodec"
        "mert"    — musical semantics, ~1.3 GB VRAM, recommended
        "encodec" — acoustic/timbral,  ~0.1 GB VRAM, faster
    device  : "cuda" | "cpu" | "mps"
    """

    def __init__(
        self,
        backend: Literal["mert", "encodec"] = "mert",
        device: str = "cuda",
    ):
        self.backend_name = backend.lower()
        self.device = device

        if self.backend_name == "mert":
            self._backend = _MERTBackend(device)
        elif self.backend_name == "encodec":
            self._backend = _EnCodecBackend(device)
        else:
            raise ValueError(f"Unknown backend '{backend}'. Use 'mert' or 'encodec'.")

    # ── embedding helpers ───────────────────────────────────────────────────

    def embed_file(self, path: str) -> torch.Tensor:
        """Embed a single audio file. Returns L2-normalised (D,) tensor."""
        wav, sr = _load_audio(path)
        return F.normalize(self._backend.embed(wav, sr).float(), dim=0)

    def embed_waveform(self, wav: np.ndarray, sr: int) -> torch.Tensor:
        """Embed a raw mono float32 waveform. Returns L2-normalised (D,) tensor."""
        return F.normalize(self._backend.embed(wav, sr).float(), dim=0)

    # ── main evaluation entry points ────────────────────────────────────────

    def evaluate(self, audio_paths: List[str]) -> DiversityReport:
        """
        Compute pairwise diversity for a list of audio file paths.

        Parameters
        ----------
        audio_paths : list of str / Path

        Returns
        -------
        DiversityReport
        """
        paths = [str(p) for p in audio_paths]
        if len(paths) < 2:
            raise ValueError("Need at least 2 files.")

        print(f"[Diversity] Embedding {len(paths)} files [{self.backend_name}] …")
        embs = []
        for i, p in enumerate(paths):
            print(f"  [{i+1}/{len(paths)}] {Path(p).name}")
            embs.append(self.embed_file(p))

        return self._build_report(embs, [Path(p).stem for p in paths])

    def evaluate_waveforms(
        self,
        waveforms: List[np.ndarray],
        sample_rates: List[int],
        labels: Optional[List[str]] = None,
    ) -> DiversityReport:
        """
        Compute pairwise diversity from raw waveforms (no disk I/O needed).

        Parameters
        ----------
        waveforms    : list of mono float32 numpy arrays
        sample_rates : list of int (one per waveform)
        labels       : optional list of string labels
        """
        n = len(waveforms)
        if n < 2:
            raise ValueError("Need at least 2 waveforms.")
        if labels is None:
            labels = [f"gen_{i}" for i in range(n)]

        print(f"[Diversity] Embedding {n} waveforms [{self.backend_name}] …")
        embs = []
        for i, (wav, sr) in enumerate(zip(waveforms, sample_rates)):
            print(f"  [{i+1}/{n}] {labels[i]}")
            embs.append(self.embed_waveform(wav, sr))

        return self._build_report(embs, labels)

    # ── internal ────────────────────────────────────────────────────────────

    def _build_report(
        self, embeddings: List[torch.Tensor], labels: List[str]
    ) -> DiversityReport:
        report = DiversityReport(backend=self.backend_name, labels=labels)

        for i, j in itertools.combinations(range(len(embeddings)), 2):
            sim  = float(torch.dot(embeddings[i], embeddings[j]).cpu())
            dist = 1.0 - sim
            report.pair_labels.append(f"{labels[i]}  vs  {labels[j]}")
            report.pair_distances.append(dist)
            report.pair_similarities.append(sim)

        d = np.array(report.pair_distances)
        report.mean_distance = float(d.mean())
        report.std_distance  = float(d.std())
        report.min_distance  = float(d.min())
        report.max_distance  = float(d.max())
        return report


# ──────────────────────────────────────────────────────────────────────────────
# ACE-Step convenience wrapper
# ──────────────────────────────────────────────────────────────────────────────

def evaluate_acestep_result(
    result,
    backend: str = "mert",
    device: str = "cuda",
) -> DiversityReport:
    """
    Pass an ACE-Step GenerationResult directly to get a DiversityReport.

    Example
    -------
        result = generate_music(dit_handler, llm_handler, params, config)
        report = evaluate_acestep_result(result, backend="mert")
        print(report.summary())
        report.plot("diversity.png")
    """
    if not result.success:
        raise ValueError(f"ACE-Step generation failed: {result.error}")

    ev = DiversityEvaluator(backend=backend, device=device)

    waveforms, srs, labels = [], [], []
    file_paths = []

    for audio in result.audios:
        tensor = audio.get("tensor")
        sr     = audio.get("sample_rate", 48_000)
        label  = audio.get("key", f"gen_{len(labels)}")

        if tensor is not None:
            wav = tensor.cpu().float().numpy()
            if wav.ndim == 2:
                wav = wav.mean(axis=0)  # stereo → mono
            waveforms.append(wav)
            srs.append(sr)
            labels.append(label)
        elif audio.get("path"):
            file_paths.append(audio["path"])

    # prefer in-memory tensors (no disk I/O)
    if waveforms:
        return ev.evaluate_waveforms(waveforms, srs, labels)
    return ev.evaluate(file_paths)


# ──────────────────────────────────────────────────────────────────────────────
# CLI
# ──────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    from pathlib import Path

    p = argparse.ArgumentParser(
        description="Pairwise embedding diversity for generated audio files."
    )

    p.add_argument(
        "files",
        nargs="*",
        help="Audio files to compare. If empty, uses outputs/music/*.wav",
    )

    p.add_argument(
        "--backend",
        choices=["mert", "encodec"],
        default="encodec",
        help="Embedding backend to use. Default: encodec",
    )

    p.add_argument(
        "--device",
        default="cuda",
        help="cuda / cpu / mps. Default: cuda",
    )

    p.add_argument(
        "--plot",
        default="diversity.png",
        help="Save plot to this path. Default: diversity.png",
    )

    args = p.parse_args()

    # If no files are passed, automatically use all .wav files in outputs/music
    if not args.files:
        default_audio_dir = Path("/home/cvcadmin/cruilla/Festival-Cruilla/outputs/music/metrics")
        args.files = sorted(str(path) for path in default_audio_dir.glob("*.wav"))

        if len(args.files) < 2:
            raise SystemExit(
                f"Need at least 2 .wav files, but found {len(args.files)} in {default_audio_dir}"
            )

        print(f"[Diversity] No files provided. Using {len(args.files)} files from {default_audio_dir}/")

    ev = DiversityEvaluator(
        backend=args.backend,
        device=args.device,
    )

    report = ev.evaluate(args.files)
    print(report.summary())

    if args.plot:
        report.plot(args.plot)