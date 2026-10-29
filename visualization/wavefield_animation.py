from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.font_manager import findfont
import numpy as np
from PIL import Image as PillowImage
from PIL import ImageDraw, ImageFont


class WavefieldFrameRecorder:
    """Capture o envelope das ondas de todos os tiros durante a propagação."""

    def __init__(self, frame_interval_s=0.01):
        self.frame_interval_s = float(frame_interval_s)
        if self.frame_interval_s <= 0.0:
            raise ValueError("frame_interval_s deve ser maior que zero")
        self.frames = []
        self.times_s = []
        self.next_frame_time_s = 0.0

    def __call__(self, wavefield, time_s):
        time_s = float(time_s)
        if time_s + 1e-12 < self.next_frame_time_s:
            return
        values = wavefield.detach().abs().cpu().numpy()
        if values.ndim == 4:
            values = values[0]
        if values.ndim == 3:
            values = np.max(values, axis=0)
        if values.ndim != 2:
            raise ValueError(
                "wavefield deve conter [fonte, profundidade, largura] "
                "ou [profundidade, largura]"
            )
        self.frames.append(np.asarray(values, dtype=np.float32))
        self.times_s.append(time_s)
        self.next_frame_time_s = (
            np.floor(time_s / self.frame_interval_s + 1.0) * self.frame_interval_s
        )


def save_wavefield_gif(
    recorder,
    output_path,
    *,
    grid_spacing_m=1.0,
    duration_ms=80,
):
    """Renderiza a propagação sísmica em tons de cinza em um GIF."""
    if not recorder.frames:
        raise ValueError("A propagação não capturou nenhum frame de wavefield")

    frame_shape = np.asarray(recorder.frames[0]).shape
    if len(frame_shape) != 2:
        raise ValueError("Cada frame deve ter a forma [profundidade, largura]")

    amplitude_scale = max(float(np.max(frame)) for frame in recorder.frames)
    if not np.isfinite(amplitude_scale) or amplitude_scale <= 0.0:
        amplitude_scale = 1.0

    height, width = frame_shape
    extent = (
        0.0,
        width * float(grid_spacing_m),
        height * float(grid_spacing_m),
        0.0,
    )
    rendered_frames = []
    for frame, time_s in zip(recorder.frames, recorder.times_s):
        normalized = np.clip(frame / amplitude_scale, 0.0, 1.0)
        figure, axis = plt.subplots(figsize=(5.6, 3.2), dpi=72)
        image = axis.imshow(
            normalized,
            cmap="gray",
            origin="upper",
            extent=extent,
            aspect="auto",
            vmin=0.0,
            vmax=1.0,
        )
        figure.colorbar(image, ax=axis, fraction=0.046, label="Amplitude normalizada")
        axis.set_xlabel("Posição x (m)")
        axis.set_ylabel("Profundidade z (m)")
        axis.set_title(f"Propagação sísmica — {time_s:.3f} s")
        figure.tight_layout()
        figure.canvas.draw()
        rendered_frames.append(
            PillowImage.fromarray(np.asarray(figure.canvas.buffer_rgba())[..., :3].copy())
        )
        plt.close(figure)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rendered_frames[0].save(
        output_path,
        save_all=True,
        append_images=rendered_frames[1:],
        duration=int(duration_ms),
        loop=0,
        optimize=True,
    )
    return output_path


def combine_wavefield_gifs(
    left_path,
    right_path,
    output_path,
    *,
    left_label="Modelo verdadeiro",
    right_label="Predição U-Net",
):
    """Combina duas animações lado a lado, mantendo o último frame até o fim."""
    left_gif = PillowImage.open(left_path)
    right_gif = PillowImage.open(right_path)
    frame_count = max(left_gif.n_frames, right_gif.n_frames)
    panel_width = max(left_gif.width, right_gif.width)
    panel_height = max(left_gif.height, right_gif.height)
    header_height = 34
    font = ImageFont.truetype(findfont("DejaVu Sans"), size=12)
    combined_frames = []
    frame_durations = []

    for frame_index in range(frame_count):
        left_gif.seek(min(frame_index, left_gif.n_frames - 1))
        right_gif.seek(min(frame_index, right_gif.n_frames - 1))
        left_frame = left_gif.convert("RGB").resize((panel_width, panel_height))
        right_frame = right_gif.convert("RGB").resize((panel_width, panel_height))
        comparison = PillowImage.new(
            "RGB",
            (panel_width * 2, panel_height + header_height),
            "white",
        )
        draw = ImageDraw.Draw(comparison)
        draw.text((12, 10), left_label, fill="black", font=font)
        draw.text((panel_width + 12, 10), right_label, fill="black", font=font)
        comparison.paste(left_frame, (0, header_height))
        comparison.paste(right_frame, (panel_width, header_height))
        combined_frames.append(comparison)
        frame_durations.append(
            max(
                int(left_gif.info.get("duration", 80)),
                int(right_gif.info.get("duration", 80)),
            )
        )

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined_frames[0].save(
        output_path,
        save_all=True,
        append_images=combined_frames[1:],
        duration=frame_durations,
        loop=0,
        optimize=True,
    )
    left_gif.close()
    right_gif.close()
    return output_path
