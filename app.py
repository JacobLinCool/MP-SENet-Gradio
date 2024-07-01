import torch
import librosa
from librosa.display import specshow
import numpy as np
from dataset import mag_pha_stft, mag_pha_istft
from env import AttrDict
from models.generator import MPNet
from accelerate import Accelerator
import matplotlib.pyplot as plt
from zero import zero, zero_is_available
import gradio as gr
from typing import Tuple

accelerator = Accelerator()
device = accelerator.device
# the required functions have not been implemented on mps devices
if device.type == "mps":
    device = torch.device("cpu")
print(f"Using device: {device}")

h = AttrDict(
    {
        "dense_channel": 64,
        "compress_factor": 0.3,
        "beta": 2.0,
        "sampling_rate": 16000,
        "segment_size": 32000,
        "n_fft": 400,
        "hop_size": 100,
        "win_size": 400,
    }
)
model = MPNet(h).to(device)
state = torch.load("g_best.pt", map_location="cpu" if zero_is_available else device)
model.load_state_dict(state["generator"])
model.eval()

# this model consumes a lot of memory
# ZeroGPU has 40GB of memory, so it can run for a longer segment
segment_duration = 10 if zero_is_available else 3  # seconds


def plot_spec(y: np.ndarray, title: str = "Spectrogram") -> plt.Figure:
    y = np.nan_to_num(y)
    stft = librosa.stft(y, n_fft=h.n_fft, hop_length=h.hop_size, win_length=h.win_size)
    D = librosa.amplitude_to_db(np.abs(stft), ref=np.max)

    fig = plt.figure(figsize=(10, 4))
    specshow(D, sr=h.sampling_rate, y_axis="linear", x_axis="time", cmap="viridis")
    plt.title(title)
    plt.tight_layout()

    return fig


def process_segment(segment: np.ndarray) -> np.ndarray:
    print(f"Processing segment", segment.shape)
    segment = torch.FloatTensor(segment).to(device)
    norm_factor = torch.sqrt(len(segment) / torch.sum(segment**2.0)).to(device)
    segment = (segment * norm_factor).unsqueeze(0)
    noisy_amp, noisy_pha, noisy_com = mag_pha_stft(
        segment, h.n_fft, h.hop_size, h.win_size, h.compress_factor
    )
    amp_g, pha_g, com_g = model(noisy_amp, noisy_pha)
    audio_g = mag_pha_istft(
        amp_g, pha_g, h.n_fft, h.hop_size, h.win_size, h.compress_factor
    )
    audio_g = audio_g / norm_factor
    audio_g = audio_g.squeeze().detach().cpu().numpy()
    print(f"Processed segmen", audio_g.shape)
    return audio_g


@zero
def run(input: str) -> Tuple[Tuple[int, np.ndarray], np.ndarray, np.ndarray]:
    noisy_wav, _ = librosa.load(input, sr=h.sampling_rate)
    segment_samples = segment_duration * h.sampling_rate
    segments = [
        noisy_wav[i : i + segment_samples]
        for i in range(0, len(noisy_wav), segment_samples)
    ]
    processed_segments = [process_segment(segment) for segment in segments]
    processed_wav = np.concatenate(processed_segments)

    noisy_spec = plot_spec(noisy_wav, title="Original Spectrogram")
    out_spec = plot_spec(processed_wav, title="Processed Spectrogram")

    return ((h.sampling_rate, processed_wav), noisy_spec, out_spec)


with gr.Blocks() as app:
    gr.Markdown(
        "# MP-SENet Speech Enhancement\n\n[MP-SENet](https://github.com/yxlu-0102/MP-SENet) with ZeroGPU support."
    )

    with gr.Row():
        with gr.Column():
            input = gr.Audio(label="Upload an audio file", type="filepath")
            original_spec = gr.Plot(label="Original Spectrogram")
            btn = gr.Button(value="Process", variant="primary")

        with gr.Column():
            output = gr.Audio(label="Processed Audio")
            processed_spec = gr.Plot(label="Processed Spectrogram")

    btn.click(
        fn=run,
        inputs=[input],
        outputs=[output, original_spec, processed_spec],
        api_name="run",
    )

    gr.Examples(
        examples=[
            ["examples/p226_007.wav"],
            ["examples/p226_016.wav"],
            ["examples/p230_005.wav"],
            ["examples/p232_032.wav"],
            ["examples/p232_232.wav"],
        ],
        inputs=input,
    )

    app.launch()
