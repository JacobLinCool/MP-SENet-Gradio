import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import time
import librosa
import spaces
from librosa.display import specshow
import numpy as np
from accelerate import Accelerator
import matplotlib.pyplot as plt
import gradio as gr
from typing import Tuple
from MPSENet import MPSENet

accelerator = Accelerator()
device = accelerator.device
print(f"Using device: {device}")

model = MPSENet.from_pretrained("JacobLinCool/MP-SENet-DNS").to(device)


def plot_spec(y: np.ndarray, title: str = "Spectrogram") -> plt.Figure:
    y[np.isnan(y)] = 0
    y[np.isinf(y)] = 0
    stft = librosa.stft(
        y, n_fft=model.h.n_fft, hop_length=model.h.hop_size, win_length=model.h.win_size
    )
    D = librosa.amplitude_to_db(np.abs(stft), ref=np.max)

    fig = plt.figure(figsize=(10, 4))
    specshow(
        D,
        sr=model.sampling_rate,
        n_fft=model.h.n_fft,
        hop_length=model.h.hop_size,
        win_length=model.h.win_size,
        y_axis="linear",
        x_axis="time",
        cmap="viridis",
    )
    plt.title(title)
    plt.tight_layout()

    return fig


def plot_input(input: str) -> plt.Figure:
    wav, _ = librosa.load(input, sr=model.sampling_rate)
    return plot_spec(wav, title="Original Spectrogram")


def plot_output(output: Tuple[int, np.ndarray]) -> plt.Figure:
    wav = output[1].astype(np.float32) / 32768.0
    return plot_spec(wav, title="Processed Spectrogram")


def process_audio(
    input: str,
    segment_size_seconds: int,
) -> Tuple[Tuple[int, np.ndarray], np.ndarray, np.ndarray, str]:
    # Load the audio
    start_time = time.time()
    noisy_wav, sr = librosa.load(input, sr=model.sampling_rate)
    print(f"{noisy_wav.shape=}, {sr=}")
    print(f"Loaded audio in {time.time() - start_time:.2f} seconds")

    # Process the audio
    start_time = time.time()
    processed_wav, sr, notation = model(
        noisy_wav, segment_size=segment_size_seconds * 16000
    )
    print(f"{processed_wav.shape=}, {sr=}, {notation=}")
    print(f"Inference in {time.time() - start_time:.2f} seconds")

    return ((sr, processed_wav), "Processed.")


@spaces.GPU()
def run(input: str, segment_size_seconds: int):
    return process_audio(input, segment_size_seconds)


@spaces.GPU(duration=60 * 2)
def run2x(input: str, segment_size_seconds: int):
    return process_audio(input, segment_size_seconds)


@spaces.GPU(duration=60 * 4)
def run4x(input: str, segment_size_seconds: int):
    return process_audio(input, segment_size_seconds)


with gr.Blocks() as app:
    gr.Markdown(
        "# MP-SENet Speech Enhancement\n\n[MP-SENet](https://github.com/yxlu-0102/MP-SENet) with ZeroGPU support.\n"
        "> Package is available at [JacobLinCool/MPSENet](https://github.com/JacobLinCool/MPSENet)"
    )

    with gr.Row():
        with gr.Column():
            input = gr.Audio(
                label="Upload an audio file", type="filepath", show_download_button=True
            )
        with gr.Column():
            original_spec = gr.Plot(label="Original Spectrogram")

    with gr.Row():
        btn = gr.Button(value="Process", variant="primary")
    with gr.Row():
        info = gr.Markdown("Press the button to process the audio.")

    with gr.Row():
        with gr.Column():
            output = gr.Audio(
                label="Processed Audio", show_download_button=True
            )
        with gr.Column():
            processed_spec = gr.Plot(label="Processed Spectrogram")

    with gr.Accordion("Advanced Settings", open=False):
        segment_size = gr.Slider(
            minimum=1,
            maximum=20,
            value=10,
            step=1,
            label="Segment Size (seconds)",
            info="The audio will be processed in segments of this size. Larger segments take more memory but may give more consistent results.",
        )

    input.change(
        fn=plot_input,
        inputs=[input],
        outputs=[original_spec],
    )
    output.change(
        fn=plot_output,
        inputs=[output],
        outputs=[processed_spec],
    )

    btn.click(
        fn=run,
        inputs=[input, segment_size],
        outputs=[output, info],
        api_name="run",
    )

    gr.Examples(
        examples=[
            ["examples/p226_007.wav", 2],
            ["examples/p226_016.wav", 2],
            ["examples/p230_005.wav", 8],
            ["examples/p232_032.wav", 2],
            ["examples/p232_232.wav", 2],
        ],
        inputs=[input, segment_size],
    )

    btn2x = gr.Button(value="Process", variant="primary", visible=False)
    btn2x.click(
        fn=run2x,
        inputs=[input, segment_size],
        outputs=[output, info],
        api_name="run2x",
    )

    btn4x = gr.Button(value="Process", variant="primary", visible=False)
    btn4x.click(
        fn=run4x,
        inputs=[input, segment_size],
        outputs=[output, info],
        api_name="run4x",
    )

    app.launch()
