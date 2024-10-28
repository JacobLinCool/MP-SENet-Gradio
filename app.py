import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

import random
import time
import torch
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
# the required functions have not been implemented on mps devices
if device.type == "mps":
    device = torch.device("cpu")
print(f"Using device: {device}")

model = MPSENet.from_pretrained("JacobLinCool/MP-SENet-DNS").to(device)

tasks = {}


def gen_task_id():
    return str(int(time.time())) + str(random.randint(1000, 9999))


def plot_spec(y: np.ndarray, title: str = "Spectrogram") -> plt.Figure:
    y[np.isnan(y)] = 0
    y[np.isinf(y)] = 0
    stft = librosa.stft(
        y, n_fft=model.h.n_fft, hop_length=model.h.hop_size, win_length=model.h.win_size
    )
    D = librosa.amplitude_to_db(np.abs(stft), ref=np.max)

    fig = plt.figure(figsize=(10, 4))
    specshow(D, sr=model.sampling_rate, y_axis="linear", x_axis="time", cmap="viridis")
    plt.title(title)
    plt.tight_layout()

    return fig


def preprocess(input: str, plot: bool) -> Tuple[str, str]:
    start_time = time.time()
    noisy_wav, sr = librosa.load(input, sr=model.sampling_rate)
    print(f"{noisy_wav.shape=}, {sr=}")
    print(f"Loaded audio in {time.time() - start_time:.2f} seconds")

    task_id = gen_task_id()
    tasks[task_id] = {
        "noisy_wav": noisy_wav,
        "plot": plot,
    }
    print(f"Task {task_id} created.")

    return (task_id, "Processing ...")


def run_task(
    task_id: str,
) -> Tuple[Tuple[int, np.ndarray], np.ndarray, np.ndarray, str]:
    task = tasks[task_id]
    if not task:
        raise gr.Error("Task not found")
    if not "noisy_wav" in task:
        del tasks[task_id]
        raise gr.Error("No audio found")

    noisy_wav = task["noisy_wav"]
    plot = task["plot"] if "plot" in task else True
    del tasks[task_id]
    print(f"Processing task {task_id}")

    start_time = time.time()
    processed_wav, sr, notation = model(noisy_wav)
    print(f"{processed_wav.shape=}, {sr=}, {notation=}")
    print(f"Inference in {time.time() - start_time:.2f} seconds")

    if plot:
        start_time = time.time()
        noisy_spec = plot_spec(noisy_wav, title="Original Spectrogram")
        out_spec = plot_spec(processed_wav, title="Processed Spectrogram")
        print(f"Plotted spectrograms in {time.time() - start_time:.2f} seconds")
    else:
        print("Skipping plotting")
        noisy_spec = out_spec = None

    return ((sr, processed_wav), noisy_spec, out_spec, "Processed.")


@spaces.GPU()
def run(task_id: str):
    return run_task(task_id)


@spaces.GPU(duration=60 * 2)
def run2x(task_id: str):
    return run_task(task_id)


@spaces.GPU(duration=60 * 4)
def run4x(task_id: str):
    return run_task(task_id)


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
            plot = gr.Checkbox(label="Plot Spectrograms", value=True)

        with gr.Column():
            original_spec = gr.Plot(label="Original Spectrogram")

    with gr.Row():
        btn = gr.Button(value="Process", variant="primary")
        task_id = gr.Textbox(label="Task ID", visible=False)
    with gr.Row():
        info = gr.Markdown("Press the button to process the audio.")

    with gr.Row():
        with gr.Column():
            output = gr.Audio(label="Processed Audio")
        with gr.Column():
            processed_spec = gr.Plot(label="Processed Spectrogram")

    btn.click(
        fn=preprocess,
        inputs=[input, plot],
        outputs=[task_id, info],
        api_name="preprocess",
    ).success(
        fn=run,
        inputs=[task_id],
        outputs=[output, original_spec, processed_spec, info],
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

    btn2x = gr.Button(value="Process", variant="primary", visible=False)
    btn2x.click(
        fn=preprocess,
        inputs=[input, plot],
        outputs=[task_id, info],
        api_name="preprocess",
    ).success(
        fn=run2x,
        inputs=[task_id],
        outputs=[output, original_spec, processed_spec, info],
        api_name="run2x",
    )

    btn4x = gr.Button(value="Process", variant="primary", visible=False)
    btn4x.click(
        fn=preprocess,
        inputs=[input, plot],
        outputs=[task_id, info],
        api_name="preprocess",
    ).success(
        fn=run4x,
        inputs=[task_id],
        outputs=[output, original_spec, processed_spec, info],
        api_name="run4x",
    )

    app.launch()
