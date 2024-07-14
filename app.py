import random
import time
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


tasks = {}


def gen_task_id():
    return str(int(time.time())) + str(random.randint(1000, 9999))


def plot_spec(y: np.ndarray, title: str = "Spectrogram") -> plt.Figure:
    y[np.isnan(y)] = 0
    y[np.isinf(y)] = 0
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


def preprocess(input: str, plot: bool) -> Tuple[str, str]:
    start_time = time.time()
    noisy_wav, _ = librosa.load(input, sr=h.sampling_rate)
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
    segment_samples = segment_duration * h.sampling_rate
    segments = [
        noisy_wav[i : i + segment_samples]
        for i in range(0, len(noisy_wav), segment_samples)
    ]
    print(f"Segmented audio in {time.time() - start_time:.2f} seconds")

    start_time = time.time()
    processed_segments = [process_segment(segment) for segment in segments]
    print(f"Inference in {time.time() - start_time:.2f} seconds")

    processed_wav = np.concatenate(processed_segments)

    if plot:
        start_time = time.time()
        noisy_spec = plot_spec(noisy_wav, title="Original Spectrogram")
        out_spec = plot_spec(processed_wav, title="Processed Spectrogram")
        print(f"Plotted spectrograms in {time.time() - start_time:.2f} seconds")
    else:
        print("Skipping plotting")
        noisy_spec = out_spec = None

    return ((h.sampling_rate, processed_wav), noisy_spec, out_spec, "Processed.")


@zero()
def run(task_id: str):
    return run_task(task_id)


@zero(duration=60 * 2)
def run2x(task_id: str):
    return run_task(task_id)


@zero(duration=60 * 4)
def run4x(task_id: str):
    return run_task(task_id)


with gr.Blocks() as app:
    gr.Markdown(
        "# MP-SENet Speech Enhancement\n\n[MP-SENet](https://github.com/yxlu-0102/MP-SENet) with ZeroGPU support."
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
            output = gr.Audio(label="Processed Audio", format="mp3")
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
