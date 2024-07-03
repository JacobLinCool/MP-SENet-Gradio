# MP-SENet Gradio App

A Gradio app for [MP-SENet](https://github.com/yxlu-0102/MP-SENet) with ZeroGPU support.

Most of the code and the model weights are from the original repository (MIT licensed), with some modifications to make it work with Gradio and ZeroGPU and handle longer audio files.

## API Usage

You can also use the model through the Gradio API. Here's an example:

```python
from gradio_client import Client, handle_file

client = Client("JacobLinCool/MP-SENet")

task_id, _ = client.predict(
    input=handle_file("path/to/audio.wav"),
    plot=False,
    api_name="/preprocess",
)
output, _, _, _ = client.predict(task_id=task_id, api_name="/run")
print(output) # The path to the output file
```

The default `/run` endpoint will try to acquire GPU for 60 seconds. It should be sufficient for audio files up to 20 minutes.
If you are working with audio files longer than 20 minutes, you can use the `/run2x` or `/run4x` endpoints, which will try to acquire GPU for 120 and 240 seconds respectively.
