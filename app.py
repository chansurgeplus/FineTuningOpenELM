from modal import App, Image, Volume
import pathlib

GPU = "A10G"
GPU_COUNT = 1

app = App()
image = Image.from_registry("pytorch/pytorch").apt_install("git").pip_install("git+https://github.com/huggingface/peft.git",
                                                                              "accelerate", "datasets", "loralib", "wandb", "python_dotenv").pip_install("bitsandbytes", gpu=GPU)

vol_openelm = Volume.from_name("openelm", create_if_missing=True)

@app.function(
    image=image,
    gpu=f"{GPU}:{GPU_COUNT}",
    volumes={
        "/root/openelm": vol_openelm
    },
    timeout=86400,
)
def run_train():
    from FineTuningOpenELM import train

    train.train()


@app.local_entrypoint()
def main():
    run_train.remote()
