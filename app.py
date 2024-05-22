from modal import App, Image, Secret, CloudBucketMount

GPU = "A100"
GPU_COUNT = 1

app = App()
image = Image.from_registry("pytorch/pytorch").apt_install("git").pip_install("git+https://github.com/huggingface/peft.git",
                                                                              "accelerate", "datasets", "loralib", "wandb", "python_dotenv").pip_install("bitsandbytes", gpu=GPU)

secret = Secret.from_dict({
    "AWS_ACCESS_KEY_ID": "jxmow5wzzlc53p5oaktfzvo6cvca",
    "AWS_SECRET_ACCESS_KEY": "jyp7xnbhzhy2evknwlb44voatmc26dpftz6yz5yfquvmuev4azfke",
})


@app.function(
    image=image,
    gpu=f"{GPU}:{GPU_COUNT}",
    volumes={
        "/bucket": CloudBucketMount(
            bucket_name="fine-tuning",
            secret=secret,
            bucket_endpoint_url="https://gateway.storjshare.io",
        )
    }
)
def run_train():
    from FineTuningOpenELM import train

    train.train()


@app.local_entrypoint()
def main():
    run_train.remote()
