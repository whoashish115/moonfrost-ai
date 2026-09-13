"""Hosts the project page and a GPU-backed chat demo on Modal.

Modal bills per second and scales to zero, so an idle demo costs nothing and a container
only spins up when somebody opens the page. That is the difference from a rented box: no
standing charge for a demo nobody is using.

    modal deploy training/deploy_demo.py

Prints a permanent URL. The site is served at / and the chat at /chat.

Cold start is the thing to watch. The weights are pulled from the Hub once and kept in a
Modal Volume, so the first request after an idle period reloads from the volume rather than
from Hugging Face: roughly 20 seconds instead of a couple of minutes.
"""
import modal

MODEL_ID = "whoashish115/Moonfrost-777M-Instruct-v2"
CACHE = "/cache"

app = modal.App("moonfrost-demo")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch==2.6.0",
        "transformers>=4.44",
        "tokenizers",
        "gradio>=4.44",
        "fastapi",
        "huggingface_hub",
    )
    .env({"HF_HOME": CACHE, "HF_HUB_DISABLE_TELEMETRY": "1"})
    # the page is small enough to bake into the image rather than mount a volume for
    .add_local_dir("docs", remote_path="/site", ignore=["archive", "logs", "wandb", "*.md"])
)

weights = modal.Volume.from_name("moonfrost-weights", create_if_missing=True)


@app.function(image=image, volumes={CACHE: weights}, timeout=1800)
def fetch_weights():
    """Run once so the first real request does not pay for the download."""
    from huggingface_hub import snapshot_download

    path = snapshot_download(MODEL_ID)
    weights.commit()
    print("cached", path)


@app.function(
    image=image,
    gpu="T4",                 # 161M active parameters; anything larger is wasted money
    volumes={CACHE: weights},
    scaledown_window=300,     # stay warm five minutes after the last request
    max_containers=1,         # a demo, not a service
    timeout=600,
)
@modal.concurrent(max_inputs=8)
@modal.asgi_app()
def web():
    import threading

    import gradio as gr
    import torch
    from fastapi import FastAPI
    from fastapi.responses import FileResponse
    from fastapi.staticfiles import StaticFiles
    from gradio.routes import mount_gradio_app
    from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_ID, trust_remote_code=True, torch_dtype=torch.float32
    ).eval()
    if torch.cuda.is_available():
        model = model.to("cuda")

    # Both end an assistant turn. Without <|user|> the model writes your next message.
    stop_ids = [i for i in (tokenizer.convert_tokens_to_ids("<|endoftext|>"),
                            tokenizer.convert_tokens_to_ids("<|user|>"))
                if isinstance(i, int) and i >= 0]

    system_default = ("You are Moonfrost, a small AI assistant. Answer clearly and briefly. "
                      "If you do not know something, say so.")

    def respond(message, history, system, max_new_tokens, temperature, top_p, penalty):
        parts = [f"<|system|>{system or system_default}"]
        for turn in history or []:
            if turn["role"] == "user":
                parts.append(f"<|user|>{turn['content']}")
            else:
                parts.append(f"<|assistant|>{turn['content']}<|endoftext|>")
        parts.append(f"<|user|>{message}<|assistant|>")

        inputs = tokenizer("".join(parts), return_tensors="pt").to(model.device)
        streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)
        threading.Thread(target=model.generate, kwargs=dict(
            **inputs, streamer=streamer,
            max_new_tokens=int(max_new_tokens),
            do_sample=temperature > 0,
            temperature=max(temperature, 1e-5),
            top_p=top_p, repetition_penalty=penalty,
            eos_token_id=stop_ids or None,
            pad_token_id=tokenizer.convert_tokens_to_ids("<|pad|>"),
        )).start()

        reply = ""
        for piece in streamer:
            reply += piece
            yield reply

    with gr.Blocks(title="Moonfrost", theme=gr.themes.Soft(primary_hue="pink")) as demo:
        gr.Markdown(
            "### Moonfrost 777M\n"
            "777M parameters, **161M active per token**, trained from scratch on 6B tokens "
            "for about $55. It invents facts confidently and knows very little. "
            "[Weights](https://huggingface.co/whoashish115/Moonfrost-777M-Instruct-v2)"
        )
        with gr.Accordion("Settings", open=False):
            system = gr.Textbox(value=system_default, label="System prompt", lines=2)
            tokens = gr.Slider(32, 400, value=180, step=8, label="Max new tokens")
            temperature = gr.Slider(0.0, 1.5, value=0.7, step=0.05, label="Temperature")
            top_p = gr.Slider(0.1, 1.0, value=0.92, step=0.01, label="Top-p")
            penalty = gr.Slider(1.0, 1.5, value=1.15, step=0.01, label="Repetition penalty")

        gr.ChatInterface(
            fn=respond, type="messages",
            additional_inputs=[system, tokens, temperature, top_p, penalty],
            examples=[["hi"], ["Who made you?"], ["Write a hello world program in Java"],
                      ["Why is the sky blue?"], ["Write a haiku about winter"]],
            cache_examples=False,
        )

    api = FastAPI()
    api.mount("/icons", StaticFiles(directory="/site/icons"), name="icons")

    @api.get("/")
    def index():
        return FileResponse("/site/index.html")

    @api.get("/logo.png")
    def logo():
        return FileResponse("/site/logo.png")

    return mount_gradio_app(app=api, blocks=demo, path="/chat")
