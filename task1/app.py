"""A minimal interactive FLUX Realism LoRA demo powered by Gradio."""
from __future__ import annotations

import json
import os
import time
from datetime import datetime
from pathlib import Path

import gradio as gr
from dotenv import load_dotenv
from huggingface_hub import InferenceClient

MODEL_ID = "XLabs-AI/flux-RealismLora"
PROVIDER = "fal-ai"
PROJECT_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = PROJECT_DIR / "outputs"
CSS = """

footer a,
footer .divider,
footer .record,
footer > *:not(.settings) { display: none !important; }
"""
DEFAULT_PROMPT = (
    "A documentary photograph of a quiet university campus after rain, "
    "students walking naturally, wet pavement reflecting soft morning light, "
    "realistic materials and proportions, subtle colors, eye-level view, "
    "35mm lens, natural depth of field"
)


def _write_record(record: dict, stem: str) -> Path:
    """Save a token-free JSON record for homework evidence."""
    record_path = OUTPUT_DIR / f"{stem}.json"
    record_path.write_text(
        json.dumps(record, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return record_path


def generate_image(
    prompt: str,
    negative_prompt: str,
    width: int,
    height: int,
    steps: int,
    guidance_scale: float,
    seed: float,
):
    """Call Hugging Face Inference Providers and return a generated image."""
    prompt = prompt.strip()
    negative_prompt = negative_prompt.strip()
    if not prompt:
        raise gr.Error("请输入提示词后再生成。")

    token = os.getenv("HF_TOKEN")
    if not token:
        raise gr.Error("没有找到 HF_TOKEN。请按照 README 创建 .env 文件后重启项目。")

    width = int(width)
    height = int(height)
    steps = int(steps)
    seed = int(seed)
    guidance_scale = float(guidance_scale)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().astimezone()
    stem = timestamp.strftime("flux_%Y%m%d_%H%M%S_%f")
    image_path = OUTPUT_DIR / f"{stem}.png"

    parameters = {
        "width": width,
        "height": height,
        "num_inference_steps": steps,
        "guidance_scale": guidance_scale,
        "seed": seed,
    }
    if negative_prompt:
        parameters["negative_prompt"] = negative_prompt

    record = {
        "status": "started",
        "requested_at": timestamp.isoformat(timespec="seconds"),
        "model": MODEL_ID,
        "provider": PROVIDER,
        "prompt": prompt,
        "parameters": parameters,
    }

    started = time.perf_counter()
    try:
        client = InferenceClient(provider=PROVIDER, api_key=token)
        image = client.text_to_image(
            prompt=prompt,
            model=MODEL_ID,
            **parameters,
        )
        elapsed = time.perf_counter() - started
        image.save(image_path)
        record.update(
            {
                "status": "success",
                "elapsed_seconds": round(elapsed, 2),
                "output": str(image_path.relative_to(PROJECT_DIR)),
            }
        )
        record_path = _write_record(record, stem)
        status = (
            f"**生成成功**（{elapsed:.2f} 秒）  \n"
            f"模型 `{MODEL_ID}` · `{width}×{height}` · seed `{seed}`  \n"
            f"记录：`{record_path.name}`"
        )
        print(
            f"[SUCCESS] model={MODEL_ID} provider={PROVIDER} "
            f"size={width}x{height} seed={seed} elapsed={elapsed:.2f}s "
            f"output={image_path.name}",
            flush=True,
        )
        return image, status
    except Exception as error:
        elapsed = time.perf_counter() - started
        record.update(
            {
                "status": "error",
                "elapsed_seconds": round(elapsed, 2),
                "error_type": type(error).__name__,
                "error_message": str(error),
            }
        )
        _write_record(record, stem)
        print(
            f"[ERROR] model={MODEL_ID} provider={PROVIDER} "
            f"type={type(error).__name__} message={error}",
            flush=True,
        )
        raise gr.Error(f"生成失败：{type(error).__name__}: {error}") from error


def build_demo() -> gr.Blocks:
    """Build the single-page Gradio interface."""
    with gr.Blocks(
        title="huggingface写实图像生成",
        theme=gr.themes.Soft(primary_hue="sky", neutral_hue="slate"),
        css=CSS,
    ) as demo:
        gr.Markdown(
            "# huggingface写实图像生成\n\n"
            "输入提示词后点击生成，只有点击才会产生云端推理费用。"
        )

        prompt = gr.Textbox(
            value=DEFAULT_PROMPT,
            label="提示词",
            lines=6,
            placeholder="描述希望生成的真实世界场景……",
        )

        generate_button = gr.Button("生成图片", variant="primary")

        output_image = gr.Image(label="生成结果", type="pil")
        status = gr.Markdown("填写提示词后点击生成。")

        with gr.Accordion("高级参数", open=False):
            negative_prompt = gr.Textbox(
                label="负向提示词（可选）",
                lines=2,
                placeholder="例如：illustration, CGI, distorted hands, blurry",
            )
            with gr.Row():
                width = gr.Dropdown(
                    choices=[512, 768, 1024], value=768, label="宽度"
                )
                height = gr.Dropdown(
                    choices=[512, 768, 1024], value=1024, label="高度"
                )
            with gr.Row():
                steps = gr.Slider(
                    minimum=1,
                    maximum=50,
                    value=28,
                    step=1,
                    label="推理步数",
                )
                guidance_scale = gr.Slider(
                    minimum=1.0,
                    maximum=10.0,
                    value=3.5,
                    step=0.1,
                    label="提示词引导强度",
                )
            seed = gr.Number(
                value=12345,
                precision=0,
                label="Seed（固定后便于比较提示词）",
            )

        inputs = [
            prompt,
            negative_prompt,
            width,
            height,
            steps,
            guidance_scale,
            seed,
        ]
        outputs = [output_image, status]

        generate_button.click(
            fn=generate_image,
            inputs=inputs,
            outputs=outputs,
            concurrency_limit=1,
        )
        prompt.submit(
            fn=generate_image,
            inputs=inputs,
            outputs=outputs,
            concurrency_limit=1,
        )

    return demo


if __name__ == "__main__":
    load_dotenv(PROJECT_DIR / ".env")
    build_demo().launch(server_name="127.0.0.1", inbrowser=True)
