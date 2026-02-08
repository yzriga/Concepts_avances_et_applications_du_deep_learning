from __future__ import annotations

import os
from PIL import Image
from pipeline_utils import DEFAULT_MODEL_ID, load_text2img, get_device, make_generator

def save(img: Image.Image, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    img.save(path)

def run_text2img_experiments() -> None:
    model_id = DEFAULT_MODEL_ID
    seed = 42
    prompt = "ultra-realistic product photo of a leather handbag on a white background, studio lighting, soft shadow, very sharp"  # TODO: choisissez un prompt e-commerce en anglais unique et gardez-le identique pour tous les runs
    negative = "text, watermark, logo, low quality, blurry, deformed"

    plan = [
        # name, scheduler, steps, guidance
        ("run01_baseline", "EulerA", 30, 7.5),
        ("run02_steps15", "EulerA", 15, 7.5),
        ("run03_steps50", "EulerA", 50, 7.5),
        ("run04_guid4",  "EulerA", 30, 4.0),
        ("run05_guid12", "EulerA", 30, 12.0),
        ("run06_ddim",   "DDIM",   30, 7.5),
    ]

    for name, scheduler_name, steps, guidance in plan:
        pipe = load_text2img(model_id, scheduler_name)
        device = get_device()
        g = make_generator(seed, device)

        out = pipe(
            prompt=prompt,
            negative_prompt=negative,
            num_inference_steps=steps,
            guidance_scale=guidance,
            height=512,
            width=512,
            generator=g,
        )

        img = out.images[0]
        save(img, f"outputs/t2i_{name}.png")
        print("T2I", name, {"scheduler": scheduler_name, "seed": seed, "steps": steps, "guidance": guidance})

def main() -> None:
    model_id = DEFAULT_MODEL_ID
    scheduler_name = "EulerA"  # "EulerA" recommandé pour démarrer
    seed = 42
    steps = 30
    guidance = 7.5

    prompt = "ultra-realistic product photo of a backpack on a white background, studio lighting, soft shadow, very sharp"
    negative = "text, watermark, logo, low quality, blurry, deformed"

    pipe = load_text2img(model_id, scheduler_name)
    device = get_device()
    g = make_generator(seed, device)

    out = pipe(
        prompt=prompt,
        negative_prompt=negative,
        num_inference_steps=steps,
        guidance_scale=guidance,
        height=512,
        width=512,
        generator=g,
    )

    img = out.images[0]
    save(img, "outputs/baseline.png")

    print("OK saved outputs/baseline.png")
    print("CONFIG:", {"model_id": model_id, "scheduler": scheduler_name, "seed": seed, "steps": steps, "guidance": guidance})
    
    # Lancer les expériences contrôlées
    print("\n=== STARTING TEXT2IMG EXPERIMENTS ===")
    run_text2img_experiments()
    print("=== TEXT2IMG EXPERIMENTS COMPLETED ===")

if __name__ == "__main__":
    main()