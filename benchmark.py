from moondream import Moondream, detect_device, LATEST_REVISION
from transformers import AutoTokenizer
from PIL import Image
import torch
import time


torch._inductor.config.compile_threads = 16
torch.set_float32_matmul_precision("high")


def benchmark(compile: bool = False):
    device, dtype = detect_device()
    model_id = "vikhyatk/moondream2"
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=LATEST_REVISION)
    moondream = Moondream.from_pretrained(
        model_id,
        revision=LATEST_REVISION,
        torch_dtype=dtype,
    ).to(device=device)

    moondream.eval()
    moondream.vision_encoder.compile = compile

    image1 = Image.open("assets/demo-1.jpg")
    image2 = Image.open("assets/demo-2.jpg")
    prompts = [
        "What is the girl doing?",
        "What color is the girl's hair?",
        "What is this?",
        "What is behind the stand?",
    ]

    def run_once(print_output: bool = False):
        answers = moondream.batch_answer(
            images=[image1, image1, image2, image2],
            prompts=prompts,
            tokenizer=tokenizer,
        )

        if print_output:
            for question, answer in zip(prompts, answers):
                print(f"Q: {question}")
                print(f"A: {answer}")
                print()

    # Warmup
    for _ in range(5):
        run_once(print_output=True)

    # Actual benchmark
    torch.cuda.synchronize()
    start = time.time()
    for _ in range(10):
        run_once()

    torch.cuda.synchronize()
    end = time.time()
    print(f"Time taken: {end - start:.2f}s")


if __name__ == "__main__":
    benchmark(compile=False)
