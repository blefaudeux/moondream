from moondream import Moondream, detect_device, LATEST_REVISION
from transformers import AutoTokenizer
from PIL import Image
import torch
import time


def run_benchmark(profile: bool = False, compile: bool = False):
    device, dtype = detect_device()
    model_id = "vikhyatk/moondream2"
    tokenizer = AutoTokenizer.from_pretrained(model_id, revision=LATEST_REVISION)
    moondream = Moondream.from_pretrained(
        model_id,
        revision=LATEST_REVISION,
        torch_dtype=dtype,
    ).to(device=device)

    moondream.eval()
    moondream.vision_encoder._compile = compile

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

    if profile:

        def trace_handler(prof):
            print(
                prof.key_averages().table(sort_by="self_cuda_time_total", row_limit=-1)
            )
            prof.export_chrome_trace("moondream_chrome_trace.json")

        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            schedule=torch.profiler.schedule(wait=0, warmup=1, active=1, repeat=0),
            on_trace_ready=trace_handler,
        ) as p:
            for _ in range(5):
                run_once()
                p.step()
    else:
        # Actual benchmark
        run_once(print_output=True)  # Warmup
        run_once()  # Warmup

        torch.cuda.synchronize()
        times = []
        for _ in range(10):
            start = time.time()
            run_once()
            torch.cuda.synchronize()
            times.append(time.time() - start)

        times = sorted(times)
        median_time = times[len(times) // 2]
        mad = sum(abs(t - median_time) for t in times) / len(times)
        print(f"Time per iteration: {median_time:.2f}s  +/- {mad:.2f}s")


if __name__ == "__main__":
    run_benchmark(profile=False, compile=False)
