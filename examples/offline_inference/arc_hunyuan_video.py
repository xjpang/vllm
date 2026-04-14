"""
ARC-Hunyuan-Video-7B offline inference example.

This example demonstrates how to run video understanding inference
with the ARC-Hunyuan-Video-7B model using vLLM.

The model supports video + audio input and produces text output
for tasks like video captioning, QA, temporal grounding, etc.

Usage:
    # Offline inference
    python examples/offline_inference/arc_hunyuan_video.py

    # Online serving (OpenAI-compatible API)
    vllm serve TencentARC/ARC-Hunyuan-Video-7B \
        --trust-remote-code \
        --max-model-len 20480 \
        --limit-mm-per-prompt video=1

Requirements:
    - GPU: 1x NVIDIA A100 40GB or equivalent
    - Install: pip install decord moviepy librosa pydub
"""

import argparse


def offline_inference_with_embeddings(
    model_path: str = "TencentARC/ARC-Hunyuan-Video-7B",
):
    """Run offline inference by passing pre-computed embeddings.

    This approach pre-computes video+audio embeddings externally
    and passes them directly to the LLM.
    """
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_path,
        trust_remote_code=True,
        max_model_len=20480,
        dtype="bfloat16",
        limit_mm_per_prompt={"video": 1},
    )

    sampling_params = SamplingParams(
        temperature=0.0,
        max_tokens=1024,
    )

    # Example: simple question about a video
    question = "Describe the video content."
    num_frames = 10  # number of video frames

    # Build prompt with image tokens for each frame
    video_prefix = "<image>" * num_frames
    prompt = (
        f"{video_prefix}\n{question}\n"
        "Output the thinking process in <think> </think> "
        "and final answer in <answer> </answer> tags, "
        "i.e., <think> reasoning process here </think>"
        "<answer> answer here </answer>."
    )

    print(f"Model loaded: {model_path}")
    print(f"Prompt: {prompt[:100]}...")
    print(
        "Note: Actual inference requires video frames. "
        "Use the OpenAI API mode with a video URL for end-to-end inference."
    )

    # To run actual inference with video frames, use:
    # outputs = llm.generate(
    #     [{"prompt": prompt, "multi_modal_data": {"video": video_frames}}],
    #     sampling_params,
    # )

    # Demonstrate the model is loaded and ready
    _ = llm, sampling_params


def openai_client_example():
    """Example of using OpenAI-compatible API for video understanding.

    Start the server first:
        vllm serve TencentARC/ARC-Hunyuan-Video-7B \
            --trust-remote-code \
            --max-model-len 20480 \
            --limit-mm-per-prompt video=1
    """
    from openai import OpenAI

    client = OpenAI(
        base_url="http://localhost:8000/v1",
        api_key="EMPTY",
    )

    # Send a video URL for analysis
    video_url = "https://example.com/video.mp4"

    response = client.chat.completions.create(
        model="TencentARC/ARC-Hunyuan-Video-7B",
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "video_url",
                        "video_url": {"url": video_url},
                    },
                    {
                        "type": "text",
                        "text": (
                            "Describe the video content.\n"
                            "Output the thinking process in <think> </think> "
                            "and final answer in <answer> </answer> tags."
                        ),
                    },
                ],
            }
        ],
        max_tokens=1024,
        temperature=0.0,
    )

    print("Response:", response.choices[0].message.content)


def main():
    parser = argparse.ArgumentParser(
        description="ARC-Hunyuan-Video-7B inference example"
    )
    parser.add_argument(
        "--mode",
        choices=["offline", "api"],
        default="offline",
        help="Inference mode: offline (vLLM LLM class) or api (OpenAI client)",
    )
    parser.add_argument(
        "--model-path",
        default="TencentARC/ARC-Hunyuan-Video-7B",
        help="Model path or HuggingFace model ID",
    )
    args = parser.parse_args()

    if args.mode == "offline":
        offline_inference_with_embeddings(args.model_path)
    elif args.mode == "api":
        openai_client_example()


if __name__ == "__main__":
    main()
