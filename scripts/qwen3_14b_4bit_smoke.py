from __future__ import annotations

import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Qwen3-14B 4bit GPU smoke test")
    parser.add_argument("--model-path", default="/home/student222/models/Qwen3-14B")
    parser.add_argument(
        "--prompt",
        default="日本語で一文だけ返答してください。GPU環境は使えていますか?",
    )
    parser.add_argument("--max-new-tokens", type=int, default=32)
    args = parser.parse_args()

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
        quantization_config=quantization_config,
    )

    print("cuda_available", torch.cuda.is_available())
    print("device_count", torch.cuda.device_count())
    print("device_map", getattr(model, "hf_device_map", None))
    if torch.cuda.is_available():
        print(
            "mem_allocated_gb",
            [round(torch.cuda.memory_allocated(i) / 1024**3, 2) for i in range(torch.cuda.device_count())],
        )

    messages = [{"role": "user", "content": args.prompt}]
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    inputs = tokenizer([text], return_tensors="pt").to(model.device)
    with torch.no_grad():
        output = model.generate(
            **inputs,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )
    response = tokenizer.decode(output[0][inputs.input_ids.shape[1] :], skip_special_tokens=True)
    print("response", response.strip())


if __name__ == "__main__":
    main()
