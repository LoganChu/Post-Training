"""Sanity-check the training environment on this machine.

Checks: library versions, Blackwell (sm_120) support in the installed torch build,
availability of the fast Gated DeltaNet kernels, and a real forward/backward pass
of a Qwen3.5 model loaded the same way TRL loads it (AutoModelForCausalLM).

Usage: uv run scripts/check_env.py [--model Qwen/Qwen3.5-0.8B-Base] [--seq-len 2048]
"""

import argparse
import importlib
import importlib.metadata
import inspect
import time

import torch


def version(module: str) -> str:
    try:
        importlib.import_module(module)
    except Exception as e:  # noqa: BLE001 - report any import failure
        return f"MISSING ({type(e).__name__}: {e})"
    dist = {"fla": "flash-linear-attention", "causal_conv1d": "causal-conv1d", "math_verify": "math-verify"}.get(module, module)
    return importlib.metadata.version(dist)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="Qwen/Qwen3.5-0.8B-Base")
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--use-kernels", action="store_true", help="load prebuilt Hub kernels (needs `kernels` package)")
    args = parser.parse_args()

    print("== Versions ==")
    for pkg in ["torch", "transformers", "trl", "vllm", "peft", "accelerate", "datasets", "fla", "causal_conv1d", "math_verify"]:
        print(f"{pkg:14s} {version(pkg)}")

    print("\n== GPU ==")
    assert torch.cuda.is_available(), "CUDA not available"
    cap = torch.cuda.get_device_capability()
    print(f"device         {torch.cuda.get_device_name()}  capability={cap}")
    print(f"torch CUDA     {torch.version.cuda}")
    arch_list = torch.cuda.get_arch_list()
    print(f"compiled archs {arch_list}")
    assert f"sm_{cap[0]}{cap[1]}" in arch_list, "torch build has no kernels for this GPU"

    print("\n== Model forward/backward ==")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda", use_kernels=args.use_kernels
    )
    print(f"class          {type(model).__name__}  params={sum(p.numel() for p in model.parameters()) / 1e9:.2f}B")

    # Report which implementation the DeltaNet layers use. Transformers wraps each PyTorch
    # reference function in a decorator that swaps in the fast package (fla / causal_conv1d)
    # when importable; the chosen implementation lives in the wrapper's closure.
    modeling = importlib.import_module(type(model).__module__)
    for fn_name in ["causal_conv1d_fn", "torch_chunk_gated_delta_rule"]:
        fn = getattr(modeling, fn_name)
        if not inspect.isfunction(fn):  # replaced by a Hub kernel via use_kernels=True
            print(f"{fn_name:28s} hub kernel: {fn!r}")
            continue
        impl = inspect.getclosurevars(fn).nonlocals.get("implementation")
        where = "torch fallback (slow)" if impl is None or impl.__module__ == modeling.__name__ else f"{impl.__module__}.{impl.__name__}"
        print(f"{fn_name:28s} {where}")

    model.gradient_checkpointing_enable()
    model.train()
    ids = torch.randint(0, tok.vocab_size, (1, args.seq_len), device="cuda")
    torch.cuda.reset_peak_memory_stats()
    for i in range(3):  # first step includes Triton autotuning/compilation
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        loss = model(input_ids=ids, labels=ids).loss
        loss.backward()
        model.zero_grad(set_to_none=True)
        torch.cuda.synchronize()
        print(f"step {i}: loss={loss.item():.3f}  time={time.perf_counter() - t0:.2f}s")
    print(f"peak memory    {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB (seq_len={args.seq_len})")
    print("\nOK")


if __name__ == "__main__":
    main()
