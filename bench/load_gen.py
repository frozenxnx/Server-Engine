import argparse, asyncio, json, time
import httpx, uuid

def make_prompt(n_words: int, seed: int) -> str:
    # Unique prefix per request so prefix caching can't hide prefill cost.
    return f"Request {seed}. " + " ".join(["hello"] * n_words)

async def one_request(client, url, model, prompt, max_tokens):
    payload = {
        "model": model, "prompt": prompt, "max_tokens": max_tokens,
        "temperature": 0, "stream": True,
        "stream_options": {"include_usage": True},
        "ignore_eos": True,  # fixed output length -> comparable runs
    }
    t0 = time.perf_counter()
    first, usage = None, None
    async with client.stream("POST", url, json=payload) as r:
        r.raise_for_status()
        async for line in r.aiter_lines():
            if not line.startswith("data: "):
                continue
            data = line[6:]
            if data == "[DONE]":
                break
            obj = json.loads(data)
            if obj.get("usage"):
                usage = obj["usage"]
            if obj.get("choices") and first is None:
                first = time.perf_counter()
    t1 = time.perf_counter()
    first = first or t1
    n_out = usage["completion_tokens"] if usage else 0
    return {
        "ttft": first - t0,
        "e2e": t1 - t0,
        "tpot": (t1 - first) / max(n_out - 1, 1),
        "out_tokens": n_out,
        "in_tokens": usage["prompt_tokens"] if usage else 0,
    }

def pct(xs, p):
    xs = sorted(xs)
    return xs[min(int(len(xs) * p), len(xs) - 1)]

async def run(args):
    url = f"http://localhost:{args.port}/v1/completions"
    sem = asyncio.Semaphore(args.concurrency)
    run_id = uuid.uuid4().hex[:8]  # new every run, so no cross-run cache hits
    async with httpx.AsyncClient(timeout=httpx.Timeout(300.0)) as client:
        async def guarded(i):
            async with sem:
                return await one_request(
                    client, url, args.model,
                    make_prompt(args.prompt_words, f"{run_id}-{i}"), args.max_tokens)
        t0 = time.perf_counter()
        results = await asyncio.gather(*[guarded(i) for i in range(args.num_requests)])
        wall = time.perf_counter() - t0

    return {
        "concurrency": args.concurrency,
        "num_requests": args.num_requests,
        "wall_s": round(wall, 2),
        "out_tok_per_s": round(sum(r["out_tokens"] for r in results) / wall, 1),
        "in_tok_avg": sum(r["in_tokens"] for r in results) / len(results),
        "ttft_p50": round(pct([r["ttft"] for r in results], 0.5), 3),
        "ttft_p95": round(pct([r["ttft"] for r in results], 0.95), 3),
        "tpot_p50": round(pct([r["tpot"] for r in results], 0.5), 4),
        "tpot_p95": round(pct([r["tpot"] for r in results], 0.95), 4),
    }

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--num-requests", type=int, default=64)
    ap.add_argument("--prompt-words", type=int, default=512)
    ap.add_argument("--max-tokens", type=int, default=128)
    print(json.dumps(asyncio.run(run(ap.parse_args())), indent=2))
