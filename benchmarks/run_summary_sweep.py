"""Comprehensive Summary Benchmark Suite (Translation Blueprint Generation).

Evaluates:
  1. Chunk size / prompt context scaling (5k, 12k, 20k tokens)
  2. Prefill vs generation throughput (TPS)
  3. Structural compliance & markdown code fence adherence
  4. Character name discovery & gender/role extraction recall
  5. KV cache quantization impact (f16 vs q8_0 vs q4_0) on extraction precision
  6. Engine eval_batch_size (1024, 2048, 4096)
  7. Cross-model shootout:
     - google/gemma-4-12b-qat
     - qwen2.5-7b-instruct-abliterated
     - qwen2.5-14b-instruct-abliterated
     - mistral-small-24b-instruct-2501
     - qwen3.5-9b-uncensored-hauhaucs-aggressive
"""

from __future__ import annotations

import json
import logging
import os
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

# pylint: disable=wrong-import-position
from mtool_translator.http_client import FastLocalHttpClient, HttpRequestError
from mtool_translator.lm_studio import load_model, unload_all_models
from mtool_translator.translator import SUMMARIZE_PROMPT

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("summary_sweep")

BLUEPRINT_KEYS = [
    "WORLD & TONE:",
    "STORY:",
    "MAIN CHARACTERS",
    "CHARACTER NAME MAPPINGS",
]

KNOWN_CHARACTERS = [
    "リーゼル",
    "マリア",
    "ライナス",
    "アレック",
    "サンドラ",
    "ミレディ",
    "ポラリス",
    "シュバルツ",
    "エリック",
    "ムールムール",
    "シルフ",
    "カルロス",
    "ジャン",
    "オルテナ",
]

CONFIG_FILE_MAP: dict[str, str] = {
    "google/gemma-4-12b-qat": r"google\gemma-4-12b-qat.json",
    "qwen2.5-7b-instruct-abliterated": r"mradermacher\Qwen2.5-7B-Instruct-abliterated-GGUF\Qwen2.5-7B-Instruct-abliterated.Q5_K_M.gguf.json",
    "qwen2.5-14b-instruct-abliterated": r"mradermacher\Qwen2.5-14B-Instruct-abliterated-GGUF\Qwen2.5-14B-Instruct-abliterated.Q4_K_M.gguf.json",
    "mistral-small-24b-instruct-2501": r"mistralai\Mistral-Small-24B-Instruct-2501\Mistral-Small-24B-Instruct-2501-Q4_K_M.gguf.json",
    "qwen3.5-9b-uncensored-hauhaucs-aggressive": r"HauhauCS\Qwen3.5-9B-Uncensored-HauhauCS-Aggressive\Qwen3.5-9B-Uncensored-HauhauCS-Aggressive-Q4_K_M.gguf.json",
}


@dataclass
class SummarySweepResult:
    experiment: str
    model: str
    chunk_tokens_label: str
    kv_cache_type: str
    eval_batch_size: int
    context_length: int
    duration_s: float
    prompt_tokens: int
    completion_tokens: int
    prompt_tps: float
    completion_tps: float
    structure_score: float
    has_code_fences: bool
    character_hits: int
    character_recall_pct: float
    found_characters: list[str]
    blueprint_snippet: str


def update_model_kv_quant(model_key: str, kv_type: str) -> None:
    rel_path = CONFIG_FILE_MAP.get(model_key)
    if not rel_path:
        return
    base_dir = Path(r"C:\Users\inoy\.lmstudio\.internal\user-concrete-model-default-config")
    target_path = base_dir / rel_path
    target_path.parent.mkdir(parents=True, exist_ok=True)

    if target_path.exists():
        data = json.loads(target_path.read_text(encoding="utf-8"))
    else:
        data = {"preset": "", "operation": {"fields": []}, "load": {"fields": []}}

    fields = data.setdefault("load", {}).setdefault("fields", [])
    found_keys = {"k": False, "v": False, "flash": False}
    for fld in fields:
        if fld.get("key") == "llm.load.llama.kCacheQuantizationType":
            found_keys["k"] = True
            fld["value"] = {"checked": False, "value": "q8_0"} if kv_type == "f16" else {"checked": True, "value": kv_type}
        elif fld.get("key") == "llm.load.llama.vCacheQuantizationType":
            found_keys["v"] = True
            fld["value"] = {"checked": False, "value": "q8_0"} if kv_type == "f16" else {"checked": True, "value": kv_type}
        elif fld.get("key") == "llm.load.llama.flashAttention":
            found_keys["flash"] = True
            fld["value"] = True

    if not found_keys["k"]:
        val = {"checked": False, "value": "q8_0"} if kv_type == "f16" else {"checked": True, "value": kv_type}
        fields.append({"key": "llm.load.llama.kCacheQuantizationType", "value": val})
    if not found_keys["v"]:
        val = {"checked": False, "value": "q8_0"} if kv_type == "f16" else {"checked": True, "value": kv_type}
        fields.append({"key": "llm.load.llama.vCacheQuantizationType", "value": val})
    if not found_keys["flash"]:
        fields.append({"key": "llm.load.llama.flashAttention", "value": True})

    target_path.write_text(json.dumps(data, indent=2), encoding="utf-8")


def score_structure(content: str) -> tuple[float, bool]:
    if not content:
        return 0.0, False
    score = 0.0
    for k in BLUEPRINT_KEYS:
        if k in content:
            score += 25.0
    has_fences = "```" in content
    if has_fences:
        score = max(0.0, score - 15.0)
    lower = content.lower()
    if lower.startswith("here is") or lower.startswith("certainly"):
        score = max(0.0, score - 10.0)
    return score, has_fences


def extract_known_characters(content: str) -> list[str]:
    found = []
    for c in KNOWN_CHARACTERS:
        if c in content:
            found.append(c)
        elif c == "リーゼル" and ("Liesel" in content or "リーゼル" in content or "Lizel" in content):
            found.append("リーゼル")
        elif c == "ポラリス" and "Polaris" in content:
            found.append("ポラリス")
        elif c == "シュバルツ" and "Schwarz" in content:
            found.append("シュバルツ")
    return list(dict.fromkeys(found))


def run_summary_trial(
    experiment: str,
    model_key: str,
    chunk_text: str,
    chunk_label: str,
    kv_cache_type: str = "q8_0",
    eval_batch_size: int = 2048,
    physical_batch_size: int = 1024,
    context_length: int = 32768,
    max_tokens: int = 1024,
    base_url: str = "http://127.0.0.1:1234",
) -> SummarySweepResult:
    update_model_kv_quant(model_key, kv_cache_type)
    unload_all_models(base_url)

    load_cfg = {
        "context_length": context_length,
        "parallel": 1,
        "eval_batch_size": eval_batch_size,
        "physical_batch_size": physical_batch_size,
        "flash_attention": True,
        "offload_kv_cache_to_gpu": True,
    }
    loaded = load_model(base_url, model_key, load_cfg, timeout=120.0)
    if not loaded:
        logger.error("Failed to load %s for %s", model_key, experiment)
        return SummarySweepResult(
            experiment=experiment,
            model=model_key,
            chunk_tokens_label=chunk_label,
            kv_cache_type=kv_cache_type,
            eval_batch_size=eval_batch_size,
            context_length=context_length,
            duration_s=0.0,
            prompt_tokens=0,
            completion_tokens=0,
            prompt_tps=0.0,
            completion_tps=0.0,
            structure_score=0.0,
            has_code_fences=False,
            character_hits=0,
            character_recall_pct=0.0,
            found_characters=[],
            blueprint_snippet="",
        )

    client = FastLocalHttpClient(max_connections=2)
    req_body: dict[str, Any] = {
        "model": model_key,
        "messages": [
            {"role": "system", "content": SUMMARIZE_PROMPT},
            {"role": "user", "content": chunk_text},
        ],
        "temperature": 0.0,
        "max_tokens": max_tokens,
        "reasoning_effort": "none",
    }

    t0 = time.perf_counter()
    content = ""
    prompt_tokens = 0
    completion_tokens = 0
    try:
        resp = client.post(f"{base_url}/v1/chat/completions", json=req_body, timeout=300)
        dur = time.perf_counter() - t0
        if resp.status_code == 200:
            data = resp.json()
            content = data["choices"][0]["message"].get("content", "").strip()
            usage = data.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", len(chunk_text) // 2)
            completion_tokens = usage.get("completion_tokens", len(content) // 4)
    except HttpRequestError as err:
        dur = time.perf_counter() - t0
        logger.error("Error during completion for %s: %s", model_key, err)
    finally:
        client.close()

    prompt_tps = prompt_tokens / max(0.01, dur)
    comp_tps = completion_tokens / max(0.01, dur)
    struct_score, fences = score_structure(content)
    found_chars = extract_known_characters(content)
    # Total distinct known characters present in chunk1_16k is 13
    char_recall = (len(found_chars) / 13.0) * 100.0

    snippet = content[:200].replace("\n", " ") if content else ""

    return SummarySweepResult(
        experiment=experiment,
        model=model_key,
        chunk_tokens_label=chunk_label,
        kv_cache_type=kv_cache_type,
        eval_batch_size=eval_batch_size,
        context_length=context_length,
        duration_s=dur,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        prompt_tps=prompt_tps,
        completion_tps=comp_tps,
        structure_score=struct_score,
        has_code_fences=fences,
        character_hits=len(found_chars),
        character_recall_pct=char_recall,
        found_characters=found_chars,
        blueprint_snippet=snippet,
    )


def run_summary_shootout() -> None:
    reconfig = getattr(sys.stdout, "reconfigure", None)
    if callable(reconfig):
        reconfig(encoding="utf-8")

    # Load chunk datasets
    p_16k = _PROJECT_ROOT / "data/benchmark/summary_sample_chunk1_16k.json"
    chunk_16k = "\n".join(json.loads(p_16k.read_text(encoding="utf-8")))

    p_1000 = _PROJECT_ROOT / "data/benchmark/summary_sample_1000.json"
    chunk_10k = "\n".join(json.loads(p_1000.read_text(encoding="utf-8")))

    p_500 = _PROJECT_ROOT / "data/benchmark/summary_sample_500.json"
    chunk_5k = "\n".join(json.loads(p_500.read_text(encoding="utf-8")))

    all_results: list[SummarySweepResult] = []

    print("=" * 80)
    print("STARTING IN-DEPTH SUMMARY (TRANSLATION BLUEPRINT) BENCHMARK SUITE")
    print("=" * 80)

    # -----------------------------------------------------------------
    # EXPERIMENT 1: CHUNK SIZE / TOKEN BUDGET SCALING (5k, 10k, 20k tokens)
    # -----------------------------------------------------------------
    print("\n--- [1. CHUNK CONTEXT SCALING (5k, 10k, 20k tokens)] ---")
    chunk_trials = [
        ("Chunk-5k", chunk_5k, "5k tokens"),
        ("Chunk-10k", chunk_10k, "10k tokens"),
        ("Chunk-20k", chunk_16k, "20k tokens"),
    ]
    for label, text, tok_label in chunk_trials:
        for m in ["google/gemma-4-12b-qat", "qwen2.5-7b-instruct-abliterated"]:
            res = run_summary_trial(
                experiment=f"{label}-{m.split('/')[-1]}",
                model_key=m,
                chunk_text=text,
                chunk_label=tok_label,
                kv_cache_type="q8_0",
                eval_batch_size=2048,
                context_length=32768,
            )
            all_results.append(res)
            print(
                f"[{m[:15]} | {tok_label:10}] Time: {res.duration_s:5.2f}s | "
                f"Prompt TPS: {res.prompt_tps:6.1f} | Struct: {res.structure_score:4.1f}% | "
                f"Chars: {res.character_hits} ({', '.join(res.found_characters[:3])})"
            )

    # -----------------------------------------------------------------
    # EXPERIMENT 2: K/V CACHE QUANTIZATION (f16 vs q8_0 vs q4_0)
    # -----------------------------------------------------------------
    print("\n--- [2. KV CACHE QUANTIZATION (f16, q8_0, q4_0) ON 20K CHUNK] ---")
    for kv in ["f16", "q8_0", "q4_0"]:
        for m in ["google/gemma-4-12b-qat", "qwen2.5-7b-instruct-abliterated"]:
            res = run_summary_trial(
                experiment=f"KV-{kv}-{m.split('/')[-1]}",
                model_key=m,
                chunk_text=chunk_16k,
                chunk_label="20k tokens",
                kv_cache_type=kv,
                eval_batch_size=2048,
                context_length=32768,
            )
            all_results.append(res)
            print(
                f"[{m[:15]} | KV={kv:4}] Time: {res.duration_s:5.2f}s | "
                f"Prompt TPS: {res.prompt_tps:6.1f} | Struct: {res.structure_score:4.1f}% | "
                f"Chars: {res.character_hits}"
            )

    # -----------------------------------------------------------------
    # EXPERIMENT 3: EVAL BATCH SIZE SCALING (1024, 2048, 4096)
    # -----------------------------------------------------------------
    print("\n--- [3. ENGINE EVAL BATCH SIZE (1024, 2048, 4096)] ---")
    for eb in [1024, 2048, 4096]:
        for m in ["google/gemma-4-12b-qat", "qwen2.5-7b-instruct-abliterated"]:
            res = run_summary_trial(
                experiment=f"EvalBatch-eb{eb}-{m.split('/')[-1]}",
                model_key=m,
                chunk_text=chunk_16k,
                chunk_label="20k tokens",
                kv_cache_type="q8_0",
                eval_batch_size=eb,
                physical_batch_size=1024,
                context_length=32768,
            )
            all_results.append(res)
            print(
                f"[{m[:15]} | eb={eb:4}] Time: {res.duration_s:5.2f}s | "
                f"Prompt TPS: {res.prompt_tps:6.1f} | Struct: {res.structure_score:4.1f}%"
            )

    # -----------------------------------------------------------------
    # EXPERIMENT 4: FULL CROSS-MODEL BLUEPRINT QUALITY SHOOTOUT
    # -----------------------------------------------------------------
    print("\n--- [4. FULL CROSS-MODEL BLUEPRINT QUALITY SHOOTOUT (20k CHUNK)] ---")
    cross_models = [
        "google/gemma-4-12b-qat",
        "qwen2.5-7b-instruct-abliterated",
        "qwen2.5-14b-instruct-abliterated",
        "mistral-small-24b-instruct-2501",
        "qwen3.5-9b-uncensored-hauhaucs-aggressive",
    ]
    for m in cross_models:
        res = run_summary_trial(
            experiment=f"ModelShootout-{m.split('/')[-1]}",
            model_key=m,
            chunk_text=chunk_16k,
            chunk_label="20k tokens",
            kv_cache_type="q8_0",
            eval_batch_size=4096 if "gemma" in m else 2048,
            physical_batch_size=1024,
            context_length=32768,
        )
        all_results.append(res)
        print(
            f"[{m:40}] Time: {res.duration_s:5.2f}s | Prompt TPS: {res.prompt_tps:6.1f} | "
            f"Struct: {res.structure_score:4.1f}% | Chars: {res.character_hits}/13 | "
            f"Fences: {res.has_code_fences}"
        )

    # Persist JSON report
    out_dir = _PROJECT_ROOT / "benchmarks/results"
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = int(time.time())
    report_file = out_dir / f"summary_parameter_sweep_{ts}.json"
    report_file.write_text(
        json.dumps([asdict(r) for r in all_results], ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(f"\nSaved summary parameter sweep report to: {report_file.resolve()}\n")


if __name__ == "__main__":
    run_summary_shootout()
