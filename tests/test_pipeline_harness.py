"""Automated Test and Performance Monitoring Harness for MTool Localization Pipeline.

Provides non-interactive end-to-end and per-step verification (Clean, Translate,
Validate, and Full Pipeline) designed for automated agent execution. Measures
latency, item/char throughput, peak memory footprint via tracemalloc, and key
fidelity. Supports both hermetic mock LLM server testing and live endpoint benchmarking.
"""

from __future__ import annotations

import argparse
import contextlib
import gc
import http.server
import io
import json
import logging
import re
import socket
import sys
import tempfile
import threading
import time
import tracemalloc
import unittest
import urllib.error
import urllib.request
from collections.abc import Iterator
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# Ensure src/ is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_SRC = _PROJECT_ROOT / "src"
if str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from mtool_translator.cleaner import is_stage1_junk, process_json_file
from mtool_translator.cli import run_pipeline
from mtool_translator.config import load_config, resolve_input_path
from mtool_translator.translator import JP_SOURCE_REGEX, process_translation
from mtool_translator.utils import (
    build_japanese_regex,
    dump_json_file,
    fast_json_dumps,
    fast_json_dumps_bytes,
    fast_json_loads,
    load_japanese_symbols,
    load_json_file,
    repair_json_string,
    strip_engine_escape_codes,
)
from mtool_translator.validator import process_validation


# ==============================================================================
# Performance & Result Data Models
# ==============================================================================

@dataclass
class StepMetrics:
    """Performance and integrity metrics for an individual pipeline step."""

    step_name: str
    status: str  # "PASSED" | "FAILED"
    duration_ms: float
    items_in: int
    items_out: int
    items_per_sec: float
    chars_processed: int
    chars_per_sec: float
    peak_memory_kb: float
    memory_delta_kb: float
    details: dict[str, Any] = field(default_factory=dict)
    error_message: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Converts step metrics to dictionary."""
        return asdict(self)


@dataclass
class PipelineHarnessReport:
    """Full execution report across all tested pipeline stages."""

    timestamp: str
    mode: str  # "mock" | "live"
    overall_status: str  # "PASSED" | "FAILED"
    total_duration_ms: float
    steps: list[StepMetrics] = field(default_factory=list)
    summary: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Converts full harness report to dictionary."""
        return asdict(self)

    def to_json(self, indent: int = 2) -> str:
        """Serializes report to structured JSON string."""
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)

    def print_console_table(self) -> None:
        """Renders formatted ASCII table summary for agent inspection."""
        sep = "=" * 84
        div = "-" * 84
        print("\n" + sep)
        print(f" PIPELINE BENCHMARK & STEP VERIFICATION REPORT ({self.mode.upper()} MODE)")
        print(f" Timestamp: {self.timestamp} | Total Duration: {self.total_duration_ms:.2f}ms")
        print(sep)
        header = (
            f"{'Step Name':<22} {'Status':<10} {'Duration':<12} "
            f"{'Items In/Out':<14} {'Throughput':<14} {'Peak Mem':<10}"
        )
        print(header)
        print(div)

        for step in self.steps:
            items_io = f"{step.items_in}/{step.items_out}"
            tput = f"{step.items_per_sec:.1f} it/s"
            mem = f"{step.peak_memory_kb:.1f} KB"
            dur = f"{step.duration_ms:.2f}ms"
            print(
                f"{step.step_name:<22} {step.status:<10} {dur:<12} "
                f"{items_io:<14} {tput:<14} {mem:<10}"
            )

        print(div)
        print(f" OVERALL RESULT: {self.overall_status}")
        print(sep + "\n")


@contextlib.contextmanager
def silence_all_output(enable: bool = True) -> Iterator[None]:
    """Suppresses stdout, stderr, and module loggers when pure JSON output is requested."""
    if not enable:
        yield
        return

    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = io.StringIO()
    sys.stderr = io.StringIO()

    loggers = [
        logging.getLogger(name)
        for name in (
            "",
            "mtool_translator",
            "mtool_translator.translator",
            "mtool_translator.cleaner",
            "mtool_translator.validator",
        )
    ]
    old_levels = [lg.level for lg in loggers]
    for lg in loggers:
        lg.setLevel(logging.CRITICAL)

    try:
        yield
    finally:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        for lg, lvl in zip(loggers, old_levels):
            lg.setLevel(lvl)


# ==============================================================================
# Model Detection Helper
# ==============================================================================

def detect_running_lm_model(api_endpoint: str) -> str | None:
    """Queries LM Studio /v1/models to detect the currently active model."""
    try:
        base_endpoint = api_endpoint.rsplit("/chat/completions", 1)[0]
        models_url = f"{base_endpoint}/models"
        req = urllib.request.Request(models_url, headers={"User-Agent": "FastLocalHttpClient"})
        with urllib.request.urlopen(req, timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            models = data.get("data", [])
            if models and isinstance(models[0], dict) and "id" in models[0]:
                return str(models[0]["id"])
    except (urllib.error.URLError, OSError, ValueError, KeyError):
        pass
    return None


# ==============================================================================
# Pipeline Mock LLM Server
# ==============================================================================

class PipelineMockLLMHandler(http.server.BaseHTTPRequestHandler):
    """Smart mock handler simulating LLM responses for all three pipeline stages."""

    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: Any) -> None:
        """Suppresses default HTTP server stdout logging."""

    def do_POST(self) -> None:  # pylint: disable=invalid-name
        """Routes and responds to chat completion requests."""
        content_length = int(self.headers.get("Content-Length", 0))
        raw_body = self.rfile.read(content_length)

        try:
            req_data = fast_json_loads(raw_body.decode("utf-8"))
        except (ValueError, UnicodeDecodeError):
            self.send_response(400)
            self.end_headers()
            return

        messages = req_data.get("messages", [])
        system_content = ""
        user_content = ""
        for msg in messages:
            if msg.get("role") == "system":
                system_content += msg.get("content", "") + "\n"
            elif msg.get("role") == "user":
                user_content += msg.get("content", "") + "\n"

        # Combine text for flexible stage detection regardless of message role
        combined_text = f"{system_content}\n{user_content}"

        # 1. Cleaner Batch Classification
        if "Identify developer junk" in combined_text:
            response_content = self._handle_cleaner_request(user_content)

        # 2. Translation Blueprint / Summarization
        elif "Translation Blueprint" in combined_text:
            response_content = self._handle_blueprint_request()

        # 3. Translation Engine Batches
        elif "You are a translation engine" in combined_text:
            response_content = self._handle_translation_batch_request(user_content)

        # 4. Validator Translation Auditor
        elif "You are a translation quality auditor" in combined_text:
            response_content = self._handle_validator_request(user_content)

        else:
            response_content = '{"status": "ok"}'

        resp_data = {
            "id": "mock-harness-id",
            "object": "chat.completion",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": response_content,
                    },
                    "finish_reason": "stop",
                }
            ],
        }

        payload = fast_json_dumps_bytes(resp_data)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.send_header("Connection", "keep-alive")
        self.end_headers()
        self.wfile.write(payload)

    @staticmethod
    def _handle_cleaner_request(user_content: str) -> str:
        """Parses batch lines and marks developer junk lines for quarantine."""
        junk_indices: list[int] = []
        for line in user_content.splitlines():
            if ":" not in line:
                continue
            idx_part, text_part = line.split(":", 1)
            try:
                idx = int(idx_part.strip())
            except ValueError:
                continue

            text_lower = text_part.lower()
            if any(
                marker in text_lower
                for marker in [
                    "todo",
                    "fixme",
                    "debug",
                    "junk",
                    "quarantine",
                    "drop_me",
                    "discard",
                ]
            ):
                junk_indices.append(idx)

        return fast_json_dumps(junk_indices, indent=False)

    @staticmethod
    def _handle_blueprint_request() -> str:
        """Generates standard translation blueprint."""
        return (
            "WORLD & TONE: Heroic fantasy setting with high magic and epic adventurous tone.\n"
            "STORY: A chosen champion embarks on a quest to vanquish the ancient evil.\n"
            "MAIN CHARACTERS (Protagonist and top 2 characters):\n"
            "- 勇者 -> Hero: Swordsman, Male, Determined\n"
            "- アリス -> Alice: Sorceress, Female, Gentle and intelligent\n"
            "- 魔王 -> Demon Lord: Ruler of shadows, Male, Imperious\n\n"
            "CHARACTER NAME MAPPINGS (Top 10):\n"
            "- 勇者 -> Hero (Male)\n"
            "- アリス -> Alice (Female)\n"
            "- 魔王 -> Demon Lord (Male)"
        )

    @staticmethod
    def _handle_translation_batch_request(user_content: str) -> str:
        """Translates Japanese JSON batch to mock English translations."""
        repaired = repair_json_string(user_content)
        try:
            batch_data = fast_json_loads(repaired)
        except ValueError:
            return "{}"

        name_dictionary = {
            "勇者": "Hero",
            "アリス": "Alice",
            "魔王": "Demon Lord",
            "ポーション": "Potion",
            "ハイポーション": "High Potion",
            "決定": "Confirm",
            "キャンセル": "Cancel",
            "セーブ": "Save",
            "ロード": "Load",
        }

        translated: dict[str, str] = {}
        for key, value in batch_data.items():
            val_str = str(value)
            if val_str in name_dictionary:
                translated[key] = name_dictionary[val_str]
            elif "INVALID_TRANSLATION_TRIGGER" in val_str:
                translated[key] = "BAD_CORRUPTED_SYSTEM_ERROR"
            else:
                translated[key] = f"[EN] {val_str}"

        return fast_json_dumps(translated, indent=False)

    @staticmethod
    def _handle_validator_request(user_content: str) -> str:
        """Audits translations and returns validation rating mapping."""
        pair_pattern = re.compile(r"^(\d+)\s*\|\s*JP:\s*(.*?)\s*\|\s*EN:\s*(.*)$")
        ratings: dict[str, int] = {}

        for line in user_content.splitlines():
            line = line.strip()
            match = pair_pattern.match(line)
            if not match:
                continue
            idx, jp_text, en_text = match.groups()
            # If translation has error markers, flag 0 (invalid)
            if any(bad in en_text for bad in ["BAD_CORRUPTED", "SYSTEM_ERROR", "ERROR"]):
                ratings[idx] = 0
            elif en_text == jp_text:
                ratings[idx] = 0
            else:
                ratings[idx] = 1

        return fast_json_dumps(ratings, indent=False)


class PipelineMockServer:
    """Hermetic local mock server for fast and isolated pipeline testing."""

    def __init__(self) -> None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("127.0.0.1", 0))
        self.port = sock.getsockname()[1]
        sock.close()

        self.server = http.server.ThreadingHTTPServer(
            ("127.0.0.1", self.port), PipelineMockLLMHandler
        )
        self.base_url = f"http://127.0.0.1:{self.port}"
        self.api_endpoint = f"{self.base_url}/v1/chat/completions"
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def start(self) -> None:
        """Starts background mock HTTP server."""
        self.thread.start()

    def stop(self) -> None:
        """Stops background mock HTTP server and closes socket."""
        self.server.shutdown()
        self.server.server_close()


# ==============================================================================
# Synthetic Dataset Generator
# ==============================================================================

def generate_harness_dataset(size: int = 30) -> dict[str, str]:
    """Generates synthetic dataset in standard MTool {jp_text: jp_text} format."""
    dialogues = [
        "【勇者】「魔王よ、今こそ世界の平和を取り戻す時だ！」",
        "【アリス】「気をつけて！このダンジョンには危険な罠が仕掛けられています。」",
        "【魔王】「愚かな人間どもめ、深き闇の力に恐れおののくがよい！」",
        "冷たい風が吹き抜け、背筋に寒気が走った。",
        "宝箱を開けると、中から神秘的な光が溢れ出した。",
        "「旅の宿へようこそ。今夜はゆっくりとお休みください。」",
        "古代の石碑には、失われた魔法の詠唱が刻まれていた。",
        "「ありがとう！あなたのおかげで村は救われました。」",
        "深い洞窟の奥から、不気味な咆哮が木霊した。",
        "「約束は守る。これが依頼の報酬だ、受け取るがよい。」",
    ]
    protected_items = [
        "アイテム【黄金の鍵】を手に入れた！",
        "ポーション",
        "ハイポーション",
        "エリクサー",
        "決定",
        "キャンセル",
        "セーブ",
        "ロード",
        "装備",
        "スキル",
    ]
    dev_junk = [
        "// TODO: イベントシーンのフラグ処理を追加すること",
        "/* FIXME: ボスのHPバランス調整 */",
        "debug_flag_scene_intro_01",
        "audio/se/battle_start.wav",
        "img/characters/hero_sprite_sheet.png",
        "DEBUG: プレイヤー座標リセット確認",
        "sys_param_variable_x99",
        "// DROP_ME: テスト用ダミー文字列",
    ]

    dataset: dict[str, str] = {}

    # 1. Dialogues
    for line in dialogues:
        if len(dataset) >= size:
            break
        dataset[line] = line

    # 2. Protected game items & UI
    for line in protected_items:
        if len(dataset) >= size:
            break
        dataset[line] = line

    # 3. Dev junk lines
    for line in dev_junk:
        if len(dataset) >= size:
            break
        dataset[line] = line

    # 4. Cycle through dialogue variations if more lines requested
    cycle_i = 0
    while len(dataset) < size:
        line = f"{dialogues[cycle_i % len(dialogues)]} #{cycle_i + 1}"
        dataset[line] = line
        cycle_i += 1

    return dataset


def create_test_config(
    temp_dir: Path,
    api_endpoint: str,
    api_key: str = "mock-key",
    model: str = "test-model",
    request_timeout: int = 10,
    batch_size: int = 15,
    max_workers: int = 2,
) -> Path:
    """Generates a fully isolated config.json file in the specified directory."""
    config_data = {
        "api_endpoint": api_endpoint,
        "api_key": api_key,
        "model": model,
        "cleanup": {
            "api_endpoint": api_endpoint,
            "api_key": api_key,
            "model": model,
            "request_timeout": request_timeout,
            "batch_size": batch_size,
            "max_workers": max_workers,
            "symbols_filename": "jp_symbols.json",
            "min_japanese_ratio": 0.8,
        },
        "translation": {
            "api_endpoint": api_endpoint,
            "api_key": api_key,
            "model": model,
            "source_language": "Japanese",
            "target_language": "English",
            "batch_size": 10,
            "max_workers": max_workers,
            "request_timeout": request_timeout,
            "enable_pre_translation": True,
            "common_translations_file": "common_translations.json",
            "progress_filename": "test_progress.json",
            "summary_filename": "test_summary.txt",
        },
        "validation": {
            "api_endpoint": api_endpoint,
            "api_key": api_key,
            "model": model,
            "batch_size": 10,
            "max_workers": max_workers,
            "request_timeout": request_timeout,
        },
    }
    cfg_path = temp_dir / "test_config.json"
    dump_json_file(cfg_path, config_data)
    return cfg_path


# ==============================================================================
# Isolated Step Runners with Performance Instrumentation
# ==============================================================================

def run_cleaner_test(
    config_file: str | Path,
    input_data: dict[str, str],
    temp_dir: Path,
    output_dir: Path | None = None,
) -> StepMetrics:
    """Executes and benchmarks Stage 1: Cleaner step."""
    raw_file = temp_dir / "stage1_raw.json"
    dump_json_file(raw_file, input_data)
    chars_total = sum(len(v) for v in input_data.values())

    gc.collect()
    tracemalloc.start()
    t_start = time.perf_counter()

    try:
        target_out = output_dir if output_dir is not None else temp_dir
        cleaned_path, quarantine_path = process_json_file(
            config_file=str(config_file),
            input_file=str(raw_file),
            output_dir=str(target_out),
        )
        duration_ms = (time.perf_counter() - t_start) * 1000
        _current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        cleaned_data = load_json_file(cleaned_path)
        quarantine_data = load_json_file(quarantine_path)

        assert isinstance(cleaned_data, dict), "Cleaned data must be a dict"
        assert isinstance(quarantine_data, dict), "Quarantine data must be a dict"
        assert len(cleaned_data) + len(quarantine_data) == len(input_data), (
            f"Key mismatch: {len(cleaned_data)} cleaned + {len(quarantine_data)} quarantined "
            f"!= {len(input_data)} input keys"
        )

        items_in = len(input_data)
        items_out = len(cleaned_data)
        items_per_sec = (items_in / (duration_ms / 1000)) if duration_ms > 0 else 0.0
        chars_per_sec = (chars_total / (duration_ms / 1000)) if duration_ms > 0 else 0.0

        return StepMetrics(
            step_name="1. Cleaner",
            status="PASSED",
            duration_ms=duration_ms,
            items_in=items_in,
            items_out=items_out,
            items_per_sec=items_per_sec,
            chars_processed=chars_total,
            chars_per_sec=chars_per_sec,
            peak_memory_kb=peak_mem / 1024.0,
            memory_delta_kb=peak_mem / 1024.0,
            details={
                "cleaned_items": len(cleaned_data),
                "quarantined_items": len(quarantine_data),
                "quarantine_ratio": len(quarantine_data) / max(1, items_in),
                "cleaned_file": str(cleaned_path),
                "quarantine_file": str(quarantine_path),
            },
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        duration_ms = (time.perf_counter() - t_start) * 1000
        tracemalloc.stop()
        return StepMetrics(
            step_name="1. Cleaner",
            status="FAILED",
            duration_ms=duration_ms,
            items_in=len(input_data),
            items_out=0,
            items_per_sec=0.0,
            chars_processed=chars_total,
            chars_per_sec=0.0,
            peak_memory_kb=0.0,
            memory_delta_kb=0.0,
            error_message=str(exc),
        )


def run_translator_test(
    config_file: str | Path,
    input_data: dict[str, str],
    temp_dir: Path,
) -> StepMetrics:
    """Executes and benchmarks Stage 2: Translator step."""
    # Filter to Japanese entries matching translator source language expectations
    search = JP_SOURCE_REGEX.search
    valid_source = {k: v for k, v in input_data.items() if search(v)}
    in_file = temp_dir / "stage2_in.json"
    out_file = temp_dir / "stage2_translated.json"
    dump_json_file(in_file, valid_source)
    chars_total = sum(len(v) for v in valid_source.values())

    gc.collect()
    tracemalloc.start()
    t_start = time.perf_counter()

    try:
        res_path = process_translation(
            config_file=str(config_file),
            input_file=str(in_file),
            output_file=str(out_file),
            auto_confirm=True,
            progress_file=temp_dir / "test_trans_progress.json",
            summary_file=temp_dir / "test_trans_summary.txt",
        )
        duration_ms = (time.perf_counter() - t_start) * 1000
        _current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        translated_data = load_json_file(res_path)
        assert isinstance(translated_data, dict), "Translated output must be a dict"
        assert len(translated_data) == len(valid_source), (
            f"Translation output count mismatch: expected {len(valid_source)}, "
            f"got {len(translated_data)}"
        )
        assert all(isinstance(v, str) and len(v.strip()) > 0 for v in translated_data.values()), (
            "All translated values must be non-empty strings"
        )

        items_in = len(valid_source)
        items_out = len(translated_data)
        items_per_sec = (items_in / (duration_ms / 1000)) if duration_ms > 0 else 0.0
        chars_per_sec = (chars_total / (duration_ms / 1000)) if duration_ms > 0 else 0.0

        return StepMetrics(
            step_name="2. Translator",
            status="PASSED",
            duration_ms=duration_ms,
            items_in=items_in,
            items_out=items_out,
            items_per_sec=items_per_sec,
            chars_processed=chars_total,
            chars_per_sec=chars_per_sec,
            peak_memory_kb=peak_mem / 1024.0,
            memory_delta_kb=peak_mem / 1024.0,
            details={
                "output_file": str(res_path.name),
                "all_keys_translated": len(translated_data) == items_in,
            },
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        duration_ms = (time.perf_counter() - t_start) * 1000
        tracemalloc.stop()
        return StepMetrics(
            step_name="2. Translator",
            status="FAILED",
            duration_ms=duration_ms,
            items_in=len(valid_source),
            items_out=0,
            items_per_sec=0.0,
            chars_processed=chars_total,
            chars_per_sec=0.0,
            peak_memory_kb=0.0,
            memory_delta_kb=0.0,
            error_message=str(exc),
        )


def run_validator_test(
    config_file: str | Path,
    original_jp: dict[str, str],
    temp_dir: Path,
) -> StepMetrics:
    """Executes and benchmarks Stage 3: Validator step."""
    # Build translated mapping {jp_source: english_translation}
    # Inject one deliberate corruption to guarantee partitioning verification
    keys = list(original_jp.keys())
    translated_data: dict[str, str] = {}
    for i, k in enumerate(keys):
        if i == len(keys) - 1:
            translated_data[k] = "BAD_CORRUPTED_SYSTEM_ERROR"
        else:
            translated_data[k] = f"[EN] Valid English for line {i + 1}"

    in_file = temp_dir / "stage3_trans.json"
    dump_json_file(in_file, translated_data)
    chars_total = sum(len(v) for v in translated_data.values())

    gc.collect()
    tracemalloc.start()
    t_start = time.perf_counter()

    try:
        val_path, retrans_path = process_validation(
            config_file=str(config_file),
            input_file=str(in_file),
            output_dir=str(temp_dir),
        )
        duration_ms = (time.perf_counter() - t_start) * 1000
        _current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        val_data = load_json_file(val_path)
        retrans_data = load_json_file(retrans_path)

        assert isinstance(val_data, dict), "Validated output must be a dict"
        assert isinstance(retrans_data, dict), "Retranslate output must be a dict"
        assert len(val_data) + len(retrans_data) == len(translated_data), (
            f"Validator count mismatch: {len(val_data)} valid + {len(retrans_data)} retrans "
            f"!= {len(translated_data)} total"
        )
        assert len(retrans_data) >= 1, "Expected at least 1 line routed to retranslate"

        items_in = len(translated_data)
        items_out = len(val_data)
        items_per_sec = (items_in / (duration_ms / 1000)) if duration_ms > 0 else 0.0
        chars_per_sec = (chars_total / (duration_ms / 1000)) if duration_ms > 0 else 0.0

        return StepMetrics(
            step_name="3. Validator",
            status="PASSED",
            duration_ms=duration_ms,
            items_in=items_in,
            items_out=items_out,
            items_per_sec=items_per_sec,
            chars_processed=chars_total,
            chars_per_sec=chars_per_sec,
            peak_memory_kb=peak_mem / 1024.0,
            memory_delta_kb=peak_mem / 1024.0,
            details={
                "validated_items": len(val_data),
                "retranslate_items": len(retrans_data),
                "pass_rate": len(val_data) / max(1, items_in),
            },
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        duration_ms = (time.perf_counter() - t_start) * 1000
        tracemalloc.stop()
        return StepMetrics(
            step_name="3. Validator",
            status="FAILED",
            duration_ms=duration_ms,
            items_in=len(translated_data),
            items_out=0,
            items_per_sec=0.0,
            chars_processed=chars_total,
            chars_per_sec=0.0,
            peak_memory_kb=0.0,
            memory_delta_kb=0.0,
            error_message=str(exc),
        )


def run_pipeline_test(
    config_file: str | Path,
    raw_data: dict[str, str],
    temp_dir: Path,
) -> StepMetrics:
    """Executes and benchmarks the full unified pipeline (Clean -> Translate -> Validate)."""
    raw_file = temp_dir / "pipeline_raw.json"
    dump_json_file(raw_file, raw_data)
    chars_total = sum(len(v) for v in raw_data.values())

    gc.collect()
    tracemalloc.start()
    t_start = time.perf_counter()

    try:
        val_path, retrans_path = run_pipeline(
            config_file=str(config_file),
            input_file=str(raw_file),
            output_dir=str(temp_dir),
            auto_confirm=True,
        )
        duration_ms = (time.perf_counter() - t_start) * 1000
        _current_mem, peak_mem = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        val_data = load_json_file(val_path)
        retrans_data = load_json_file(retrans_path)

        assert isinstance(val_data, dict), "Validated output must be a dict"
        assert isinstance(retrans_data, dict), "Retranslate output must be a dict"
        assert len(val_data) > 0, "Pipeline should yield at least 1 validated line"

        items_in = len(raw_data)
        items_out = len(val_data)
        items_per_sec = (items_in / (duration_ms / 1000)) if duration_ms > 0 else 0.0
        chars_per_sec = (chars_total / (duration_ms / 1000)) if duration_ms > 0 else 0.0

        return StepMetrics(
            step_name="Full Pipeline (E2E)",
            status="PASSED",
            duration_ms=duration_ms,
            items_in=items_in,
            items_out=items_out,
            items_per_sec=items_per_sec,
            chars_processed=chars_total,
            chars_per_sec=chars_per_sec,
            peak_memory_kb=peak_mem / 1024.0,
            memory_delta_kb=peak_mem / 1024.0,
            details={
                "raw_inputs": items_in,
                "validated_items": len(val_data),
                "retranslate_items": len(retrans_data),
            },
        )
    except Exception as exc:  # pylint: disable=broad-exception-caught
        duration_ms = (time.perf_counter() - t_start) * 1000
        tracemalloc.stop()
        return StepMetrics(
            step_name="Full Pipeline (E2E)",
            status="FAILED",
            duration_ms=duration_ms,
            items_in=len(raw_data),
            items_out=0,
            items_per_sec=0.0,
            chars_processed=chars_total,
            chars_per_sec=0.0,
            peak_memory_kb=0.0,
            memory_delta_kb=0.0,
            error_message=str(exc),
        )


# ==============================================================================
# Unittest Test Suite
# ==============================================================================

class TestPipelineHarness(unittest.TestCase):
    """Standard unit test suite discoverable by python -m unittest."""

    mock_server: PipelineMockServer
    temp_dir: tempfile.TemporaryDirectory[str]
    config_path: Path
    test_dataset: dict[str, str]

    @classmethod
    def setUpClass(cls) -> None:
        """Initializes shared mock server and isolated temporary workspace."""
        cls.mock_server = PipelineMockServer()
        cls.mock_server.start()

        cls.temp_dir = tempfile.TemporaryDirectory()
        cls.temp_path = Path(cls.temp_dir.name)
        cls.config_path = create_test_config(cls.temp_path, cls.mock_server.api_endpoint)
        cls.test_dataset = generate_harness_dataset(size=25)

    @classmethod
    def tearDownClass(cls) -> None:
        """Shuts down mock server and cleans temporary workspace."""
        cls.mock_server.stop()
        cls.temp_dir.cleanup()

    def test_01_stage1_cleaner_step(self) -> None:
        """Verifies Step 1 (Cleaner) filters junk and preserves game dialogue."""
        metrics = run_cleaner_test(self.config_path, self.test_dataset, self.temp_path)
        self.assertEqual(metrics.status, "PASSED", f"Cleaner failed: {metrics.error_message}")
        self.assertGreater(metrics.items_out, 0)
        self.assertGreater(metrics.details.get("quarantined_items", 0), 0)

    def test_02_stage2_translator_step(self) -> None:
        """Verifies Step 2 (Translator) translates Japanese text to English."""
        metrics = run_translator_test(self.config_path, self.test_dataset, self.temp_path)
        self.assertEqual(metrics.status, "PASSED", f"Translator failed: {metrics.error_message}")
        self.assertGreater(metrics.items_out, 0)

    def test_03_stage3_validator_step(self) -> None:
        """Verifies Step 3 (Validator) partitions valid vs invalid translations."""
        search = JP_SOURCE_REGEX.search
        dialogue_only = {k: v for k, v in self.test_dataset.items() if search(v)}
        metrics = run_validator_test(self.config_path, dialogue_only, self.temp_path)
        self.assertEqual(metrics.status, "PASSED", f"Validator failed: {metrics.error_message}")
        self.assertGreater(metrics.details.get("retranslate_items", 0), 0)

    def test_04_full_pipeline_step(self) -> None:
        """Verifies end-to-end pipeline execution from raw text to validated output."""
        metrics = run_pipeline_test(self.config_path, self.test_dataset, self.temp_path)
        self.assertEqual(metrics.status, "PASSED", f"Pipeline failed: {metrics.error_message}")
        self.assertGreater(metrics.items_out, 0)

    def test_05_rpg_maker_escape_codes_preservation(self) -> None:
        """Verifies strings with RPG Maker control codes pass Stage 1 without path quarantine."""
        # 1. Direct unit verification of strip_engine_escape_codes
        self.assertEqual(
            strip_engine_escape_codes(r"\C[1]勇者よ、旅立ちの時だ！\!"),
            "勇者よ、旅立ちの時だ！",
        )
        self.assertEqual(
            strip_engine_escape_codes(r"\N[1]様、\V[12]ゴールド手に入れた！\G"),
            "様、ゴールド手に入れた！",
        )
        self.assertEqual(strip_engine_escape_codes(r"\I[45]ポーション"), "ポーション")
        self.assertEqual(strip_engine_escape_codes(r"待て\.よ\^"), "待てよ")
        self.assertEqual(strip_engine_escape_codes("通常テキスト"), "通常テキスト")

        # Windows paths should NEVER be stripped or corrupted
        windows_path = r"C:\Games\Data\actor.png"
        self.assertEqual(strip_engine_escape_codes(windows_path), windows_path)

        # 2. Verify is_stage1_junk behavior
        jp_regex = build_japanese_regex(load_japanese_symbols())

        valid_control_code_lines = [
            (r"\C[1]勇者よ、旅立ちの時だ！\!", r"\C[1]勇者よ、旅立ちの時だ！\!"),
            (r"\N[1]様、\V[12]ゴールド手に入れた！\G", r"\N[1]様、\V[12]ゴールド手に入れた！\G"),
            (r"\I[45]ポーション", r"\I[45]ポーション"),
            (r"待て\.よ\^", r"待て\.よ\^"),
            (r"HP/MP回復", r"HP/MP回復"),
            (r"戦闘／勝利！", r"戦闘／勝利！"),
            (r"決定／キャンセル", r"決定／キャンセル"),
        ]

        for k, text in valid_control_code_lines:
            is_junk, reason = is_stage1_junk(k, text, jp_regex)
            self.assertNotEqual(
                reason,
                "filepath_or_asset",
                f"Line '{text}' was falsely quarantined as '{reason}'",
            )
            self.assertFalse(
                is_junk,
                f"Line '{text}' was marked as junk: {reason}",
            )

        # Real asset paths must still be quarantined
        junk_paths = [
            ("img/faces/hero.png", "img/faces/hero.png"),
            (r"audio\se\battle.wav", r"audio\se\battle.wav"),
            (r"data\actors.json", r"data\actors.json"),
            ("system/graphics/window", "system/graphics/window"),
        ]
        for k, text in junk_paths:
            is_junk, reason = is_stage1_junk(k, text, jp_regex)
            self.assertTrue(is_junk, f"Asset path '{text}' was not marked as junk")
            self.assertEqual(reason, "filepath_or_asset")


# ==============================================================================
# Agent CLI Driver & Benchmark Runner
# ==============================================================================

def execute_harness(
    step: str = "all",
    live: bool = False,
    num_items: int = 30,
    input_file: str | None = None,
    output_dir: str | None = None,
    model_override: str | None = None,
) -> PipelineHarnessReport:
    """Executes requested test stages and generates a comprehensive performance report."""
    mock_server: PipelineMockServer | None = None
    temp_dir = tempfile.TemporaryDirectory()
    temp_path = Path(temp_dir.name)
    mode = "live" if live else "mock"

    try:
        if live:
            cfg = load_config("config.json")
            endpoint = cfg.get("api_endpoint", "http://127.0.0.1:1234/v1/chat/completions")
            api_key = cfg.get("api_key", "lm-studio")
            detected_model = detect_running_lm_model(endpoint)
            model = model_override or detected_model or cfg.get("model", "test-model")
            clean_cfg = cfg.get("cleanup", {})
            timeout = int(clean_cfg.get("request_timeout", 60))
            batch_size = int(clean_cfg.get("batch_size", 30))
            max_workers = int(clean_cfg.get("max_workers", 4))
            config_path = create_test_config(
                temp_path,
                endpoint,
                api_key=api_key,
                model=model,
                request_timeout=timeout,
                batch_size=batch_size,
                max_workers=max_workers,
            )
        else:
            mock_server = PipelineMockServer()
            mock_server.start()
            config_path = create_test_config(temp_path, mock_server.api_endpoint)

        if input_file:
            resolved_in = resolve_input_path(input_file, default_subfolder="raw")
            loaded_data = load_json_file(resolved_in)
            if not isinstance(loaded_data, dict):
                raise ValueError(f"Input file '{input_file}' must contain a JSON dictionary.")
            dataset = loaded_data
        else:
            dataset = generate_harness_dataset(size=num_items)

        out_path = Path(output_dir) if output_dir else None
        if out_path:
            out_path.mkdir(parents=True, exist_ok=True)

        steps_to_run = ["clean", "translate", "validate", "pipeline"] if step == "all" else [step]

        results: list[StepMetrics] = []
        overall_passed = True
        t_all_start = time.perf_counter()

        for st in steps_to_run:
            if st == "clean":
                res = run_cleaner_test(config_path, dataset, temp_path, output_dir=out_path)
            elif st == "translate":
                res = run_translator_test(config_path, dataset, temp_path)
            elif st == "validate":
                search = JP_SOURCE_REGEX.search
                dialogues = {k: v for k, v in dataset.items() if search(v)}
                res = run_validator_test(config_path, dialogues, temp_path)
            elif st == "pipeline":
                res = run_pipeline_test(config_path, dataset, temp_path)
            else:
                continue

            results.append(res)
            if res.status != "PASSED":
                overall_passed = False

        total_dur = (time.perf_counter() - t_all_start) * 1000

        report = PipelineHarnessReport(
            timestamp=datetime.now(timezone.utc).isoformat(),
            mode=mode,
            overall_status="PASSED" if overall_passed else "FAILED",
            total_duration_ms=total_dur,
            steps=results,
            summary={
                "steps_executed": len(results),
                "steps_passed": sum(1 for r in results if r.status == "PASSED"),
                "total_items_processed": sum(r.items_in for r in results),
            },
        )
        return report
    finally:
        if mock_server:
            mock_server.stop()
        temp_dir.cleanup()


def main() -> None:
    """CLI entrypoint for standalone test and benchmark execution by the AI agent."""
    parser = argparse.ArgumentParser(
        description="Agent Pipeline Test & Performance Monitoring Harness",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--step",
        choices=["clean", "translate", "validate", "pipeline", "all"],
        default="all",
        help="Target step or full pipeline to test",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Execute against live LM Studio instance configured in config.json",
    )
    parser.add_argument(
        "-i",
        "--input-file",
        type=str,
        default=None,
        help="Input JSON file path to use instead of synthetic dataset",
    )
    parser.add_argument(
        "-o",
        "--output-dir",
        type=str,
        default=None,
        help="Optional directory to persist cleaned or translated output files",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Explicit model name override for live mode",
    )
    parser.add_argument(
        "--num-items",
        type=int,
        default=25,
        help="Number of synthetic items to generate for testing",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output raw machine-readable JSON metrics to stdout",
    )
    parser.add_argument(
        "--report-file",
        type=str,
        default=None,
        help="Optional path to write JSON report file",
    )

    args = parser.parse_args()

    with silence_all_output(enable=args.json):
        report = execute_harness(
            step=args.step,
            live=args.live,
            num_items=args.num_items,
            input_file=args.input_file,
            output_dir=args.output_dir,
            model_override=args.model,
        )

    if args.report_file:
        out_p = Path(args.report_file)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        with open(out_p, "w", encoding="utf-8") as f:
            f.write(report.to_json())

    if args.json:
        print(report.to_json())
    else:
        report.print_console_table()

    sys.exit(0 if report.overall_status == "PASSED" else 1)


if __name__ == "__main__":
    main()
