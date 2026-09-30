"""MTool Translation Diagnostic Runner.

Executes translation logic from src/mtool_translator/translator.py with deep
diagnostic telemetry. Logs detailed context for all errors, timeouts, retries,
prompts, raw LLM responses, and key mappings to dedicated log files.
"""

from __future__ import annotations

import argparse
import datetime
import logging
import sys
import threading
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any

# Ensure src/ is on sys.path
_PROJECT_ROOT = Path(__file__).resolve().parent
_SRC_DIR = _PROJECT_ROOT / "src"
if str(_SRC_DIR) not in sys.path:
    sys.path.insert(0, str(_SRC_DIR))

# pylint: disable=wrong-import-position
from mtool_translator.config import load_config, resolve_input_path, resolve_output_path
from mtool_translator.http_client import HttpRequestError
from mtool_translator.translator import (
    DEFAULT_MAX_TOKENS,
    JSONTranslator,
    clean_japanese_text,
    parse_llm_json_response,
)
from mtool_translator.utils import (
    dump_json_file,
    fast_json_dumps,
    fast_json_loads,
)


class DiagnosticTelemetryRecorder:  # pylint: disable=too-many-instance-attributes
    """Thread-safe recorder for rich translation troubleshooting telemetry."""

    def __init__(self, log_dir: Path, session_id: str) -> None:
        self.log_dir = log_dir
        self.session_id = session_id
        self.log_dir.mkdir(parents=True, exist_ok=True)

        self.text_log_path = self.log_dir / f"translation_{session_id}.log"
        self.jsonl_path = self.log_dir / f"translation_{session_id}_diagnostics.jsonl"
        self.summary_path = self.log_dir / f"translation_{session_id}_summary.json"

        self._lock = threading.Lock()
        self._event_count = 0
        self._error_counts: dict[str, int] = {}

        self.logger = self._setup_file_logger()

    def _setup_file_logger(self) -> logging.Logger:
        logger = logging.getLogger(f"DiagnosticTranslator_{self.session_id}")
        logger.setLevel(logging.INFO)
        logger.propagate = False

        formatter = logging.Formatter(
            "%(asctime)s - [%(levelname)s] - (%(threadName)s) %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )

        file_handler = logging.FileHandler(self.text_log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

        return logger

    def record_event(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
        self,
        event_type: str,
        stage: str,
        message: str,
        batch_index: int | None = None,
        attempt: int | None = None,
        duration_ms: float | None = None,
        status_code: int | None = None,
        prompt_system: str | None = None,
        prompt_user: str | None = None,
        raw_response: str | None = None,
        error_type: str | None = None,
        error_details: str | None = None,
        exc: BaseException | None = None,
        batch_items: list[tuple[str, str]] | None = None,
        extra_diagnostics: dict[str, Any] | None = None,
    ) -> None:
        """Records a structured troubleshooting event to JSONL."""
        now_iso = datetime.datetime.now(datetime.timezone.utc).isoformat()
        tb_str = None
        if exc is not None:
            tb_str = "".join(traceback.format_exception(type(exc), exc, exc.__traceback__))
            if error_type is None:
                error_type = exc.__class__.__name__
            if error_details is None:
                error_details = str(exc)

        record: dict[str, Any] = {
            "timestamp": now_iso,
            "session_id": self.session_id,
            "event_type": event_type,
            "stage": stage,
            "message": message,
            "batch_index": batch_index,
            "attempt": attempt,
            "duration_ms": duration_ms,
            "status_code": status_code,
            "error_type": error_type,
            "error_details": error_details,
            "traceback": tb_str,
            "prompt_system": prompt_system,
            "prompt_user": prompt_user,
            "raw_response": raw_response,
            "batch_items_sample": (
                [{"key": k, "val": v} for k, v in batch_items[:5]] if batch_items else None
            ),
            "batch_items_total": len(batch_items) if batch_items else 0,
            "diagnostics": extra_diagnostics or {},
        }

        # Filter None values to keep JSON clean
        compact_record = {k: v for k, v in record.items() if v is not None}

        with self._lock:
            self._event_count += 1
            self._error_counts[event_type] = self._error_counts.get(event_type, 0) + 1
            try:
                with open(self.jsonl_path, "a", encoding="utf-8") as f:
                    f.write(fast_json_dumps(compact_record) + "\n")
            except OSError as err:
                self.logger.error("Failed to write diagnostic record: %s", err)

    def write_summary(self, summary_data: dict[str, Any]) -> None:
        """Persists aggregate diagnostic session summary."""
        with self._lock:
            summary_data["total_diagnostic_events"] = self._event_count
            summary_data["event_type_breakdown"] = dict(self._error_counts)
            try:
                dump_json_file(self.summary_path, summary_data, indent=True)
            except OSError as err:
                self.logger.error("Failed to write summary file: %s", err)


class DiagnosticTranslator(JSONTranslator):
    """Subclass of JSONTranslator with deep troubleshooting telemetry."""

    def __init__(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        config_file: str = "config.json",
        telemetry: DiagnosticTelemetryRecorder | None = None,
        max_batches: int | None = None,
        sample_blueprint_lines: int | None = 2000,
    ) -> None:
        super().__init__(config_file=config_file)
        self.telemetry = telemetry
        self.max_batches = max_batches
        self.sample_blueprint_lines = sample_blueprint_lines

        self.stats_lock = threading.Lock()
        self.stats = {
            "total_batches_executed": 0,
            "successful_batches": 0,
            "retried_batches": 0,
            "failed_batches": 0,
            "total_items_processed": 0,
            "valid_translations": 0,
            "missing_keys_count": 0,
            "validation_failures": 0,
            "total_api_latency_ms": 0.0,
        }

    def _request_blueprint_summary(self, prompt: str, item: str, log_tag: str) -> str | None:
        """Instruments blueprint summary generation with detailed timing and error logging."""
        data = {
            "model": self.config["model"],
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": item},
            ],
            "temperature": 0.0,
            "max_tokens": DEFAULT_MAX_TOKENS,
        }
        t0 = time.perf_counter()
        try:
            resp = self.session.post(
                self.api_url,
                headers=self.api_headers,
                json=data,
                timeout=self.config["request_timeout"],
            )
            duration_ms = (time.perf_counter() - t0) * 1000.0
            if resp.status_code == 200:
                result_stripped = resp.json()["choices"][0]["message"]["content"].strip()
                if self.telemetry:
                    self.telemetry.logger.info(
                        "%s generated successfully in %.1fms.", log_tag, duration_ms
                    )
                return result_stripped

            # HTTP Error response
            if self.telemetry:
                self.telemetry.record_event(
                    event_type="blueprint_http_error",
                    stage="blueprint",
                    message=f"{log_tag} returned status {resp.status_code}",
                    duration_ms=duration_ms,
                    status_code=resp.status_code,
                    prompt_system=prompt,
                    prompt_user=item[:1000],
                    raw_response=resp.text,
                    error_details=resp.text,
                )
        except (HttpRequestError, ValueError, KeyError) as err:
            duration_ms = (time.perf_counter() - t0) * 1000.0
            if self.telemetry:
                self.telemetry.record_event(
                    event_type="blueprint_exception",
                    stage="blueprint",
                    message=f"{log_tag} failed with exception: {err}",
                    duration_ms=duration_ms,
                    prompt_system=prompt,
                    prompt_user=item[:1000],
                    exc=err,
                )
        return None

    def generate_blueprint(
        self,
        original_data: dict[str, Any],
        summary_path: Path,
        auto_confirm: bool = False,
    ) -> str:
        """Generates or loads Blueprint with intelligent sampling and section logging."""
        if not summary_path.exists():
            if self.telemetry:
                self.telemetry.logger.info(
                    "--- Generating Translation Blueprint (Diagnostic Mode) ---"
                )

            # Extract dialogue and narrative lines preferentially for world lore
            raw_texts = [
                str(v)
                for v in original_data.values()
                if v and len(str(v).strip()) > 3 and not str(v).strip().isdigit()
            ]

            if self.sample_blueprint_lines and len(raw_texts) > self.sample_blueprint_lines:
                if self.telemetry:
                    self.telemetry.logger.info(
                        "Sampling top %d story/dialogue lines out of %d for blueprint...",
                        self.sample_blueprint_lines,
                        len(raw_texts),
                    )
                sampled_texts = raw_texts[: self.sample_blueprint_lines]
            else:
                sampled_texts = raw_texts

            summary_batches = self.chunker.process_all(sampled_texts, self.summarize)
            summary = self.reduce_summaries(summary_batches)

            summary_path.parent.mkdir(parents=True, exist_ok=True)
            with open(summary_path, "w", encoding="utf-8") as file:
                file.write(summary)

            if self.telemetry:
                self.telemetry.logger.info(
                    "Translation Blueprint successfully cached to %s", summary_path
                )

            if not auto_confirm:
                input(f"Summary saved to '{summary_path}'. Review it and press Enter...")

        with open(summary_path, "r", encoding="utf-8") as f:
            return f.read()

    def translate_batch(self, item: tuple[int, list[tuple[str, str]], str]) -> dict[str, str]:
        """Translates a batch and records comprehensive diagnostic telemetry."""
        index, texts, summary = item
        fallback_results = dict(texts)

        if not texts:
            return {}

        cleaned_dict = {
            str(i + 1): clean_japanese_text(value) for i, (_k, value) in enumerate(texts)
        }
        json_batch = fast_json_dumps(cleaned_dict, indent=False)

        source_lang = self.config.get("source_language", "Japanese")
        target_lang = self.config.get("target_language", "English")

        prompt = (
            f"You are a translation engine.\n"
            f"Translate all JSON string values from {source_lang} to {target_lang}.\n\n"
            f"TRANSLATION BLUEPRINT:\n{summary}\n\n"
            f"RULES:\n"
            f"1. Translate JSON values only. Keep keys unchanged.\n"
            f"2. Follow character names, gender, and tone from the TRANSLATION BLUEPRINT.\n"
            f"3. Keep technical terms, code, and control characters (\\n, \\t) unchanged.\n"
            f"4. Output raw JSON only. Do not use Markdown code blocks. Do not add explanations."
        )

        headers, api_url = self._get_api_headers_and_url()
        data = {
            "model": self.config["model"],
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": json_batch},
            ],
            "temperature": 0.2,
            "max_tokens": DEFAULT_MAX_TOKENS,
        }

        return self._send_diagnostic_translation_request(
            batch_index=index,
            api_url=api_url,
            headers=headers,
            data=data,
            texts=texts,
            fallback_results=fallback_results,
        )

    def _send_diagnostic_translation_request(  # pylint: disable=too-many-arguments,too-many-positional-arguments,too-many-locals
        self,
        batch_index: int,
        api_url: str,
        headers: dict[str, str],
        data: dict[str, Any],
        texts: list[tuple[str, str]],
        fallback_results: dict[str, str],
    ) -> dict[str, str]:
        """Handles HTTP dispatch with per-attempt diagnostics and error capture."""
        max_retries = self.config.get("max_retries", 3)
        system_prompt = data["messages"][0]["content"]
        user_prompt = data["messages"][1]["content"]

        with self.stats_lock:
            self.stats["total_batches_executed"] += 1
            self.stats["total_items_processed"] += len(texts)

        for attempt in range(max_retries):
            t0 = time.perf_counter()
            resp = None
            raw_content = None
            try:
                resp = self.session.post(
                    api_url,
                    headers=headers,
                    json=data,
                    timeout=self.config["request_timeout"],
                )
                duration_ms = (time.perf_counter() - t0) * 1000.0

                with self.stats_lock:
                    self.stats["total_api_latency_ms"] += duration_ms

                if resp.status_code == 200:
                    result = resp.json()
                    choices = result.get("choices", [])
                    if choices:
                        raw_content = choices[0]["message"]["content"].strip()
                        parsed = parse_llm_json_response(raw_content)

                        mapped = self._map_diagnostic_response(
                            batch_index=batch_index,
                            texts=texts,
                            translated_json=parsed,
                            raw_content=raw_content,
                            user_prompt=user_prompt,
                        )

                        with self.stats_lock:
                            self.stats["successful_batches"] += 1
                            if attempt > 0:
                                self.stats["retried_batches"] += 1

                        return mapped

                elif resp.status_code == 429:
                    if self.telemetry:
                        self.telemetry.record_event(
                            event_type="rate_limit_429",
                            stage="translation",
                            message=f"Batch {batch_index} hit rate limit on attempt {attempt + 1}",
                            batch_index=batch_index,
                            attempt=attempt + 1,
                            duration_ms=duration_ms,
                            status_code=429,
                            prompt_system=system_prompt,
                            prompt_user=user_prompt,
                        )
                    time.sleep(self.config["retry_delay"])
                else:
                    if self.telemetry:
                        self.telemetry.record_event(
                            event_type="http_error",
                            stage="translation",
                            message=f"Batch {batch_index} HTTP Error {resp.status_code}",
                            batch_index=batch_index,
                            attempt=attempt + 1,
                            duration_ms=duration_ms,
                            status_code=resp.status_code,
                            prompt_system=system_prompt,
                            prompt_user=user_prompt,
                            raw_response=resp.text,
                        )

            except (HttpRequestError, ValueError, KeyError) as err:
                duration_ms = (time.perf_counter() - t0) * 1000.0
                if self.telemetry:
                    self.telemetry.record_event(
                        event_type="request_exception",
                        stage="translation",
                        message=f"Batch {batch_index} attempt {attempt + 1} failed: {err}",
                        batch_index=batch_index,
                        attempt=attempt + 1,
                        duration_ms=duration_ms,
                        status_code=resp.status_code if resp else None,
                        prompt_system=system_prompt,
                        prompt_user=user_prompt,
                        raw_response=raw_content or (resp.text if resp else None),
                        exc=err,
                        batch_items=texts,
                    )

            if attempt < max_retries - 1:
                time.sleep(self.config.get("retry_delay", 1))

        # Max retries exhausted
        with self.stats_lock:
            self.stats["failed_batches"] += 1

        if self.telemetry:
            self.telemetry.record_event(
                event_type="batch_failed_all_retries",
                stage="translation",
                message=(
                    f"Batch {batch_index} exhausted all {max_retries} attempts. "
                    "Returning fallbacks."
                ),
                batch_index=batch_index,
                prompt_system=system_prompt,
                prompt_user=user_prompt,
                batch_items=texts,
            )

        return fallback_results

    def _map_diagnostic_response(  # pylint: disable=too-many-arguments,too-many-positional-arguments
        self,
        batch_index: int,
        texts: list[tuple[str, str]],
        translated_json: dict[str, Any],
        raw_content: str,
        user_prompt: str,
    ) -> dict[str, str]:
        """Maps JSON response to keys and audits missing keys and validation errors."""
        translated_results: dict[str, str] = {}
        missing_keys: list[str] = []
        validation_errors: list[dict[str, str]] = []

        for i, (key, original_value) in enumerate(texts):
            lookup_key = str(i + 1)
            if lookup_key in translated_json and translated_json[lookup_key] is not None:
                translated_line = str(translated_json[lookup_key]).strip()
                if self.is_valid_translation(translated_line):
                    translated_results[key] = translated_line
                    with self.stats_lock:
                        self.stats["valid_translations"] += 1
                else:
                    translated_results[key] = original_value
                    with self.stats_lock:
                        self.stats["validation_failures"] += 1
                    validation_errors.append(
                        {
                            "key": key,
                            "original": original_value,
                            "rejected_translation": translated_line,
                        }
                    )
            else:
                translated_results[key] = original_value
                missing_keys.append(lookup_key)
                with self.stats_lock:
                    self.stats["missing_keys_count"] += 1

        if missing_keys and self.telemetry:
            self.telemetry.record_event(
                event_type="missing_keys_in_response",
                stage="translation",
                message=(
                    f"Batch {batch_index} missing {len(missing_keys)}/{len(texts)} "
                    "keys in response."
                ),
                batch_index=batch_index,
                raw_response=raw_content,
                prompt_user=user_prompt,
                batch_items=texts,
                extra_diagnostics={
                    "missing_keys": missing_keys,
                    "expected_count": len(texts),
                    "received_count": len(translated_json),
                },
            )

        if validation_errors and self.telemetry:
            self.telemetry.record_event(
                event_type="translation_validation_failure",
                stage="translation",
                message=f"Batch {batch_index} had {len(validation_errors)} validation failures.",
                batch_index=batch_index,
                raw_response=raw_content,
                extra_diagnostics={"validation_errors": validation_errors},
            )

        return translated_results

    def _translate_batches(
        self,
        items: list[tuple[str, str]],
        summary: str,
        translated_data: dict[str, str],
        progress_path: Path,
    ) -> None:
        """Overrides _translate_batches to enforce optional --max-batches limit."""
        batch_size = self.config.get("batch_size", 30)
        batches = [items[i : i + batch_size] for i in range(0, len(items), batch_size)]

        if self.max_batches is not None and self.max_batches > 0:
            if self.telemetry:
                self.telemetry.logger.info(
                    "Limiting execution to %d batches (out of %d total batches)",
                    self.max_batches,
                    len(batches),
                )
            batches = batches[: self.max_batches]

        worker_count = self.config.get("max_workers", 4)
        giga_chunks = [
            batches[i : i + worker_count] for i in range(0, len(batches), worker_count)
        ]

        if self.telemetry:
            self.telemetry.logger.info(
                "Processing %d batches across %d waves (Worker count: %d)...",
                len(batches),
                len(giga_chunks),
                worker_count,
            )

        from concurrent.futures import ThreadPoolExecutor  # pylint: disable=import-outside-toplevel

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            for giga_index, giga_chunk in enumerate(giga_chunks):
                t_wave_start = time.perf_counter()
                self._run_wave_with_executor(executor, giga_chunk, summary, translated_data)
                self.save_progress(translated_data, progress_path)
                wave_duration = time.perf_counter() - t_wave_start

                if self.telemetry:
                    self.telemetry.logger.info(
                        "Wave %d/%d complete (%.2fs). Checkpoint saved (%d total items).",
                        giga_index + 1,
                        len(giga_chunks),
                        wave_duration,
                        len(translated_data),
                    )


def inspect_diagnostics_file(jsonl_path: Path) -> None:
    """Reads and pretty-prints an existing diagnostic telemetry file."""
    if not jsonl_path.exists():
        print(f"Error: Telemetry file '{jsonl_path}' not found.")
        return

    print("=" * 70)
    print(f"DIAGNOSTIC TELEMETRY INSPECTION: {jsonl_path.name}")
    print("=" * 70)

    events: list[dict[str, Any]] = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                try:
                    events.append(fast_json_loads(line))
                except (ValueError, TypeError):
                    pass

    print(f"Total Recorded Anomaly Events: {len(events)}\n")
    if not events:
        print("No errors or anomalies recorded! Translation ran clean.")
        return

    counts = Counter(e.get("event_type", "unknown") for e in events)
    print("Event Type Breakdown:")
    for etype, count in counts.items():
        print(f"  - {etype}: {count}")

    print("\n--- Event Details (Chronological) ---")
    for i, e in enumerate(events, 1):
        print(
            f"\n[{i}] {e.get('timestamp')} | Event: {e.get('event_type')} | Stage: {e.get('stage')}"
        )
        print(f"    Message: {e.get('message')}")
        if e.get("batch_index") is not None:
            lat = e.get("duration_ms", 0)
            print(
                f"    Batch: {e.get('batch_index')}, Attempt: {e.get('attempt')}, "
                f"Duration: {lat:.1f}ms"
            )
        if e.get("error_type"):
            print(f"    Error: {e.get('error_type')}: {e.get('error_details')}")
        if e.get("prompt_user"):
            user_sample = e["prompt_user"][:200].replace("\n", " ")
            print(f"    User Prompt Sample: {user_sample}...")
        if e.get("raw_response"):
            resp_sample = e["raw_response"][:200].replace("\n", " ")
            print(f"    Raw LLM Output: {resp_sample}...")
        if e.get("diagnostics"):
            print(f"    Extra Diagnostics: {e['diagnostics']}")

    print("\n" + "=" * 70)


def build_arg_parser() -> argparse.ArgumentParser:
    """Builds CLI argument parser for the diagnostic translation runner."""
    parser = argparse.ArgumentParser(
        description="MTool Translation Diagnostic Runner & Telemetry Logger",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("-i", "--input", help="Path to cleaned input JSON", default=None)
    parser.add_argument("-o", "--output", help="Path to target output JSON", default=None)
    parser.add_argument("-c", "--config", help="Path to config.json", default="config.json")
    parser.add_argument("-p", "--progress", help="Path to progress checkpoint JSON", default=None)
    parser.add_argument("-s", "--summary", help="Path to translation blueprint file", default=None)
    parser.add_argument("--log-dir", help="Directory for diagnostic logs", default="logs")
    parser.add_argument(
        "--max-batches",
        type=int,
        default=None,
        help="Optional maximum number of batches to run (for canary testing)",
    )
    parser.add_argument(
        "--sample-blueprint",
        type=int,
        default=2000,
        help="Max story lines to sample for Blueprint generation to avoid timeouts",
    )
    parser.add_argument(
        "-y", "--yes", action="store_true", help="Auto-confirm prompts without pausing"
    )
    parser.add_argument(
        "--inspect",
        type=str,
        default=None,
        help="Path to an existing .jsonl diagnostic file to inspect and summarize",
    )
    return parser


def main() -> None:  # pylint: disable=too-many-locals
    """Diagnostic translation entry point."""
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.inspect:
        inspect_diagnostics_file(Path(args.inspect))
        return

    session_id = datetime.datetime.now(datetime.timezone.utc).strftime("%Y%m%d_%H%M%S")
    log_dir = Path(args.log_dir)
    telemetry = DiagnosticTelemetryRecorder(log_dir=log_dir, session_id=session_id)

    telemetry.logger.info("============================================================")
    telemetry.logger.info("Starting MTool Translation with Diagnostic Telemetry")
    telemetry.logger.info("Session ID:      %s", session_id)
    telemetry.logger.info("Log File:        %s", telemetry.text_log_path)
    telemetry.logger.info("JSONL Errors:    %s", telemetry.jsonl_path)
    telemetry.logger.info("============================================================")

    config = load_config(args.config, section="translation")
    in_file = resolve_input_path(
        args.input or config.get("input_filename", "ManualTransFile_cleaned.json"),
        default_subfolder="processed",
    )
    out_file = resolve_output_path(
        args.output or config.get("output_filename", "ManualTransFile_translated.json"),
        default_subfolder="processed",
    )
    progress_file = (
        Path(args.progress)
        if args.progress
        else resolve_output_path(
            config.get("progress_filename", "translation_progress.json"),
            default_subfolder="processed",
        )
    )
    summary_file = (
        Path(args.summary)
        if args.summary
        else resolve_output_path(
            config.get("summary_filename", "summary.txt"),
            default_subfolder="processed",
        )
    )

    t_run_start = time.perf_counter()

    try:
        with DiagnosticTranslator(
            config_file=args.config,
            telemetry=telemetry,
            max_batches=args.max_batches,
            sample_blueprint_lines=args.sample_blueprint,
        ) as translator:
            success = translator.translate_json_file(
                input_file=in_file,
                output_file=out_file,
                progress_file=progress_file,
                summary_file=summary_file,
                auto_confirm=args.yes,
            )

        duration_total = time.perf_counter() - t_run_start

        telemetry.logger.info("============================================================")
        telemetry.logger.info("Translation Execution Completed in %.2fs", duration_total)
        telemetry.logger.info("Batches Executed:   %d", translator.stats["total_batches_executed"])
        telemetry.logger.info("Successful:         %d", translator.stats["successful_batches"])
        telemetry.logger.info("Retried Batches:    %d", translator.stats["retried_batches"])
        telemetry.logger.info("Failed Batches:     %d", translator.stats["failed_batches"])
        telemetry.logger.info("Items Processed:    %d", translator.stats["total_items_processed"])
        telemetry.logger.info("Valid Translations: %d", translator.stats["valid_translations"])
        telemetry.logger.info("Missing Keys:       %d", translator.stats["missing_keys_count"])
        telemetry.logger.info("Validation Errors:  %d", translator.stats["validation_failures"])
        telemetry.logger.info("Diagnostic Records: %s", telemetry.jsonl_path)
        telemetry.logger.info("============================================================")

        summary_payload = {
            "session_id": session_id,
            "success": success,
            "duration_total_seconds": duration_total,
            "input_file": str(in_file),
            "output_file": str(out_file),
            "stats": translator.stats,
        }
        telemetry.write_summary(summary_payload)

    except KeyboardInterrupt:
        telemetry.logger.warning("Translation process interrupted by user.")
    except Exception as err:  # noqa: BLE001 # pylint: disable=broad-exception-caught
        telemetry.logger.error("Fatal error during diagnostic translation: %s", err)
        telemetry.record_event(
            event_type="fatal_runner_exception",
            stage="runner",
            message=f"Fatal exception: {err}",
            exc=err,
        )


if __name__ == "__main__":
    main()
