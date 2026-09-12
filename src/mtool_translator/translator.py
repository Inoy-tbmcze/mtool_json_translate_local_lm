"""Batched LLM translation module with token-aware chunking and progressive checkpointing."""

from __future__ import annotations

import logging
import math
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from types import TracebackType
from typing import Any, Self

from .config import load_config, resolve_input_path, resolve_output_path
from .http_client import FastLocalHttpClient, HttpRequestError
from .native_core import fast_count_jp_and_ascii
from .utils import (
    clean_japanese_text,
    dump_json_file,
    fast_json_dumps,
    load_json_file,
    parse_llm_json_response,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

ERROR_PATTERNS = [
    "system error",
    "internal error",
    "error:",
    "exception:",
    "traceback",
    "rate limit exceeded",
    "quota exceeded",
    "model overloaded",
]

DEFAULT_MAX_TOKENS = 1500
JP_SOURCE_REGEX = re.compile(r"[\u3040-\u30ff\u4e00-\u9faf]")
JP_TOKEN_RATIO = 1.1
ASCII_TOKEN_RATIO = 0.28

BLUEPRINT_STRUCTURE = (
    "Output MUST follow this exact structure:\n\n"
    "WORLD & TONE: [Maximum 15 words. Describe setting and tone]\n"
    "STORY: [Maximum 15 words. Describe core story]\n"
    "MAIN CHARACTERS (Protagonist and top 2 characters):\n"
    "- [Original Name] -> [English Name]: [Role], [Gender], [Speaking style]\n"
    "- [Original Name] -> [English Name]: [Role], [Gender], [Speaking style]\n"
    "- [Original Name] -> [English Name]: [Role], [Gender], [Speaking style]\n\n"
    "CHARACTER NAME MAPPINGS (Top 10):\n"
    "- [Original Name] -> [English Name] ([Gender])\n\n"
    "Do not output markdown code blocks. Do not add explanations."
)

SUMMARIZE_PROMPT = (
    "Analyze the raw text and create a concise Translation Blueprint.\n\n" + BLUEPRINT_STRUCTURE
)

SUMMARIZE_SUMMARIES_PROMPT = (
    "Synthesize multi-part translation notes into one short Translation Blueprint.\n\n"
    + BLUEPRINT_STRUCTURE
)


class TokenAwareChunker:
    """Chunks text into token-budgeted batches using an ultra-fast heuristic estimator."""

    __slots__ = ("max_tokens", "model_name")

    def __init__(
        self,
        model_name: str | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
    ) -> None:
        self.model_name = model_name
        self.max_tokens = max_tokens

    def estimate_tokens(self, text: str) -> int:
        """Estimates token count using character/script ratio heuristics.

        Formula: max(1, floor(jp_count * 1.1 + ascii_count * 0.28))
        """
        if not text:
            return 0

        jp_count, ascii_count = fast_count_jp_and_ascii(text)
        return max(1, math.floor(jp_count * JP_TOKEN_RATIO + ascii_count * ASCII_TOKEN_RATIO))

    def create_chunks(self, texts: list[str]) -> list[list[str]]:
        """Groups texts into chunks that fit within the token budget."""
        chunks: list[list[str]] = []
        current_chunk: list[str] = []
        current_tokens = 0

        for text in texts:
            tokens = self.estimate_tokens(text)
            if current_tokens + tokens > self.max_tokens and current_chunk:
                chunks.append(current_chunk)
                current_chunk = []
                current_tokens = 0
            current_chunk.append(text)
            current_tokens += tokens

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def process_all(self, texts: list[str], process_func: Any) -> list[Any]:
        """Processes chunks sequentially using the provided worker function."""
        chunks = self.create_chunks(texts)
        results = []

        for i, chunk in enumerate(chunks):
            chunk_text = "\n".join(chunk)
            result = process_func(chunk_text)
            if result:
                results.append(result)
            else:
                logger.warning("Chunk %d returned None.", i)

        return results


class JSONTranslator:
    """Translation manager handling prompts, retries, and checkpointing."""

    def __init__(self, config_file: str = "config.json"):
        self.chunker = TokenAwareChunker()
        self.config = self._init_config(config_file)
        self.logger = logger
        self.print_summary = True

        max_workers = self.config.get("max_workers", 4)
        self.session = FastLocalHttpClient(max_connections=max_workers)
        self.api_headers, self.api_url = self._get_api_headers_and_url()

    def close(self) -> None:
        """Closes the underlying HTTP session."""
        self.session.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

    def _init_config(self, config_file: str) -> dict[str, Any]:
        config = load_config(config_file, section="translation")

        required_keys = [
            "api_endpoint",
            "api_key",
            "model",
            "source_language",
            "target_language",
        ]
        for key in required_keys:
            if key not in config:
                raise ValueError(f"Missing required configuration key: {key}")

        config.setdefault("max_retries", 3)
        config.setdefault("retry_delay", 5)
        config.setdefault("request_timeout", 60)
        config.setdefault("batch_size", 30)
        config.setdefault("save_interval", 100)
        config.setdefault("api_type", "openai")
        config.setdefault("enable_pre_translation", True)
        config.setdefault("common_translations_file", "common_translations.json")

        return config

    def _get_api_headers_and_url(self) -> tuple[dict[str, str], str]:
        if self.config.get("api_type", "openai") == "google":
            headers = {"Content-Type": "application/json"}
            api_url = f"{self.config['api_endpoint']}?key={self.config['api_key']}"
        else:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config['api_key']}",
            }
            api_url = self.config["api_endpoint"]
        return headers, api_url

    def _load_common_translations(self) -> dict[str, str]:
        """Loads common game translations dictionary from configured JSON file."""
        if not self.config.get("enable_pre_translation", True):
            return {}

        dict_file = self.config.get("common_translations_file", "common_translations.json")
        dict_path = resolve_input_path(dict_file, default_subfolder="")
        if not dict_path.exists():
            dict_path = resolve_input_path(dict_file, default_subfolder="raw")

        if not dict_path.exists():
            self.logger.info(
                "Common translations file '%s' not found. Skipping pre-translation.",
                dict_file,
            )
            return {}

        try:
            data = load_json_file(dict_path)
            if isinstance(data, dict):
                self.logger.info(
                    "Loaded %d common translations from '%s'.",
                    len(data),
                    dict_path.name,
                )
                return {str(k): str(v) for k, v in data.items()}
            self.logger.warning(
                "Common translations file '%s' is not a JSON object.",
                dict_file,
            )
        except (OSError, ValueError, TypeError, KeyError) as err:
            self.logger.error("Failed to load common translations from '%s': %s", dict_file, err)

        return {}

    def _apply_pre_translations(
        self,
        original_data: dict[str, Any],
        translated_data: dict[str, str],
        common_dict: dict[str, str],
    ) -> int:
        """Applies exact dictionary matches before sending batches to the LLM."""
        if not common_dict:
            return 0

        pre_count = 0
        for key, value in original_data.items():
            if key in translated_data:
                continue

            target_str = str(value) if value is not None else ""
            if not target_str:
                continue

            if target_str in common_dict:
                translated_data[key] = common_dict[target_str]
                pre_count += 1
                continue

            stripped = target_str.strip()
            if stripped in common_dict:
                n_orig = len(target_str)
                n_strip = len(stripped)
                if n_orig == n_strip:
                    translated_data[key] = common_dict[stripped]
                else:
                    start = target_str.find(stripped)
                    leading = target_str[:start]
                    trailing = target_str[start + n_strip :]
                    translated_data[key] = f"{leading}{common_dict[stripped]}{trailing}"
                pre_count += 1

        return pre_count

    def _request_blueprint_summary(self, prompt: str, item: str, log_tag: str) -> str | None:
        data = {
            "model": self.config["model"],
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": item},
            ],
            "temperature": 0.0,
            "max_tokens": 2048,
        }
        try:
            resp = self.session.post(
                self.api_url,
                headers=self.api_headers,
                json=data,
                timeout=self.config["request_timeout"],
            )
            if resp.status_code == 200:
                result_stripped = resp.json()["choices"][0]["message"]["content"].strip()
                self.logger.info("%s generated successfully.", log_tag)
                return result_stripped
        except (HttpRequestError, ValueError, KeyError) as err:
            self.logger.error("%s request failed: %s", log_tag, err)
        return None

    def summarize(self, item: str) -> Any:
        """Generates a concise Translation Blueprint for character and tone consistency."""
        return self._request_blueprint_summary(SUMMARIZE_PROMPT, item, "Section summary")

    def summarize_summaries(self, item: str) -> Any:
        """Synthesizes multiple summary parts into a single blueprint."""
        return self._request_blueprint_summary(SUMMARIZE_SUMMARIES_PROMPT, item, "Reduced summary")

    def reduce_summaries(self, lst: list[str], max_depth: int = 5) -> str:
        """Combines and reduces summaries hierarchically."""
        if not lst:
            return ""
        if len(lst) == 1:
            return lst[0]

        current_items = lst
        depth = 0

        while len(current_items) > 1 and depth < max_depth:
            depth += 1
            print(f"[Hierarchical Pass {depth}] Combining {len(current_items)} summary items...")

            chunks = self.chunker.create_chunks(current_items)
            if len(chunks) == len(current_items) and all(len(c) == 1 for c in chunks):
                break

            next_level_summaries = []
            for chunk in chunks:
                chunk_text = "\n".join(chunk)
                summary = self.summarize_summaries(chunk_text)
                if summary:
                    next_level_summaries.append(summary)
                else:
                    next_level_summaries.extend(chunk)

            if len(next_level_summaries) >= len(current_items):
                current_items = next_level_summaries
                break

            current_items = next_level_summaries

        if len(current_items) == 1:
            return current_items[0]

        final_concat = "\n\n".join(current_items)
        final_summary = self.summarize_summaries(final_concat)
        return final_summary if final_summary else final_concat

    def translate_batch(self, item: tuple[int, list[tuple[str, str]], str]) -> dict[str, str]:
        """Translates a single batch and returns key-translation pairs."""
        _index, texts, summary = item
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

        if self.print_summary:
            self.print_summary = False
            self.logger.info("\nFinal Prompt Sample:\n%s", prompt)

        headers, api_url = self._get_api_headers_and_url()
        data = {
            "model": self.config["model"],
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": json_batch},
            ],
            "temperature": 0.2,
            "max_tokens": 65536,
        }

        return self._send_translation_request(api_url, headers, data, texts, fallback_results)

    def _send_translation_request(
        self,
        api_url: str,
        headers: dict[str, str],
        data: dict[str, Any],
        texts: list[tuple[str, str]],
        fallback_results: dict[str, str],
    ) -> dict[str, str]:
        for attempt in range(self.config["max_retries"]):
            try:
                resp = self.session.post(
                    api_url,
                    headers=headers,
                    json=data,
                    timeout=self.config["request_timeout"],
                )
                if resp.status_code == 200:
                    result = resp.json()
                    choices = result.get("choices", [])
                    if choices:
                        raw_content = choices[0]["message"]["content"].strip()
                        parsed = parse_llm_json_response(raw_content)
                        return self._map_translation_response(texts, parsed)
                elif resp.status_code == 429:
                    self.logger.warning(
                        "Rate limit hit. Waiting %ds...", self.config["retry_delay"]
                    )
                    time.sleep(self.config["retry_delay"])
                elif resp.status_code == 401:
                    self.logger.error("API key unauthorized. Aborting batch.")
                    return fallback_results
                else:
                    self.logger.error("API Error (%d): %s", resp.status_code, resp.text)
            except (HttpRequestError, ValueError, KeyError) as err:
                self.logger.error("Attempt %d error during translation: %s", attempt + 1, err)

            if attempt < self.config["max_retries"] - 1:
                time.sleep(self.config["retry_delay"])

        self.logger.error("Max retries exhausted for batch. Returning fallbacks.")
        return fallback_results

    def _map_translation_response(
        self, texts: list[tuple[str, str]], translated_json: dict[str, Any]
    ) -> dict[str, str]:
        translated_results: dict[str, str] = {}
        for i, (key, original_value) in enumerate(texts):
            lookup_key = str(i + 1)
            if lookup_key in translated_json and translated_json[lookup_key] is not None:
                translated_line = str(translated_json[lookup_key]).strip()
                if self.is_valid_translation(translated_line):
                    translated_results[key] = translated_line
                else:
                    translated_results[key] = original_value
                    self.logger.warning("Validation failed for key '%s'.", key)
            else:
                translated_results[key] = original_value
                self.logger.warning("Key '%s' missing from response.", lookup_key)
        return translated_results

    def is_valid_translation(self, translation: str) -> bool:
        """Validates that translation string does not contain system error messages."""
        if not translation or not translation.strip():
            return False

        translation_lower = translation.lower()
        return not any(pattern in translation_lower for pattern in ERROR_PATTERNS)

    def save_progress(self, translated_data: dict[str, str], progress_file: Path) -> None:
        """Saves current translation progress dictionary to disk."""
        try:
            dump_json_file(progress_file, translated_data, indent=True)
            self.logger.info("Progress saved to %s", progress_file)
        except OSError as err:
            self.logger.error("Failed to save progress: %s", err)

    def load_progress(self, progress_file: Path) -> dict[str, str]:
        """Loads existing progress dictionary if checkpoint file exists."""
        if progress_file.exists():
            try:
                data = load_json_file(progress_file)
                if isinstance(data, dict):
                    self.logger.info("Loaded %d items from progress file.", len(data))
                    return data
            except (OSError, ValueError, TypeError, KeyError) as err:
                self.logger.error("Failed to load progress file: %s", err)
        return {}

    def _filter_source_data(self, original_data: dict[str, Any]) -> dict[str, Any]:
        """Filters out non-Japanese lines when source_language is Japanese."""
        if self.config.get("source_language") != "Japanese":
            return original_data

        search = JP_SOURCE_REGEX.search
        filtered_data = {
            key: value
            for key, value in original_data.items()
            if value and search(str(value) if not isinstance(value, str) else value)
        }
        filtered_out = len(original_data) - len(filtered_data)
        if filtered_out > 0:
            self.logger.info("Filtered out %d non-Japanese lines.", filtered_out)
        return filtered_data

    def generate_blueprint(
        self,
        original_data: dict[str, Any],
        summary_path: Path,
        auto_confirm: bool = False,
    ) -> str:
        """Generates or loads existing Translation Blueprint for character and tone consistency."""
        if not summary_path.exists():
            print("\n--- Generating Translation Blueprint ---")
            raw_texts = [
                str(v)
                for v in original_data.values()
                if v and len(str(v).strip()) > 3 and not str(v).strip().isdigit()
            ]
            summary_batches = self.chunker.process_all(raw_texts, self.summarize)
            summary = self.reduce_summaries(summary_batches)

            summary_path.parent.mkdir(parents=True, exist_ok=True)
            with open(summary_path, "w", encoding="utf-8") as file:
                file.write(summary)

            if not auto_confirm:
                input(f"Summary saved to '{summary_path}'. Review it and press Enter...")

        with open(summary_path, "r", encoding="utf-8") as f:
            return f.read()

    def _run_wave_with_executor(
        self,
        executor: ThreadPoolExecutor,
        giga_chunk: list[list[tuple[str, str]]],
        summary: str,
        translated_data: dict[str, str],
    ) -> None:
        """Runs a single parallel wave of translation batches using persistent thread pool."""
        futures = {
            executor.submit(self.translate_batch, (idx, chunk, summary)): idx
            for idx, chunk in enumerate(giga_chunk)
        }
        for future in as_completed(futures):
            res = future.result()
            if isinstance(res, dict):
                translated_data.update(res)

    def _translate_batches(
        self,
        items: list[tuple[str, str]],
        summary: str,
        translated_data: dict[str, str],
        progress_path: Path,
    ) -> None:
        """Executes batched translations in waves with intermediate autosaves."""
        batch_size = self.config["batch_size"]
        worker_count = self.config.get("max_workers", 4)
        all_batches = [items[i : i + batch_size] for i in range(0, len(items), batch_size)]
        giga_chunks = [
            all_batches[i : i + worker_count] for i in range(0, len(all_batches), worker_count)
        ]

        with ThreadPoolExecutor(max_workers=worker_count) as executor:
            for giga_index, giga_chunk in enumerate(giga_chunks):
                self._run_wave_with_executor(executor, giga_chunk, summary, translated_data)
                self.save_progress(translated_data, progress_path)
                self.logger.info("Wave %d/%d complete.", giga_index + 1, len(giga_chunks))

    def translate_json_file(  # pylint: disable=too-many-locals
        self,
        input_file: Path,
        output_file: Path,
        progress_file: Path | None = None,
        summary_file: Path | None = None,
        auto_confirm: bool = False,
    ) -> bool:
        """Translates an entire JSON file in batches with progressive checkpointing."""
        progress_name = self.config.get("progress_filename", "translation_progress.json")
        progress_path = progress_file or resolve_output_path(
            progress_name, default_subfolder="processed"
        )
        summary_name = self.config.get("summary_filename", "summary.txt")
        summary_path = summary_file or resolve_output_path(
            summary_name, default_subfolder="processed"
        )

        original_data = load_json_file(input_file)
        original_data = self._filter_source_data(original_data)

        translated_data = self.load_progress(progress_path)
        common_dict = self._load_common_translations()

        pre_translated_count = self._apply_pre_translations(
            original_data, translated_data, common_dict
        )
        if pre_translated_count > 0:
            self.logger.info("Applied %d common pre-translations.", pre_translated_count)
            self.save_progress(translated_data, progress_path)

        untranslated_items = [
            (str(key), str(val)) for key, val in original_data.items() if key not in translated_data
        ]

        if not untranslated_items:
            self.logger.info("All items are already translated.")
            self.save_progress(translated_data, output_file)
            return True

        summary = self.generate_blueprint(original_data, summary_path, auto_confirm=auto_confirm)

        self._translate_batches(untranslated_items, summary, translated_data, progress_path)

        self.save_progress(translated_data, output_file)
        self.logger.info("Translation complete. Saved to %s", output_file)
        return True


def process_translation(  # pylint: disable=too-many-arguments,too-many-positional-arguments
    config_file: str = "config.json",
    input_file: str | None = None,
    output_file: str | None = None,
    auto_confirm: bool = False,
    progress_file: Path | None = None,
    summary_file: Path | None = None,
) -> Path:
    """Processes JSON translation and returns the output path."""
    config = load_config(config_file, section="translation")
    in_file = resolve_input_path(
        input_file or config.get("input_filename", "ManualTransFile_cleaned.json"),
        default_subfolder="processed",
    )
    out_file = resolve_output_path(
        output_file or config.get("output_filename", "ManualTransFile_translated.json"),
        default_subfolder="processed",
    )

    with JSONTranslator(config_file=config_file) as translator:
        translator.translate_json_file(
            input_file=in_file,
            output_file=out_file,
            progress_file=progress_file,
            summary_file=summary_file,
            auto_confirm=auto_confirm,
        )
    return out_file
