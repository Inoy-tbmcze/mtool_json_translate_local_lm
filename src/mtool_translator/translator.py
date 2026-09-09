"""
Stage 2 Translation Engine for Game Localization JSON.
------------------------------------------------------
Translates Japanese game text into English using local or OpenAI-compatible LLMs.
Includes token-aware chunking, translation blueprint generation, and checkpoint recovery.
"""

from __future__ import annotations

import json
import logging
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests

from .config import load_config, resolve_input_path, resolve_output_path
from .utils import clean_japanese_text, parse_llm_json_response

logger = logging.getLogger("mtool_translator.translator")

SUMMARY_BATCH_SIZE = 1000
WORKER_COUNT = 2


class TokenAwareChunker:
    """Calculates context limits and splits text arrays into safe batches."""

    def __init__(
        self,
        n_ctx: int = 131072,
        reasoning_buffer: int = 8000,
        expansion_factor: float = 1.2,
        model_name: Optional[str] = None,
    ):
        self.n_ctx = n_ctx
        self.tokenizer = None

        if model_name:
            try:
                from transformers import AutoTokenizer  # pylint: disable=import-outside-toplevel

                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            except Exception as err:  # pylint: disable=broad-exception-caught
                logger.warning("Could not load tokenizer (%s). Using heuristic.", err)

        available_output_space = (n_ctx - reasoning_buffer) / (1 + expansion_factor)
        self.max_chunk_tokens = int(available_output_space * 0.9)

    def estimate_tokens(self, text: str) -> int:
        """Estimates or counts tokens for a string."""
        if self.tokenizer:
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        return int(len(text) / 3.0)

    def create_chunks(self, lst: List[str]) -> List[List[str]]:
        """Groups list items into chunks that do not exceed max_chunk_tokens."""
        chunks: List[List[str]] = []
        current_chunk: List[str] = []
        current_tokens = 0

        for item in lst:
            item_tokens = self.estimate_tokens(item)

            if item_tokens > self.max_chunk_tokens:
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = []
                    current_tokens = 0

                sub_items = item.split("\n")
                if len(sub_items) > 1:
                    sub_chunks = self.create_chunks(sub_items)
                    for sc in sub_chunks:
                        chunks.append(sc)
                else:
                    chunks.append([item])
                continue

            if current_tokens + item_tokens > self.max_chunk_tokens:
                chunks.append(current_chunk)
                current_chunk = [item]
                current_tokens = item_tokens
            else:
                current_chunk.append(item)
                current_tokens += item_tokens

        if current_chunk:
            chunks.append(current_chunk)

        return chunks

    def process_all(self, lst: List[str], process_fn) -> List[str]:
        """Splits text list into safe token chunks and processes each chunk."""
        chunks = self.create_chunks(lst)
        results = []

        for i, chunk in enumerate(chunks, 1):
            chunk_text = "\n".join(chunk)
            estimated_tokens = self.estimate_tokens(chunk_text)
            logger.info(
                "[Chunk %d/%d] Tokens: ~%d / Max: %d",
                i,
                len(chunks),
                estimated_tokens,
                self.max_chunk_tokens,
            )

            result = process_fn(chunk_text)
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

    def _init_config(self, config_file: str) -> Dict[str, Any]:
        config = load_config(config_file, section="translation")

        required_keys = ["api_endpoint", "api_key", "model", "source_language", "target_language"]
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

    def _get_api_headers_and_url(self) -> Tuple[Dict[str, str], str]:
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

    def _load_common_translations(self) -> Dict[str, str]:
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
            with open(dict_path, "r", encoding="utf-8") as f:
                data = json.load(f)
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
        except (json.JSONDecodeError, OSError) as err:
            self.logger.error("Failed to load common translations from '%s': %s", dict_file, err)

        return {}

    def _apply_pre_translations(
        self,
        original_data: Dict[str, Any],
        translated_data: Dict[str, str],
        common_dict: Dict[str, str],
    ) -> int:
        """Applies pre-translations from common dictionary for exact or trimmed matches."""
        if not common_dict:
            return 0

        pre_count = 0
        for key, val in original_data.items():
            if key in translated_data:
                continue

            target_str = str(val) if val is not None and str(val).strip() else str(key)

            # Exact match
            if target_str in common_dict:
                translated_data[key] = common_dict[target_str]
                pre_count += 1
                continue

            # Whitespace-trimmed match with whitespace preservation
            stripped = target_str.strip()
            if stripped and stripped in common_dict:
                leading = target_str[: len(target_str) - len(target_str.lstrip())]
                trailing = target_str[len(target_str.rstrip()) :]
                translated_data[key] = f"{leading}{common_dict[stripped]}{trailing}"
                pre_count += 1

        return pre_count

    def summarize(self, item: str) -> Any:
        """Generates a concise Translation Blueprint for character and tone consistency."""
        prompt = (
            "Analyze the raw text and create a concise Translation Blueprint.\n\n"
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

        headers, api_url = self._get_api_headers_and_url()
        data = {
            "model": self.config["model"],
            "messages": [{"role": "system", "content": prompt}, {"role": "user", "content": item}],
            "temperature": 0.0,
            "max_tokens": 2048,
        }

        try:
            resp = requests.post(
                api_url, headers=headers, json=data, timeout=self.config["request_timeout"]
            )
            if resp.status_code == 200:
                result_stripped = resp.json()["choices"][0]["message"]["content"].strip()
                self.logger.info("Section summary generated successfully.")
                return result_stripped
        except (requests.RequestException, KeyError, json.JSONDecodeError) as err:
            self.logger.error("Summarize request failed: %s", err)
        return None

    def summarize_summaries(self, item: str) -> Any:
        """Synthesizes multiple summary parts into a single blueprint."""
        prompt = (
            "Synthesize multi-part translation notes into one short Translation Blueprint.\n\n"
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

        headers, api_url = self._get_api_headers_and_url()
        data = {
            "model": self.config["model"],
            "messages": [{"role": "system", "content": prompt}, {"role": "user", "content": item}],
            "temperature": 0.0,
            "max_tokens": 2048,
        }

        try:
            resp = requests.post(
                api_url, headers=headers, json=data, timeout=self.config["request_timeout"]
            )
            if resp.status_code == 200:
                result_stripped = resp.json()["choices"][0]["message"]["content"].strip()
                self.logger.info("Reduced summary generated successfully.")
                return result_stripped
        except (requests.RequestException, KeyError, json.JSONDecodeError) as err:
            self.logger.error("Summarize summaries request failed: %s", err)
        return None

    def reduce_summaries(self, lst: List[str], max_depth: int = 5) -> str:
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

    def translate_batch(self, item: Tuple[int, List[Tuple[str, str]], str]) -> Dict[str, str]:
        """Translates a single batch and returns key-translation pairs."""
        _index, texts, summary = item
        fallback_results = dict(texts)

        if not texts:
            return {}

        input_dict = {str(i + 1): value for i, (_k, value) in enumerate(texts)}
        cleaned_dict = {key: clean_japanese_text(val) for key, val in input_dict.items()}
        json_batch = json.dumps(cleaned_dict, ensure_ascii=False)

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
        headers: Dict[str, str],
        data: Dict[str, Any],
        texts: List[Tuple[str, str]],
        fallback_results: Dict[str, str],
    ) -> Dict[str, str]:
        for attempt in range(self.config["max_retries"]):
            try:
                resp = requests.post(
                    api_url, headers=headers, json=data, timeout=self.config["request_timeout"]
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
            except (requests.RequestException, ValueError, json.JSONDecodeError) as err:
                self.logger.error("Attempt %d error during translation: %s", attempt + 1, err)

            if attempt < self.config["max_retries"] - 1:
                time.sleep(self.config["retry_delay"])

        self.logger.error("Max retries exhausted for batch. Returning fallbacks.")
        return fallback_results

    def _map_translation_response(
        self, texts: List[Tuple[str, str]], translated_json: Dict[str, Any]
    ) -> Dict[str, str]:
        translated_results: Dict[str, str] = {}
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

        error_patterns = [
            "translation failed",
            "unable to translate",
            "error occurred",
            "something went wrong",
            "as an ai",
            "i cannot translate",
            "translator note:",
        ]

        translation_lower = translation.lower()
        return not any(pattern in translation_lower for pattern in error_patterns)

    def save_progress(self, translated_data: Dict[str, str], progress_file: Path) -> None:
        """Saves current translation progress dictionary to disk."""
        try:
            progress_file.parent.mkdir(parents=True, exist_ok=True)
            with open(progress_file, "w", encoding="utf-8") as f:
                json.dump(translated_data, f, ensure_ascii=False, indent=2)
            self.logger.info("Progress saved to %s", progress_file)
        except OSError as err:
            self.logger.error("Failed to save progress: %s", err)

    def load_progress(self, progress_file: Path) -> Dict[str, str]:
        """Loads existing progress dictionary if checkpoint file exists."""
        if progress_file.exists():
            try:
                with open(progress_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.logger.info("Loaded %d items from progress file.", len(data))
                return data
            except (json.JSONDecodeError, OSError) as err:
                self.logger.error("Failed to load progress file: %s", err)
        return {}

    def _filter_source_data(self, original_data: Dict[str, Any]) -> Dict[str, Any]:
        """Filters out non-Japanese lines when source_language is Japanese."""
        if self.config.get("source_language") != "Japanese":
            return original_data

        jp_regex = re.compile(r"[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]")
        filtered_data = {k: v for k, v in original_data.items() if jp_regex.search(str(v))}
        excluded = len(original_data) - len(filtered_data)
        print(f"Kept {len(filtered_data)} Japanese lines. Excluded {excluded} non-Japanese lines.")
        return filtered_data

    def _ensure_summary(self, summary_path: Path, data: Dict[str, Any], auto_confirm: bool) -> str:
        """Loads or generates translation summary blueprint."""
        if not summary_path.exists():
            raw_texts = [str(v) for v in data.values()]
            summary_batches = self.chunker.process_all(raw_texts, self.summarize)
            summary = self.reduce_summaries(summary_batches)

            summary_path.parent.mkdir(parents=True, exist_ok=True)
            with open(summary_path, "w", encoding="utf-8") as file:
                file.write(summary)

            if not auto_confirm:
                input(f"Summary saved to '{summary_path}'. Review it and press Enter...")

        with open(summary_path, "r", encoding="utf-8") as f:
            return f.read()

    def _run_wave(
        self, giga_chunk: List[List[Tuple[str, str]]], summary: str, translated_data: Dict[str, str]
    ) -> None:
        """Runs a single parallel wave of translation batches."""
        with ThreadPoolExecutor(max_workers=WORKER_COUNT) as executor:
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
        items: List[Tuple[str, str]],
        summary: str,
        translated_data: Dict[str, str],
        progress_path: Path,
    ) -> None:
        """Executes batched translations in waves with intermediate autosaves."""
        batch_size = self.config["batch_size"]
        all_batches = [items[i : i + batch_size] for i in range(0, len(items), batch_size)]
        giga_chunks = [
            all_batches[i : i + WORKER_COUNT] for i in range(0, len(all_batches), WORKER_COUNT)
        ]

        for giga_index, giga_chunk in enumerate(giga_chunks):
            self._run_wave(giga_chunk, summary, translated_data)
            self.save_progress(translated_data, progress_path)
            self.logger.info("Wave %d/%d complete.", giga_index + 1, len(giga_chunks))

    def translate_json_file(  # pylint: disable=too-many-locals
        self,
        input_file: Path,
        output_file: Path,
        progress_file: Optional[Path] = None,
        summary_file: Optional[Path] = None,
        auto_confirm: bool = False,
    ) -> bool:
        """Translates an entire JSON file in batches with progressive checkpointing."""
        progress_path = progress_file or resolve_output_path(
            "translation_progress.json", default_subfolder="processed"
        )
        summary_path = summary_file or resolve_output_path(
            "summary.txt", default_subfolder="processed"
        )

        with open(input_file, "r", encoding="utf-8") as f:
            original_data = json.load(f)

        original_data = self._filter_source_data(original_data)

        translated_data = self.load_progress(progress_path)
        common_dict = self._load_common_translations()
        pre_count = self._apply_pre_translations(original_data, translated_data, common_dict)
        if pre_count > 0:
            self.logger.info("Pre-translated %d lines using common dictionary.", pre_count)
            self.save_progress(translated_data, progress_path)

        items = [
            (k, v)
            for k, v in original_data.items()
            if k not in translated_data and v and str(v).strip()
        ]

        self.logger.info("Total lines: %d | Pending: %d", len(original_data), len(items))

        if items:
            summary_data = dict(items)
            summary = self._ensure_summary(summary_path, summary_data, auto_confirm)
            self._translate_batches(items, summary, translated_data, progress_path)
        else:
            self.logger.info(
                "All lines resolved via pre-translation or checkpoint. No LLM calls needed."
            )

        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(translated_data, f, ensure_ascii=False, indent=2)

        self.logger.info("Translation completed successfully: %s", output_file)

        if progress_path.exists():
            try:
                progress_path.unlink()
            except OSError:
                pass

        return True


def _setup_translation_logger(log_path: Path) -> None:
    """Configures file and console logging handlers for translation."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    if not logger.handlers:
        file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
        stream_handler = logging.StreamHandler()
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        file_handler.setFormatter(formatter)
        stream_handler.setFormatter(formatter)
        logger.setLevel(logging.INFO)
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)


def _resolve_translation_paths(
    translator: JSONTranslator, input_file: Optional[str], output_file: Optional[str]
) -> Tuple[Path, Path]:
    """Resolves input and output file paths for translation."""
    target_input = input_file or translator.config.get(
        "input_filename", "ManualTransFile_cleaned.json"
    )
    resolved_in = resolve_input_path(target_input, default_subfolder="processed")
    if not resolved_in.exists():
        resolved_in = resolve_input_path(target_input, default_subfolder="raw")

    if output_file:
        resolved_out = Path(output_file)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        resolved_out = resolve_output_path(
            f"translated_{timestamp}.json", default_subfolder="processed"
        )
    return resolved_in, resolved_out


def _check_resume_prompts(progress_file: Path, summary_file: Path, auto_confirm: bool) -> None:
    """Interactively prompts user to resume or discard prior checkpoints."""
    if auto_confirm:
        return
    if progress_file.exists():
        response = input("Found existing progress file. Resume? (y/n): ")
        if response.lower() not in ["y", "yes"]:
            progress_file.unlink()

    if summary_file.exists():
        response = input("Found existing summary file. Use it? (y/n): ")
        if response.lower() not in ["y", "yes"]:
            summary_file.unlink()


def process_translation(
    config_file: str = "config.json",
    input_file: Optional[str] = None,
    output_file: Optional[str] = None,
    auto_confirm: bool = False,
) -> Path:
    """Entrypoint function for translation stage."""
    log_path = resolve_output_path("translation.log", default_subfolder="processed")
    _setup_translation_logger(log_path)

    translator = JSONTranslator(config_file)
    resolved_input, resolved_output = _resolve_translation_paths(
        translator, input_file, output_file
    )

    if not resolved_input.exists():
        print(f"Error: Translation input file '{resolved_input.name}' not found.")
        return resolved_input

    progress_file = resolve_output_path("translation_progress.json", default_subfolder="processed")
    summary_file = resolve_output_path("summary.txt", default_subfolder="processed")

    _check_resume_prompts(progress_file, summary_file, auto_confirm)

    print(f"Starting translation processing: {resolved_input} -> {resolved_output}...")
    translator.translate_json_file(
        resolved_input, resolved_output, progress_file, summary_file, auto_confirm=auto_confirm
    )
    return resolved_output
