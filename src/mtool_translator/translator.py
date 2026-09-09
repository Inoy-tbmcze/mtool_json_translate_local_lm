"""
Stage 2 Translation Engine for Game Localization JSON.
------------------------------------------------------
Translates Japanese game text into English using local or OpenAI-compatible LLMs.
Includes token-aware chunking, translation blueprint generation, and checkpoint recovery.
"""

import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional
import requests

from .config import load_config, resolve_input_path, resolve_output_path, get_project_root
from .utils import clean_japanese_text, parse_llm_json_response

logger = logging.getLogger("mtool_translator.translator")

SUMMARY_BATCH_SIZE = 1000
WORKER_COUNT = 2


class TokenAwareChunker:
    def __init__(
        self,
        n_ctx: int = 131072,
        reasoning_buffer: int = 8000,
        expansion_factor: float = 1.2,
        model_name: Optional[str] = None
    ):
        """
        :param n_ctx: Total server context window (e.g., 65536 or 131072)
        :param reasoning_buffer: Max tokens reserved for model reasoning
        :param expansion_factor: Expected length growth during translation (1.1-1.3)
        :param model_name: Optional HF model name for exact tokenizer
        """
        self.n_ctx = n_ctx
        self.tokenizer = None

        if model_name:
            try:
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            except Exception as e:
                logger.warning(f"Could not load tokenizer ({e}). Falling back to heuristic estimation.")

        available_output_space = (n_ctx - reasoning_buffer) / (1 + expansion_factor)
        self.max_chunk_tokens = int(available_output_space * 0.9)

    def estimate_tokens(self, text: str) -> int:
        """Estimates or counts tokens for a string."""
        if self.tokenizer:
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        return int(len(text) / 3.0)

    def create_chunks(self, lst: List[str]) -> List[List[str]]:
        """Groups list items into chunks that do not exceed max_chunk_tokens."""
        chunks = []
        current_chunk = []
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
            logger.info(f"[Chunk {i}/{len(chunks)}] Tokens: ~{estimated_tokens} / Max: {self.max_chunk_tokens}")

            result = process_fn(chunk_text)
            if result:
                results.append(result)
            else:
                logger.warning(f"Chunk {i} returned None.")

        return results


class JSONTranslator:
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

        return config

    def _get_api_headers_and_url(self) -> Tuple[Dict[str, str], str]:
        if self.config.get("api_type", "openai") == "google":
            headers = {"Content-Type": "application/json"}
            api_url = f"{self.config['api_endpoint']}?key={self.config['api_key']}"
        else:
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.config['api_key']}"
            }
            api_url = self.config["api_endpoint"]
        return headers, api_url

    def summarize(self, item: str) -> Any:
        prompt = """Analyze the raw text and create a concise Translation Blueprint.

Output MUST follow this exact structure:

WORLD & TONE: [Maximum 15 words. Describe setting and tone (e.g., casual, formal, gritty)]
STORY: [Maximum 15 words. Describe core story]
MAIN CHARACTERS (Protagonist and top 2 characters):
- [Original Name] -> [English Name]: [Role], [Gender], [Speaking style]
- [Original Name] -> [English Name]: [Role], [Gender], [Speaking style]
- [Original Name] -> [English Name]: [Role], [Gender], [Speaking style]

CHARACTER NAME MAPPINGS (Top 10):
- [Original Name] -> [English Name] ([Gender])

Do not output markdown code blocks. Do not add explanations."""

        headers, api_url = self._get_api_headers_and_url()
        data = {
            "model": self.config["model"],
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": item}
            ],
            "temperature": 0.0,
            "max_tokens": 2048,
            "safety_settings": [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
            ]
        }

        try:
            response = requests.post(api_url, headers=headers, json=data, timeout=self.config["request_timeout"])
            if response.status_code == 200:
                result_stripped = response.json()["choices"][0]["message"]["content"].strip()
                self.logger.info("Section summary generated successfully.")
                return result_stripped
        except Exception as e:
            self.logger.error(f"Summarize request failed: {e}")
        return None

    def summarize_summaries(self, item: str) -> Any:
        prompt = """Synthesize multi-part translation notes into one short Translation Blueprint.

Output MUST follow this exact structure:

WORLD & TONE: [Maximum 15 words. Describe setting and tone (e.g., casual, formal, gritty)]
STORY: [Maximum 15 words. Describe core story]
MAIN CHARACTERS (Protagonist and top 2 characters):
- [Original Name] -> [English Name]: [Role], [Gender], [Speaking style]
- [Original Name] -> [English Name]: [Role], [Gender], [Speaking style]
- [Original Name] -> [English Name]: [Role], [Gender], [Speaking style]

CHARACTER NAME MAPPINGS (Top 10):
- [Original Name] -> [English Name] ([Gender])

Do not output markdown code blocks. Do not add explanations."""

        headers, api_url = self._get_api_headers_and_url()
        data = {
            "model": self.config["model"],
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": item}
            ],
            "temperature": 0.0,
            "max_tokens": 2048,
            "safety_settings": [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
            ]
        }

        try:
            response = requests.post(api_url, headers=headers, json=data, timeout=self.config["request_timeout"])
            if response.status_code == 200:
                result_stripped = response.json()["choices"][0]["message"]["content"].strip()
                self.logger.info("Reduced summary generated successfully.")
                return result_stripped
        except Exception as e:
            self.logger.error(f"Summarize summaries request failed: {e}")
        return None

    def reduce_summaries(self, lst: List[str], max_depth: int = 5) -> str:
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
        """Translates a single batch and strictly returns a dictionary of key:translation pairs."""
        index, texts, summary = item
        fallback_results = {key: value for key, value in texts}

        if not texts:
            return {}

        input_dict = {str(i + 1): value for i, (key, value) in enumerate(texts)}
        cleaned_dict = {key: clean_japanese_text(val) for key, val in input_dict.items()}
        json_batch = json.dumps(cleaned_dict, ensure_ascii=False)

        source_lang = self.config.get("source_language", "Japanese")
        target_lang = self.config.get("target_language", "English")

        prompt = f"""You are a translation engine.
Translate all JSON string values from {source_lang} to {target_lang}.

TRANSLATION BLUEPRINT:
{summary}

RULES:
1. Translate JSON values only. Keep keys unchanged.
2. Follow character names, gender, and tone from the TRANSLATION BLUEPRINT.
3. Keep technical terms, code, and control characters (\\n, \\t) unchanged.
4. Output raw JSON only. Do not use Markdown code blocks. Do not add explanations."""

        if self.print_summary:
            self.print_summary = False
            self.logger.info(
                f"\nFinal Prompt Sample:\n----------------------------------\n{prompt}\n----------------------------------")

        headers, api_url = self._get_api_headers_and_url()
        data = {
            "model": self.config["model"],
            "messages": [
                {"role": "system", "content": prompt},
                {"role": "user", "content": json_batch}
            ],
            "temperature": 0.2,
            "max_tokens": 65536,
            "safety_settings": [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
            ]
        }

        for attempt in range(self.config["max_retries"]):
            try:
                response = requests.post(
                    api_url,
                    headers=headers,
                    json=data,
                    timeout=self.config["request_timeout"]
                )

                if response.status_code == 200:
                    result = response.json()
                    if "choices" in result and len(result["choices"]) > 0:
                        translation_text = result["choices"][0]["message"]["content"].strip()
                        translated_json = parse_llm_json_response(translation_text)
                        translated_results = {}

                        for i, (key, original_value) in enumerate(texts):
                            lookup_key = str(i + 1)
                            if lookup_key in translated_json and translated_json[lookup_key] is not None:
                                translated_line = str(translated_json[lookup_key]).strip()

                                if self.is_valid_translation(original_value, translated_line):
                                    translated_results[key] = translated_line
                                else:
                                    translated_results[key] = original_value
                                    self.logger.warning(f"Validation failed for key '{key}'. Reverting to original.")
                            else:
                                translated_results[key] = original_value
                                self.logger.warning(
                                    f"Key '{lookup_key}' missing from JSON response. Reverting to original.")

                        return translated_results

                elif response.status_code == 429:
                    self.logger.warning(f"Rate limit hit. Waiting {self.config['retry_delay']}s...")
                    time.sleep(self.config["retry_delay"])

                elif response.status_code == 401:
                    self.logger.error("API key unauthorized. Aborting batch.")
                    return fallback_results

                else:
                    self.logger.error(f"API Error ({response.status_code}): {response.text}")

            except Exception as e:
                self.logger.error(f"Attempt {attempt + 1} exception during batch translation: {e}")

            if attempt < self.config["max_retries"] - 1:
                time.sleep(self.config["retry_delay"])

        self.logger.error("Max retries exhausted for batch. Returning fallbacks.")
        return fallback_results

    def is_valid_translation(self, original: str, translation: str) -> bool:
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
        for pattern in error_patterns:
            if pattern in translation_lower:
                return False

        return True

    def save_progress(self, translated_data: Dict[str, str], progress_file: Path):
        try:
            progress_file.parent.mkdir(parents=True, exist_ok=True)
            with open(progress_file, "w", encoding="utf-8") as f:
                json.dump(translated_data, f, ensure_ascii=False, indent=2)
            self.logger.info(f"Progress saved to {progress_file}")
        except Exception as e:
            self.logger.error(f"Failed to save progress: {e}")

    def load_progress(self, progress_file: Path) -> Dict[str, str]:
        if progress_file.exists():
            try:
                with open(progress_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.logger.info(f"Loaded {len(data)} items from progress file.")
                return data
            except Exception as e:
                self.logger.error(f"Failed to load progress file: {e}")
        return {}

    def translate_json_file(
        self,
        input_file: Path,
        output_file: Path,
        progress_file: Optional[Path] = None,
        summary_file: Optional[Path] = None,
        auto_confirm: bool = False
    ) -> bool:
        root = get_project_root()
        progress_file = progress_file or resolve_output_path("translation_progress.json", default_subfolder="processed")
        summary_file = summary_file or resolve_output_path("summary.txt", default_subfolder="processed")

        with open(input_file, "r", encoding="utf-8") as f:
            original_data = json.load(f)

        if self.config.get("source_language") == "Japanese":
            japanese_regex = re.compile(r"[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]")
            filtered_data = {k: v for k, v in original_data.items() if japanese_regex.search(str(v))}
            print(
                f"Kept {len(filtered_data)} Japanese lines. Excluded {len(original_data) - len(filtered_data)} non-Japanese lines.")
            original_data = filtered_data

        if not summary_file.exists():
            raw_texts = [str(v) for v in original_data.values()]
            summary_batches = self.chunker.process_all(raw_texts, self.summarize)
            summary = self.reduce_summaries(summary_batches)

            summary_file.parent.mkdir(parents=True, exist_ok=True)
            with open(summary_file, "w", encoding="utf-8") as file:
                file.write(summary)

            if not auto_confirm:
                input(f"Summary saved to '{summary_file}'. Review it and press Enter to continue...")

        with open(summary_file, "r", encoding="utf-8") as f:
            summary = f.read()

        translated_data = self.load_progress(progress_file)

        items_to_translate = [
            (k, v) for k, v in original_data.items() if k not in translated_data and v and str(v).strip()
        ]

        self.logger.info(f"Total lines: {len(original_data)} | Pending: {len(items_to_translate)}")

        batch_size = self.config["batch_size"]
        all_batches = [items_to_translate[i:i + batch_size] for i in range(0, len(items_to_translate), batch_size)]
        giga_chunks = [all_batches[i:i + WORKER_COUNT] for i in range(0, len(all_batches), WORKER_COUNT)]

        for giga_index, giga_chunk in enumerate(giga_chunks):
            results = []
            with ThreadPoolExecutor(max_workers=WORKER_COUNT) as executor:
                futures = {
                    executor.submit(self.translate_batch, (idx, chunk, summary)): idx
                    for idx, chunk in enumerate(giga_chunk)
                }

                for future in as_completed(futures):
                    res = future.result()
                    if isinstance(res, dict):
                        results.append(res)

            for result in results:
                translated_data.update(result)

            self.save_progress(translated_data, progress_file)
            self.logger.info(f"Wave {giga_index + 1}/{len(giga_chunks)} complete.")

        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(translated_data, f, ensure_ascii=False, indent=2)

        self.logger.info(f"Translation completed successfully: {output_file}")

        if progress_file.exists():
            try:
                progress_file.unlink()
            except Exception:
                pass

        return True


def process_translation(
    config_file: str = "config.json",
    input_file: Optional[str] = None,
    output_file: Optional[str] = None,
    auto_confirm: bool = False
) -> Path:
    """Entrypoint function for translation stage."""
    # Configure logging
    log_path = resolve_output_path("translation.log", default_subfolder="processed")
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

    translator = JSONTranslator(config_file)

    target_input = input_file or translator.config.get("input_filename", "ManualTransFile_cleaned.json")
    resolved_input = resolve_input_path(target_input, default_subfolder="processed")

    if not resolved_input.exists():
        # Check raw subfolder as fallback
        resolved_input = resolve_input_path(target_input, default_subfolder="raw")

    if not resolved_input.exists():
        print(f"Error: Translation input file '{target_input}' not found.")
        return Path(target_input)

    if output_file:
        resolved_output = Path(output_file)
    else:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        resolved_output = resolve_output_path(f"translated_{timestamp}.json", default_subfolder="processed")

    progress_file = resolve_output_path("translation_progress.json", default_subfolder="processed")
    summary_file = resolve_output_path("summary.txt", default_subfolder="processed")

    if not auto_confirm:
        if progress_file.exists():
            response = input("Found existing progress file. Resume? (y/n): ")
            if response.lower() not in ["y", "yes"]:
                progress_file.unlink()

        if summary_file.exists():
            response = input("Found existing summary file. Use it? (y/n): ")
            if response.lower() not in ["y", "yes"]:
                summary_file.unlink()

    print(f"Starting translation processing: {resolved_input} -> {resolved_output}...")
    translator.translate_json_file(
        resolved_input,
        resolved_output,
        progress_file,
        summary_file,
        auto_confirm=auto_confirm
    )
    return resolved_output
