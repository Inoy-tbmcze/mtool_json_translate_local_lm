import json
import re

import requests
import time
import os
from typing import Dict, Any
from json_repair import repair_json
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed

SUMMARY_BATCH_SIZE = 1000
WORKER_COUNT = 2


class TokenAwareChunker:
    def __init__(self, n_ctx=131072, reasoning_buffer=8000, expansion_factor=1.2, model_name=None):
        """
        :param n_ctx: Total server context window (e.g. 65536 or 131072)
        :param reasoning_buffer: Max tokens reserved for model reasoning
        :param expansion_factor: Expected length growth during translation (1.1-1.3)
        :param model_name: Optional HF model name for exact tokenizer (e.g., 'google/gemma-2-9b')
        """
        self.n_ctx = n_ctx
        self.tokenizer = None

        if model_name:
            try:
                from transformers import AutoTokenizer
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            except Exception as e:
                print(f"Warning: Could not load tokenizer ({e}). Falling back to heuristic estimation.")

        # Calculate max input tokens per chunk
        available_output_space = (n_ctx - reasoning_buffer) / (1 + expansion_factor)
        # Apply a 10% safety margin
        self.max_chunk_tokens = int(available_output_space * 0.9)

    def estimate_tokens(self, text: str) -> int:
        """Estimates or counts tokens for a string."""
        if self.tokenizer:
            return len(self.tokenizer.encode(text, add_special_tokens=False))
        # Heuristic fallback: ~3.5 chars per token for English/code, ~1.5-2 chars for CJK
        return int(len(text) / 3.0)

    def create_chunks(self, lst: list[str]) -> list[list[str]]:
        """
        Groups list items into chunks such that no chunk exceeds self.max_chunk_tokens.
        """
        chunks = []
        current_chunk = []
        current_tokens = 0

        for item in lst:
            item_tokens = self.estimate_tokens(item)

            # If a single item is larger than the entire allowed chunk size
            if item_tokens > self.max_chunk_tokens:
                # Flush existing chunk
                if current_chunk:
                    chunks.append(current_chunk)
                    current_chunk = []
                    current_tokens = 0

                # Force split the oversized single item by line/paragraph
                sub_items = item.split("\n")
                if len(sub_items) > 1:
                    sub_chunks = self.create_chunks(sub_items)
                    for sc in sub_chunks:
                        chunks.append(sc)
                else:
                    # Hard fallback for an unsplitable huge block
                    chunks.append([item])
                continue

            # If adding item exceeds the limit, seal current chunk and start new one
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

    def process_all(self, lst: list[str], process_fn) -> list[str]:
        """
        Splits text list into safe token chunks and processes each chunk.
        :param lst: List of text blocks/strings to translate/summarize.
        :param process_fn: Function that sends text to LLM (takes `\n`.join(chunk)).
        """
        chunks = self.create_chunks(lst)
        results = []

        for i, chunk in enumerate(chunks, 1):
            chunk_text = "\n".join(chunk)
            estimated_tokens = self.estimate_tokens(chunk_text)
            print(f"[Chunk {i}/{len(chunks)}] Tokens: ~{estimated_tokens} / Max: {self.max_chunk_tokens}")

            result = process_fn(chunk_text)
            if result:
                results.append(result)
            else:
                print(f"[Warning] Chunk {i} returned None. Consider reducing max_chunk_tokens further.")

        return results


def repair_and_load_json_string(json_string):
    # 1. Strip whitespace
    json_string = json_string.strip()

    # 2. Fix the quote mismatch (e.g., "text'} -> "text"})
    # This targets single quotes right before a closing object brace
    if json_string.endswith("'}"):
        json_string = json_string[:-2] + '"}'

    # This targets single quotes right before a closing array or next element comma
    json_string = re.sub(r"'\s*\}", '"}', json_string)
    json_string = re.sub(r"'\s*,", '",', json_string)

    # 3. Handle previous anomalies (smart quotes, nested newlines)
    json_string = json_string.replace('“', '"').replace('”', '"')

    try:
        return json.loads(json_string)
    except json.JSONDecodeError as e:
        # Fallback to the line-by-line repair logic if there are deeper issues
        lines = json_string.splitlines()
        line_idx = e.lineno - 1

        if "Unterminated string" in e.msg and line_idx < len(lines):
            problem_line = lines[line_idx].rstrip()
            if problem_line.endswith('}'):
                lines[line_idx] = problem_line[:-1].rstrip().rstrip("'") + '"}'
                return json.loads("\n".join(lines))
        raise e


def repair_and_load_json_string2(json_string):
    # 1. Normalize all variations of smart quotes immediately
    json_string = json_string.replace('“', '"').replace('”', '"')

    # 2. Extract the actual payload text inside the JSON boundaries.
    # This regex matches the pattern {"any_key": "any_value"} even across multiple lines
    match = re.match(r'^\{\s*"[^"]+"\s*:\s*"(.*)"\s*\}\s*$', json_string.strip(), flags=re.DOTALL)

    if match:
        payload = match.group(1)

        # --- Clean the payload text ---
        # A. If smart quote normalization created double double-quotes at the ends (""text""), fix them
        if payload.startswith('"'): payload = payload[1:]
        if payload.endswith('"'): payload = payload[:-1]

        # B. Clean up unescaped inner quotes so they don't break the string boundary
        # Safely escapes any loose double quotes inside the text
        payload = re.sub(r'(?<!\\)"', r'\"', payload)

        # C. Ensure any literal single quotes at the very end are stripped if they were meant to close it
        payload = payload.rstrip("'")

        # D. Normalize line breaks to proper JSON escaped newlines
        payload = payload.replace('\n', '\\n')

        # Reconstruct a perfectly clean, valid JSON object string
        # We find the original key dynamically to preserve it
        key_match = re.search(r'^\{\s*("[^"]+")', json_string)
        key = key_match.group(1) if key_match else '"1"'

        json_string = f'{{{key}: "{payload}"}}'

    # 3. Final Parse Attempt
    try:
        return json.loads(json_string)
    except json.JSONDecodeError:
        # Emergency Fallback: If regex didn't match the wrapper structure, force string-level replacements
        cleaned = json_string.strip()
        cleaned = re.sub(r'^\{\s*"1"\s*:\s*"\s*"?', '{"1": "', cleaned)
        cleaned = re.sub(r'"?\s*\'?\}\s*$', '"}', cleaned)
        return json.loads(cleaned)


def parse_llm_json_response(response_text: str) -> dict:
    if not response_text or not response_text.strip():
        raise ValueError("Response text is empty or whitespace.")

    # repair_json automatically fixes unescaped inner quotes, raw linebreaks,
    # single quotes, and markdown backticks.
    repaired_json_str = repair_json(response_text)

    parsed = json.loads(repaired_json_str)
    if isinstance(parsed, dict):
        return parsed

    raise ValueError("Parsed output is not a dictionary.")


class JSONTranslator:
    def __init__(self, config_file: str = "translate_config.json"):
        self.chunker = TokenAwareChunker()
        self.config = self.load_config(config_file)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler('translation.log', encoding='utf-8'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.print_summary = True

    def load_config(self, config_file: str) -> Dict[str, Any]:
        with open(config_file, 'r', encoding='utf-8') as f:
            config = json.load(f)

        required_keys = ['api_endpoint', 'api_key', 'model', 'source_language', 'target_language']
        for key in required_keys:
            if key not in config:
                raise ValueError(f"load_config error: {key}")

        config.setdefault('max_retries', 3)
        config.setdefault('retry_delay', 5)
        config.setdefault('request_timeout', 60)
        config.setdefault('batch_size', 50)
        config.setdefault('save_interval', 100)
        config.setdefault('api_type', 'openai')

        return config

    def summarize(self, item) -> Any:
        prompt = ("""
        You are an expert translation strategist. Analyze the following raw text and create a highly condensed, dense "Translation Blueprint" for another AI to use as a style guide. 
        
        Keep your step-by-step thinking brief and concise before generating the response.
        
        Extract only the core information required to ensure flawless, consistent translation across batches. Structure your output exactly like this:

        1. CORE CONTEXT & TONE: (e.g., "A dark sci-fi story. Use an informal, gritty tone. Characters use military jargon.")
        2. CHARACTER/ENTITY PROFILES: (List key characters, their name in original language, genders, ages and relationships so pronouns and verb inflections match across batches.)
        3. Respond ONLY with text that will be passed to another AI. Do not include markdown formatting or explanations.
        """)

        data = {
            'model': self.config['model'],
            'messages': [
                {"role": "system", "content": prompt},
                {"role": "user", "content": item}
            ],
            'max_tokens': 16384,
            'safety_settings': [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_CIVIC_INTEGRITY", "threshold": "BLOCK_NONE"}
            ]
        }

        if self.config.get('api_type', 'openai') == 'google':
            headers = {
                'Content-Type': 'application/json'
            }
            api_url = f"{self.config['api_endpoint']}?key={self.config['api_key']}"
        else:
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {self.config["api_key"]}'
            }
            api_url = self.config['api_endpoint']

        response = requests.post(
            api_url,
            headers=headers,
            json=data,
            timeout=self.config['request_timeout']
        )

        if response.status_code == 200:
            result = response.json()
            result_stripped = result['choices'][0]['message']['content'].strip()
            self.logger.info(result_stripped)
            return result_stripped
        return None

    def summarize_summaries(self, item) -> Any:
        prompt = f"""You are a Translation Strategy Specialist. Synthesize the multi-part translation notes below into a single, highly condensed "Translation Blueprint" for a downstream translation AI.

        PROCESSING DIRECTIVES:
        1. CONTEXT & TONE: Merge all setting, tone, narrative style, and register notes into a single cohesive summary.
        2. CHARACTER MERGING: Combine recurring characters across summaries into unified profiles. Prioritize main characters and protagonists.
        3. ENTITY FILTERING: Include only key named characters and essential entities required for grammar/pronoun consistency. Omit background extras, generic monsters, and unnamed minor characters.

        OUTPUT FORMAT REQUIREMENTS:
        Output ONLY the final blueprint using the exact structure below. Do NOT include markdown code blocks (```), introductory text, explanations, or reasoning.

        CORE CONTEXT & TONE:
        [Condensed genre, setting, tone, narrative voice, and register rules]

        CHARACTER/ENTITY PROFILES:
        - [Original Name] / [Translated Name]: [Gender/Pronouns], [Age/Role], [Key Relationships], [Linguistic/Translation notes]
        """

        data = {
            'model': self.config['model'],
            'messages': [
                {"role": "system", "content": prompt},
                {"role": "user", "content": item}
            ],
            'max_tokens': 16384,
            'safety_settings': [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_CIVIC_INTEGRITY", "threshold": "BLOCK_NONE"}
            ]
        }

        if self.config.get('api_type', 'openai') == 'google':
            headers = {
                'Content-Type': 'application/json'
            }
            api_url = f"{self.config['api_endpoint']}?key={self.config['api_key']}"
        else:
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {self.config["api_key"]}'
            }
            api_url = self.config['api_endpoint']

        response = requests.post(
            api_url,
            headers=headers,
            json=data,
            timeout=self.config['request_timeout']
        )

        if response.status_code == 200:
            result = response.json()
            result_stripped = result['choices'][0]['message']['content'].strip()
            self.logger.info(result_stripped)
            return result_stripped
        return None

    def split_and_summarize(self, lst):
        # Initialize chunker for 64k context
        chunker = TokenAwareChunker(
            n_ctx=65536,
            reasoning_buffer=8000,  # Headroom for model thinking
            expansion_factor=1.2  # Translation length expansion factor
        )

        # Your existing LLM call wrapper
        def my_llm_call(text_block):
            return self.summarize(text_block)

        # Process safely in pre-calculated batches
        final_results = chunker.process_all(lst, my_llm_call)
        return final_results

    def reduce_summaries(self, lst: list[str], max_depth: int = 5) -> str:
        """
        Hierarchically combines a list of summary strings into a single final summary
        while guaranteeing that token context limits are never exceeded.
        """
        if not lst:
            return ""

        # If there is only one item, check if it needs a final pass or can be returned directly
        if len(lst) == 1:
            return lst[0]

        current_items = lst
        depth = 0

        # Loop hierarchically until we reduce down to 1 summary or hit max depth
        while len(current_items) > 1 and depth < max_depth:
            depth += 1
            print(f"[Hierarchical Pass {depth}] Combining {len(current_items)} summary items...")

            # 1. Group current summaries into token-safe chunks
            chunks = self.chunker.create_chunks(current_items)

            # Prevent infinite loop if items cannot be grouped further
            if len(chunks) == len(current_items) and all(len(c) == 1 for c in chunks):
                print("[Warning] Summaries cannot be condensed further without exceeding chunk limits.")
                break

            next_level_summaries = []

            # 2. Summarize each chunk
            for i, chunk in enumerate(chunks, 1):
                chunk_text = "\n".join(chunk)
                summary = self.summarize_summaries(chunk_text)

                if summary:
                    next_level_summaries.append(summary)
                else:
                    # Fallback if API returns None: keep individual chunk items for next pass
                    print(f"[Warning] summarize_summaries failed for chunk {i}. Keeping unsummarized text.")
                    next_level_summaries.extend(chunk)

            # Safety check: if no reduction occurred in this pass, break out
            if len(next_level_summaries) >= len(current_items):
                print("[Info] Token limit reached or text could not be further reduced.")
                current_items = next_level_summaries
                break

            current_items = next_level_summaries

        # Return single final summary or join remaining items
        if len(current_items) == 1:
            return current_items[0]
        else:
            final_concat = "\n\n".join(current_items)
            final_summary = self.summarize_summaries(final_concat)
            return final_summary if final_summary else final_concat

    def translate_batch(self, item) -> tuple:

        index, texts, summary = item

        if not texts:
            return -1, {}

        # Set different request headers and URLs based on the API type.
        if self.config.get('api_type', 'openai') == 'google':
            headers = {
                'Content-Type': 'application/json'
            }
            api_url = f"{self.config['api_endpoint']}?key={self.config['api_key']}"
        else:
            headers = {
                'Content-Type': 'application/json',
                'Authorization': f'Bearer {self.config["api_key"]}'
            }
            api_url = self.config['api_endpoint']

        # Build prompts for batch translation and use special delimiters to avoid line break confusion.
        batch_text = ""
        for i, (key, value) in enumerate(texts):
            # Convert newline characters to visible markers to avoid confusion during batch processing.
            escaped_value = value.replace('\n', '\\n').replace('\t', '\\t')
            batch_text += f"[{i + 1}] {escaped_value}\n"

        input_dict = {}
        for line in batch_text.strip().split('\n'):
            if line.startswith('[') and ']' in line:
                parts = line.split(']', 1)
                key = parts[0].replace('[', '').strip()
                val = parts[1].strip()
                input_dict[key] = val

        json_batch = json.dumps(input_dict, ensure_ascii=False)

        source_lang = self.config.get('source_language')
        target_lang = self.config.get('target_language')

        prompt = f"""You are a precise JSON machine translation component.
        Translate the text values from {source_lang} to {target_lang} according to the blueprint below.

        TRANSLATION BLUEPRINT:
        {summary}

        STRICT TRANSLATION RULES:
        1. TRANSLATE VALUES ONLY: Translate all string values literally and objectively. Keep non-translatable text (code, identifiers, URLs, English technical terms) exactly as-is. Treat every string as separate task, never combine or divide strings. Never treat next string as continuation of previous one.
        2. PRESERVE STRUCTURE: Retain all JSON keys, key names, and hierarchy without omission or merging.
        3. PRESERVE FORMATTING: Retain all control characters (\\n, \\r, \\t), whitespace, and structural line breaks in their exact positions.
        4. LITERAL FIDELITY: Do not censor, modify tone, summarize, or insert placeholder notes. Translate sentence fragments as fragments.

        OUTPUT FORMAT REQUIREMENTS:
        - Output valid, raw JSON only.
        - Use standard double quotes (") for all keys and strings.
        - Do NOT wrap the output in markdown code blocks (e.g., ```json).
        - Do NOT output any introductory text, explanation, or reasoning.
        """

        if self.print_summary:
            self.print_summary = False
            self.logger.info(f"""
            Final prompt:
            ----------------------------------
            {prompt}
            ----------------------------------
            """)

        data = {
            'model': self.config['model'],
            'messages': [
                {"role": "system", "content": prompt},
                {"role": "user", "content": json_batch}
            ],
            'max_tokens': 65536,
            'safety_settings': [
                {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"},
                {"category": "HARM_CATEGORY_CIVIC_INTEGRITY", "threshold": "BLOCK_NONE"}
            ]
        }

        for attempt in range(self.config['max_retries']):
            try:
                response = requests.post(
                    api_url,
                    headers=headers,
                    json=data,
                    timeout=self.config['request_timeout']
                )

                if response.status_code == 200:
                    result = response.json()
                    if 'choices' in result and len(result['choices']) > 0:
                        translation_text = result['choices'][0]['message']['content'].strip()

                        # Robust single-pass JSON parse and repair
                        try:
                            translated_json = parse_llm_json_response(translation_text)
                        except Exception as e:
                            self.logger.error(
                                f"Gemma output is not valid JSON on attempt {attempt + 1}: {e}, problematic text: {translation_text}"
                            )
                            raise ValueError(f"Model failed to adhere to JSON format: {e}")

                        translated_results = {}

                        for i, (key, original_value) in enumerate(texts):
                            lookup_key = str(i + 1)

                            if lookup_key in translated_json and translated_json[lookup_key] is not None:
                                translated_line = str(translated_json[lookup_key]).strip()
                                translated_line = translated_line.replace('\\n', '\n').replace('\\t', '\t')

                                # Validate string content only
                                if self.is_valid_translation(original_value, translated_line):
                                    translated_results[key] = translated_line
                                    self.logger.info(
                                        f"Batch translation successful: {original_value} -> {translated_line}"
                                    )
                                else:
                                    translated_results[key] = original_value
                                    self.logger.warning(
                                        f"Translation validation failed. Retaining original: {original_value} -> {translated_line}"
                                    )
                            else:
                                translated_results[key] = original_value
                                self.logger.warning(
                                    f"Key '{lookup_key}' missing from JSON response; falling back to original value."
                                )

                        return translated_results
                    else:
                        self.logger.error(f"API response format error: {result}")

                elif response.status_code == 429:
                    self.logger.warning(f"Rate limit hit. Sleeping {self.config['retry_delay']}s...")
                    time.sleep(self.config['retry_delay'])

                elif response.status_code == 401:
                    self.logger.error("API authorization key is invalid. Aborting batch.")
                    return {}

                else:
                    self.logger.error(
                        f"API request failure. Status code: {response.status_code}, Response: {response.text}"
                    )

            except requests.exceptions.Timeout:
                self.logger.warning(
                    f"Request timed out. Retrying attempt {attempt + 1}/{self.config['max_retries']}..."
                )

            except requests.exceptions.ConnectionError:
                self.logger.warning(
                    f"Network connection failed. Retrying attempt {attempt + 1}/{self.config['max_retries']}..."
                )

            except Exception as e:
                self.logger.error(f"Unexpected exception encountered during batch processing: {str(e)}")

            if attempt < self.config['max_retries'] - 1:
                time.sleep(self.config['retry_delay'])

        fallback_results = {key: original_value for key, original_value in texts}
        self.logger.error("All translation retries exhausted. Returned un-translated fallbacks.")

        # If batch translation fails, return the original text.
        return index, {key: value for key, value in texts}

    def clean_translation_result(self, text: str) -> str:
        import re

        # Remove <think> tags and their contents (including incomplete tags).
        text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
        text = re.sub(r'<think>.*', '', text, flags=re.DOTALL)  # Remove incomplete start tags
        text = re.sub(r'.*</think>', '', text, flags=re.DOTALL)  # Remove incomplete closing tags

        # Clean up excess whitespace and line breaks.
        text = re.sub(r'\s+', ' ', text).strip()

        return text

    def is_valid_translation(self, original: str, translation: str, is_json: bool = True) -> bool:
        # 1. Check for empty or whitespace-only response
        if not translation or not translation.strip():
            print("Validation Failure: Translation result is empty or whitespace.")
            return False

        original_clean = original.strip()
        translation_clean = translation.strip()

        # 2. Case-insensitive error & LLM meta-talk pattern check
        # MUST be strictly lowercase
        error_patterns = [
            'translation failed',
            'unable to translate',
            'error occurred',
            'something went wrong',
            'unable to process',
            'as an ai',
            'i cannot translate',
            'here is the translation',
            'translator note:',
            'note from the translator:',
            'note: here is',
            'note: the above',
        ]

        translation_lower = translation_clean.lower()
        for pattern in error_patterns:
            if pattern in translation_lower:
                print(f"Validation Failure: Detected error message or LLM meta-text: '{pattern}'")
                return False

        # DO NOT perform json.loads() here! translated_line is plain text.
        return True

    def save_progress(self, translated_data: Dict[str, str], progress_file: str):
        try:
            with open(progress_file, 'w', encoding='utf-8') as f:
                json.dump(translated_data, f, ensure_ascii=False, indent=2)
            self.logger.info(f"Translation progress has been saved.: {progress_file}")
        except Exception as e:
            self.logger.error(f"Save progress failed: {str(e)}")

    def load_progress(self, progress_file: str) -> Dict[str, str]:
        if os.path.exists(progress_file):
            try:
                with open(progress_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                self.logger.info(f"from {progress_file} Loaded {len(data)} Translation record")
                return data
            except Exception as e:
                self.logger.error(f"Failed to load progress: {str(e)}")
                return {}
        return {}

    def should_translate(self, key: str, value: str) -> bool:
        # Skip only empty strings.
        if not value or not value.strip():
            return False

        # Process all other text and let the AI determine whether translation is required.
        return True

    def translate_json_file(self,
                            input_file: str,
                            output_file: str,
                            progress_file: str = None,
                            summary_file: str = None) -> bool:
        if not progress_file:
            progress_file = f"{input_file}.progress.json"
        if not summary_file:
            summary_file = "summary.txt"

        # Load raw data
        with open(input_file, 'r', encoding='utf-8') as f:
            original_data = json.load(f)

        if self.config.get('source_language') == "Japanese":
            japanese_regex = re.compile(r'[\u3040-\u309F\u30A0-\u30FF\u4E00-\u9FFF]')
            filtered_data = {
                key: value
                for key, value in original_data.items()
                if japanese_regex.search(value)
            }

            skipped_count = len(original_data) - len(filtered_data)
            print(f"Kept {len(filtered_data)} lines. Excluded {skipped_count} lines with no Japanese characters.")
            original_data = filtered_data

        self.logger.info(f"Loaded {len(original_data)} records")

        if not os.path.exists(summary_file):
            # 1. Chunk and summarize original data safely without exceeding model limits
            raw_texts = list(original_data.values())
            summary_batches = self.chunker.process_all(raw_texts, self.summarize)
            self.logger.info(f"Generated {len(summary_batches)} section summaries.")

            # 2. Hierarchically reduce the section summaries into a single final summary
            summary = self.reduce_summaries(summary_batches)
            self.logger.info(f"Final Summary generated successfully: {len(summary)} characters.")

            # 3. Save to file
            with open(summary_file, "w", encoding="utf-8") as file:
                file.write(summary)

            input(f"Summary is saved to '{summary_file}'. Review it and press Enter to continue...")

        with open("summary.txt", "r", encoding="utf-8") as f:
            summary = f.read()

        # Loading progress (if available)
        translated_data = self.load_progress(progress_file)

        # Statistical Information
        total_items = len(original_data)
        completed_items = len(translated_data)

        self.logger.info(f"total: {total_items} items, completed: {completed_items} strip")

        # Collect items requiring translation
        items_to_translate = []
        for key, value in original_data.items():
            # Skip translated items
            if key in translated_data:
                continue

            # Determine whether translation is needed.
            if not self.should_translate(key, value):
                translated_data[key] = value
                continue

            items_to_translate.append((key, value))

        self.logger.info(f"Items requiring translation: {len(items_to_translate)} strip")

        # Batch translation
        batch_size = self.config['batch_size']

        all_batches = []
        for i in range(0, len(items_to_translate), batch_size):
            batch = items_to_translate[i:i + batch_size]

            all_batches.append(batch)

        self.logger.info(f"Batches count: {len(all_batches)}")

        giga_chunks = [all_batches[i:i + WORKER_COUNT] for i in range(0, len(all_batches), WORKER_COUNT)]

        for giga_index, giga_chunk in enumerate(giga_chunks):
            results = [None] * len(giga_chunk)

            with ThreadPoolExecutor(max_workers=WORKER_COUNT) as executor:
                for i in range(0, len(giga_chunk), WORKER_COUNT):
                    sub_batch = giga_chunk[i: i + WORKER_COUNT]
                    futures = {}

                    # 1. Submit workers with a staggered delay
                    for offset, chunk in enumerate(sub_batch):
                        idx = i + offset
                        future = executor.submit(self.translate_batch, (idx, chunk, summary))
                        futures[future] = idx

                        # Delay the submission of the next worker in this batch
                        if offset < len(sub_batch) - 1:
                            time.sleep(0.2)

                    # 2. Block until the entire sub-batch completes
                    for future in as_completed(futures):
                        idx = futures[future]
                        result = future.result()
                        results[idx] = result
                        print(f"Batch translation successful, completed {giga_index}: {idx}")

            for result in results:
                if result is not None:
                    try:
                        translated_data.update(result)
                    except Exception as e:
                        print(result)
                        print(e)
            self.save_progress(translated_data, progress_file)

        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(translated_data, f, ensure_ascii=False, indent=2)

        self.logger.info(f"Translation complete! The result has been saved to: {output_file}")

        # Delete progress file
        if os.path.exists(progress_file):
            os.remove(progress_file)
            self.logger.info("Progress file deleted.")

        return True


def main():
    print("JSON")
    print("=" * 50)

    try:
        translator = JSONTranslator()
    except Exception as e:
        print(f"Failed to initialize the translator.: {e}")
        return

    # Set file path
    input_file = "translated_20260826_221112_retranslate.json"
    output_file = f"translated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    progress_file = "translation_progress.json"
    summary_file = "summary.txt"

    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Progress file: {progress_file}")
    print(f"Summary file: {summary_file}")

    # Check for the existence of a progress file.
    if os.path.exists(progress_file):
        response = input(
            "A translation progress file has been found. Do you want to resume the previous translation?(y/n): ")
        if response.lower() not in ['y', 'yes', '是']:
            os.remove(progress_file)
            print("Progress file deleted; translation will restart.")

    if os.path.exists(summary_file):
        response = input(
            "A summary file has been found. Do you want use it?(y/n): ")
        if response.lower() not in ['y', 'yes', '是']:
            os.remove(summary_file)
            print("Progress file deleted; translation will restart.")

    print("Starting translation...")
    success = translator.translate_json_file(input_file, output_file, progress_file, summary_file)

    if success:
        print("Translation successfully completed!")
    else:
        print("Translation interrupted; you can resume later.")


if __name__ == "__main__":
    main()
