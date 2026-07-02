import json
import requests
import time
import os
from typing import Dict, Any
import logging
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, as_completed


class JSONTranslator:
    def __init__(self, config_file: str = "translate_config.json"):
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

    def translate_batch(self, item) -> tuple:

        index, texts = item

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

        prompt = (
            f"You are a data translation utility. Your input is a JSON object containing line IDs and text values. "
            f"Translate the values from {source_lang} to {target_lang} and return a JSON object with the EXACT same keys.\n\n"
            f"Rules:\n"
            f"1. Never merge keys or omit keys. The output JSON must have identical keys to the input JSON.\n"
            f"2. Translate sentence fragments exactly as fragments. Do not combine text across different keys.\n"
            f"3. If a value is pure English, code, or a system ID, leave it exactly as is.\n"
            f"4. Respond ONLY with the valid JSON object. Do not include markdown formatting or explanations."
        )

        data = {
            'model': self.config['model'],
            'messages': [
                {"role": "system", "content": prompt},
                {"role": "user", "content": json_batch}
            ],
            'max_tokens': 4000,
            'temperature': 0.0,
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

                        # Securely sanitize markdown block wrappers without using literal markdown sequence in code
                        markdown_marker = chr(96) * 3
                        if translation_text.startswith(markdown_marker):
                            translation_text = translation_text.strip(markdown_marker).strip()
                            if translation_text.lower().startswith("json"):
                                translation_text = translation_text[4:].strip()

                        try:
                            translated_json = json.loads(translation_text)
                        except json.JSONDecodeError as e:
                            self.logger.error(
                                f"Gemma-3-12b output is not valid JSON on attempt {attempt + 1}: {e}"
                            )
                            raise ValueError("Model failed to adhere to the requested JSON format constraint.")

                        translated_results = {}

                        for i, (key, original_value) in enumerate(texts):
                            lookup_key = str(i + 1)

                            if lookup_key in translated_json and translated_json[lookup_key] is not None:
                                translated_line = str(translated_json[lookup_key]).strip()
                                translated_line = translated_line.replace('\\n', '\n').replace('\\t', '\t')

                                if self.is_valid_translation(original_value, translated_line):
                                    translated_results[key] = translated_line
                                    self.logger.info(
                                        f"Batch translation successful: {original_value} -> {translated_line}")
                                else:
                                    translated_results[key] = original_value
                                    self.logger.warning(
                                        f"Translation validation failed. Retaining original: {original_value}")
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
                    f"Request timed out. Retrying attempt {attempt + 1}/{self.config['max_retries']}...")

            except requests.exceptions.ConnectionError:
                self.logger.warning(
                    f"Network connection failed. Retrying attempt {attempt + 1}/{self.config['max_retries']}...")

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

    def is_valid_translation(self, original: str, translation: str) -> bool:
        if not translation or not translation.strip():
            print(f"The translation result is empty or contains only whitespace characters.")
            return False

        # Comparison after removing whitespace characters
        original_clean = original.strip()
        translation_clean = translation.strip()

        # Check the translation results for obvious errors.
        error_patterns = [
            'translation failed', 'Translation failed', 'Unable to translate', 'mistake',
            'sorry', 'i cannot', 'i can\'t', 'unable to', 'error occurred',
            'something went wrong', 'An error occurred', 'Unable to process'
        ]

        translation_lower = translation_clean.lower()
        for pattern in error_patterns:
            if pattern in translation_lower:
                print(f"The translation results contain error messages.")
                return False

        # Check if the translation is too long (it may contain errors or explanations).
        if len(translation_clean) > len(original_clean) * 10:  # Allow for larger length differences
            print(f"Translation result length abnormal")
            return False

        # All other cases are considered valid.
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
                            progress_file: str = None) -> bool:
        if not progress_file:
            progress_file = f"{input_file}.progress.json"

        # Load raw data
        with open(input_file, 'r', encoding='utf-8') as f:
            original_data = json.load(f)

        self.logger.info(f"Loaded {len(original_data)} records")

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
            self.logger.info(
                f"Processing batch {i // batch_size + 1}/{(len(items_to_translate) + batch_size - 1) // batch_size},Include {len(batch)} Project")

            all_batches.append(batch)

        giga_chunks = [all_batches[i:i + 10] for i in range(0, len(all_batches), 10)]

        for giga_index, giga_chunk in enumerate(giga_chunks):
            results = [None] * len(giga_chunk)
            with ThreadPoolExecutor(max_workers=4) as executor:
                futures = {}
                for index, chunk in enumerate(giga_chunk):
                    future = executor.submit(self.translate_batch, (index, chunk))
                    futures[future] = index

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
    input_file = "ManualTransFile.json"
    output_file = f"translated_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    progress_file = "translation_progress.json"

    print(f"Input file: {input_file}")
    print(f"Output file: {output_file}")
    print(f"Progress file: {progress_file}")

    # Check for the existence of a progress file.
    if os.path.exists(progress_file):
        response = input(
            "A translation progress file has been found. Do you want to resume the previous translation?(y/n): ")
        if response.lower() not in ['y', 'yes', '是']:
            os.remove(progress_file)
            print("Progress file deleted; translation will restart.")

    print("Starting translation...")
    success = translator.translate_json_file(input_file, output_file, progress_file)

    if success:
        print("Translation successfully completed!")
    else:
        print("Translation interrupted; you can resume later.")


if __name__ == "__main__":
    main()
