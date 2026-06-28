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
            escaped_value = value.replace('\n', '\\n').replace('\t', '\\t').replace('\r', '\\r')
            batch_text += f"[{i + 1}] {escaped_value}\n"

        prompt = f"""
        You are an expert {self.config.get('source_language')}-to-{self.config.get('target_language')} translator.
        Translate the user's {self.config.get('source_language')} text into natural idiomatic, and fluid {self.config.get('target_language')}.
        Please batch process text according to the following rules:
        1. If the text contains {self.config.get('source_language')} (Hiragana, Katakana, Kanji), please translate it into {self.config.get('target_language')}.
        2. If the text is purely English, numbers, symbols, or IDs, please leave it as is.
        3. All formatting in the original text must be preserved, including newlines(\\n), \\r, \\t, spaces, punctuation, etc.
        5. Return results in the order of the input numbers, one result per line.
        6 Only return the processed results; do not add serial numbers, explanations, or other content.
        7. If the original text contains \\n, the translation result must also include \\n in the corresponding position.
        9 If the original text contains \\r, the translation result must also include \\r in the corresponding position.
        11. If you are unsure how to process the text, please leave the original text unchanged.
        12. If original text contains file extensions or looks like file path, please leave the original text unchanged.
        """

        data = {
            'model': self.config['model'],
            'messages': [
                {
                    "role": "system",
                    "content": prompt},
                {
                    'role': 'user',
                    'content': batch_text
                }
            ],
            'max_tokens': 4000,
            'temperature': 0.6
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

                        translated_results = {}
                        translation_lines = translation_text.split('\n')

                        for i, (key, original_value) in enumerate(texts):
                            if i < len(translation_lines):
                                translated_line = translation_lines[i].strip()
                                # Remove potential numbering prefixes.
                                import re
                                translated_line = re.sub(r'^\[\d+\]\s*', '', translated_line)
                                # Restore line breaks and tab characters
                                translated_line = translated_line.replace('\\n', '\n').replace('\\t', '\t')

                                if self.is_valid_translation(original_value, translated_line):
                                    translated_results[key] = translated_line
                                    self.logger.info(
                                        f"Batch translation successful: {original_value} -> {translated_line}")
                                else:
                                    translated_results[key] = original_value
                                    self.logger.warning(
                                        f"Batch translation results are invalid; please retain the original text.: {original_value}")
                            else:
                                # If the translated result does not have enough lines, retain the original text.
                                translated_results[key] = original_value
                                self.logger.warning(
                                    f"The batch translation results are insufficient in number of lines; the original text will be retained.: {key}")

                        return translated_results
                    else:
                        self.logger.error(f"API response format error: {result}")

                elif response.status_code == 429:
                    self.logger.warning(f"API limit reached, waiting {self.config['retry_delay']} Retry in seconds...")
                    time.sleep(self.config['retry_delay'])

                elif response.status_code == 401:
                    self.logger.error("API key error, please check configuration.")
                    return {}

                else:
                    self.logger.error(
                        f"Batch translation API request failed, status code: {response.status_code}, response: {response.text}")

            except requests.exceptions.Timeout:
                self.logger.warning(f"Batch translation request timed out，第 {attempt + 1} 次尝试...")

            except requests.exceptions.ConnectionError:
                self.logger.warning(f"批量翻译连接失败，第 {attempt + 1} 次尝试...")

            except Exception as e:
                self.logger.error(f"批量翻译过程中出现错误: {str(e)}")

            if attempt < self.config['max_retries'] - 1:
                time.sleep(self.config['retry_delay'])

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

        giga_chunks = [all_batches[i:i + 30] for i in range(0, len(all_batches), 30)]

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
                    print(result)
                    translated_data.update(result)
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
    print()

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
