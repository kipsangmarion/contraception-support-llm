"""
Multi-Language LLM Client with Language-Specific Model Routing

Routes queries to appropriate models based on language:
- English: llama3.2 (default, best performance)
- French: llama3.2 (good performance)
- Kinyarwanda: aya:8b (trained on African languages)

This eliminates the Swahili mixing issue in Kinyarwanda responses.
"""

import os
import requests
from typing import Dict, List, Optional
from loguru import logger


class MultiLanguageLLMClient:
    """LLM client that routes to language-specific models."""

    # Language-to-model mapping
    LANGUAGE_MODELS = {
        'english': 'llama3.2',
        'french': 'llama3.2',
        'kinyarwanda': 'aya:8b'
    }

    def __init__(self, config: Dict):
        """
        Initialize multi-language LLM client.

        Args:
            config: LLM configuration dictionary
        """
        # Handle both full config and direct llm_config
        if 'models' in config:
            llm_config = config['models']['llm']
        elif 'provider' in config:
            llm_config = config
        else:
            raise ValueError("Invalid config format")

        self.provider = llm_config.get('provider', 'ollama')
        self.default_model = llm_config.get('model_name', 'llama3.2')
        self.temperature = llm_config.get('temperature', 0.1)
        self.max_tokens = llm_config.get('max_tokens', 1024)
        self.base_url = llm_config.get('base_url', 'http://localhost:11434')

        logger.info(f"Multi-language LLM client initialized with provider: {self.provider}")
        logger.info(f"Language routing: {self.LANGUAGE_MODELS}")

        if self.provider != 'ollama':
            logger.warning(
                "Multi-language routing currently only supports Ollama. "
                f"Using default model for all languages: {self.default_model}"
            )

    def _get_model_for_language(self, language: str) -> str:
        """
        Get the appropriate model for a given language.

        Args:
            language: Language code ('english', 'french', 'kinyarwanda')

        Returns:
            Model name to use
        """
        language_lower = language.lower() if language else 'english'
        model = self.LANGUAGE_MODELS.get(language_lower, self.default_model)
        logger.debug(f"Routing {language} to model: {model}")
        return model

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        language: str = 'english'
    ) -> str:
        """
        Generate text from prompt using language-appropriate model.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Optional temperature override
            max_tokens: Optional max_tokens override
            language: Target language for routing

        Returns:
            Generated text
        """
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens

        # Get language-specific model
        model = self._get_model_for_language(language)

        if self.provider == "ollama":
            return self._generate_ollama(
                prompt, system_prompt, temp, max_tok, model
            )
        else:
            # Fallback for other providers (use default model)
            logger.warning(
                f"Language routing not available for {self.provider}, "
                f"using default model: {self.default_model}"
            )
            return self._generate_ollama(
                prompt, system_prompt, temp, max_tok, self.default_model
            )

    def _generate_ollama(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
        model: str
    ) -> str:
        """Generate using Ollama local API with specified model."""
        url = f"{self.base_url}/api/generate"

        # Combine system prompt and user prompt
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        payload = {
            "model": model,
            "prompt": full_prompt,
            "temperature": temperature,
            "options": {
                "num_predict": max_tokens
            },
            "stream": False
        }

        try:
            logger.info(f"Generating with model: {model}")
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            return result.get('response', '')

        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. "
                f"Make sure Ollama is running and model '{model}' is downloaded."
            )
        except Exception as e:
            logger.error(f"Ollama generation error with model {model}: {e}")
            raise

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        language: str = 'english'
    ) -> str:
        """
        Chat completion with language-specific model.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Optional temperature override
            max_tokens: Optional max_tokens override
            language: Target language for routing

        Returns:
            Generated response
        """
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens

        # Get language-specific model
        model = self._get_model_for_language(language)

        if self.provider == "ollama":
            url = f"{self.base_url}/api/chat"

            payload = {
                "model": model,
                "messages": messages,
                "temperature": temp,
                "options": {
                    "num_predict": max_tok
                },
                "stream": False
            }

            try:
                logger.info(f"Chat with model: {model} (language: {language})")
                response = requests.post(url, json=payload, timeout=120)
                response.raise_for_status()
                result = response.json()
                return result.get('message', {}).get('content', '')

            except requests.exceptions.ConnectionError:
                raise ConnectionError(
                    f"Cannot connect to Ollama at {self.base_url}. "
                    f"Make sure Ollama is running and model '{model}' is downloaded."
                )
            except Exception as e:
                logger.error(f"Chat error with model {model}: {e}")
                raise
        else:
            raise ValueError(f"Chat not implemented for provider: {self.provider}")


def test_multilang_llm(config: Dict) -> bool:
    """
    Test multi-language LLM routing.

    Args:
        config: Configuration dictionary

    Returns:
        True if all tests successful
    """
    try:
        client = MultiLanguageLLMClient(config)

        test_cases = [
            {
                'language': 'english',
                'prompt': 'Say hello in one word.',
                'expected_model': 'llama3.2'
            },
            {
                'language': 'french',
                'prompt': 'Dis bonjour en un mot.',
                'expected_model': 'llama3.2'
            },
            {
                'language': 'kinyarwanda',
                'prompt': 'Vuga mwaramutse mu jambo rimwe.',
                'expected_model': 'aya:8b'
            }
        ]

        success = True
        for test in test_cases:
            logger.info(f"\nTesting {test['language']}...")
            logger.info(f"Expected model: {test['expected_model']}")

            try:
                response = client.generate(
                    test['prompt'],
                    language=test['language']
                )
                logger.info(f"Response: {response[:100]}...")
                logger.info(f"✓ {test['language']} test passed")
            except Exception as e:
                logger.error(f"✗ {test['language']} test failed: {e}")
                success = False

        return success

    except Exception as e:
        logger.error(f"Multi-language LLM test failed: {e}")
        return False


if __name__ == "__main__":
    import yaml
    from pathlib import Path

    # Load config
    config_path = Path(__file__).parent.parent.parent / "configs" / "config.yaml"
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    # Test multi-language routing
    logger.info("Testing Multi-Language LLM Client...")
    success = test_multilang_llm(config)

    if success:
        logger.info("\n✓ All tests passed!")
    else:
        logger.warning("\n✗ Some tests failed")
