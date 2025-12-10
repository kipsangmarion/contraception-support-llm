"""
Multi-Language LLM Client with Language-Specific Model Routing

Routes queries to appropriate models based on language:
- All languages: Claude Opus 4.5 (based on Experiment 2 results)
  - English: 76.25% compliant, 0 critical issues
  - Multilingual support for French and Kinyarwanda

Uses Anthropic API for all language routing.
"""

import os
from typing import Dict, List, Optional
from loguru import logger
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class MultiLanguageLLMClient:
    """LLM client that routes to language-specific models."""

    # Language-to-model mapping
    # Updated to use Claude Opus 4.5 for all languages (Experiment 2 winner)
    LANGUAGE_MODELS = {
        'english': 'claude-opus-4-5-20251101',
        'french': 'claude-opus-4-5-20251101',
        'kinyarwanda': 'claude-opus-4-5-20251101'
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

        self.provider = llm_config.get('provider', 'anthropic')
        self.default_model = llm_config.get('model_name', 'claude-opus-4-5-20251101')
        self.temperature = llm_config.get('temperature', 0.7)
        self.max_tokens = llm_config.get('max_tokens', 1024)

        # Initialize Anthropic client
        try:
            import anthropic
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if not api_key:
                raise ValueError(
                    "ANTHROPIC_API_KEY not found in environment variables. "
                    "Please set it in your .env file or run: python setup_env.py"
                )
            self.anthropic_client = anthropic.Anthropic(api_key=api_key)
            logger.info("Initialized Anthropic client for Claude Opus 4.5")
        except ImportError:
            raise ImportError(
                "anthropic package not installed. "
                "Please run: pip install anthropic"
            )

        logger.info(f"Multi-language LLM client initialized with provider: {self.provider}")
        logger.info(f"Language routing: {self.LANGUAGE_MODELS}")

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

        return self._generate_anthropic(
            prompt, system_prompt, temp, max_tok, model
        )

    def _generate_anthropic(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int,
        model: str
    ) -> str:
        """Generate using Anthropic Claude API."""
        try:
            logger.info(f"Generating with model: {model}")

            # Claude uses messages API
            messages = [{"role": "user", "content": prompt}]

            response = self.anthropic_client.messages.create(
                model=model,
                max_tokens=max_tokens,
                temperature=temperature,
                system=system_prompt if system_prompt else "",
                messages=messages
            )

            return response.content[0].text

        except Exception as e:
            logger.error(f"Anthropic generation error with model {model}: {e}")
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

        try:
            logger.info(f"Chat with model: {model} (language: {language})")

            response = self.anthropic_client.messages.create(
                model=model,
                max_tokens=max_tok,
                temperature=temp,
                messages=messages
            )

            return response.content[0].text

        except Exception as e:
            logger.error(f"Chat error with model {model}: {e}")
            raise


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
                'expected_model': 'claude-opus-4-5-20251101'
            },
            {
                'language': 'french',
                'prompt': 'Dis bonjour en un mot.',
                'expected_model': 'claude-opus-4-5-20251101'
            },
            {
                'language': 'kinyarwanda',
                'prompt': 'Vuga mwaramutse mu jambo rimwe.',
                'expected_model': 'claude-opus-4-5-20251101'
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
