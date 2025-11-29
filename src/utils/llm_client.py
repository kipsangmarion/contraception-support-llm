"""
Unified LLM client supporting multiple providers.
Supports: OpenAI, Anthropic, Ollama, and Groq.
"""

import os
import requests
from typing import Dict, List, Optional
from loguru import logger


class LLMClient:
    """Unified client for different LLM providers."""

    def __init__(self, config: Dict):
        """
        Initialize LLM client from configuration.

        Args:
            config: Configuration dictionary with 'models' section
        """
        llm_config = config.get('models', {}).get('llm', {})

        self.provider = llm_config.get('provider', 'openai')
        self.model_name = llm_config.get('model_name', 'gpt-4o-mini')
        self.temperature = llm_config.get('temperature', 0.1)
        self.max_tokens = llm_config.get('max_tokens', 1024)
        self.base_url = llm_config.get('base_url', 'http://localhost:11434')

        logger.info(f"Initializing LLM client: {self.provider} - {self.model_name}")

        self._setup_client()

    def _setup_client(self):
        """Set up the appropriate client based on provider."""
        if self.provider == "openai":
            from openai import OpenAI
            self.client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

        elif self.provider == "anthropic":
            from anthropic import Anthropic
            self.client = Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

        elif self.provider == "ollama":
            # Ollama uses REST API
            self.client = None
            logger.info(f"Ollama client configured for {self.base_url}")

        elif self.provider == "groq":
            from openai import OpenAI
            # Groq uses OpenAI-compatible API
            self.client = OpenAI(
                api_key=os.getenv("GROQ_API_KEY"),
                base_url="https://api.groq.com/openai/v1"
            )

        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Generate text from prompt.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            temperature: Optional temperature override
            max_tokens: Optional max_tokens override

        Returns:
            Generated text
        """
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens

        if self.provider == "openai" or self.provider == "groq":
            return self._generate_openai(prompt, system_prompt, temp, max_tok)

        elif self.provider == "anthropic":
            return self._generate_anthropic(prompt, system_prompt, temp, max_tok)

        elif self.provider == "ollama":
            return self._generate_ollama(prompt, system_prompt, temp, max_tok)

        else:
            raise ValueError(f"Unknown provider: {self.provider}")

    def _generate_openai(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int
    ) -> str:
        """Generate using OpenAI API."""
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        response = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens
        )

        return response.choices[0].message.content

    def _generate_anthropic(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int
    ) -> str:
        """Generate using Anthropic Claude API."""
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=max_tokens,
            temperature=temperature,
            system=system_prompt or "",
            messages=[{"role": "user", "content": prompt}]
        )

        return response.content[0].text

    def _generate_ollama(
        self,
        prompt: str,
        system_prompt: Optional[str],
        temperature: float,
        max_tokens: int
    ) -> str:
        """Generate using Ollama local API."""
        url = f"{self.base_url}/api/generate"

        # Combine system prompt and user prompt
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"

        payload = {
            "model": self.model_name,
            "prompt": full_prompt,
            "temperature": temperature,
            "options": {
                "num_predict": max_tokens
            },
            "stream": False
        }

        try:
            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            return result.get('response', '')

        except requests.exceptions.ConnectionError:
            raise ConnectionError(
                f"Cannot connect to Ollama at {self.base_url}. "
                "Make sure Ollama is running (ollama serve) and the model is downloaded."
            )
        except Exception as e:
            logger.error(f"Ollama generation error: {e}")
            raise

    def chat(
        self,
        messages: List[Dict[str, str]],
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Chat completion with conversation history.

        Args:
            messages: List of message dicts with 'role' and 'content'
            temperature: Optional temperature override
            max_tokens: Optional max_tokens override

        Returns:
            Generated response
        """
        temp = temperature if temperature is not None else self.temperature
        max_tok = max_tokens if max_tokens is not None else self.max_tokens

        if self.provider == "openai" or self.provider == "groq":
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=temp,
                max_tokens=max_tok
            )
            return response.choices[0].message.content

        elif self.provider == "anthropic":
            # Anthropic requires system message separately
            system_msg = ""
            user_messages = []

            for msg in messages:
                if msg['role'] == 'system':
                    system_msg = msg['content']
                else:
                    user_messages.append(msg)

            response = self.client.messages.create(
                model=self.model_name,
                max_tokens=max_tok,
                temperature=temp,
                system=system_msg,
                messages=user_messages
            )
            return response.content[0].text

        elif self.provider == "ollama":
            url = f"{self.base_url}/api/chat"

            payload = {
                "model": self.model_name,
                "messages": messages,
                "temperature": temp,
                "options": {
                    "num_predict": max_tok
                },
                "stream": False
            }

            response = requests.post(url, json=payload, timeout=120)
            response.raise_for_status()
            result = response.json()
            return result.get('message', {}).get('content', '')

        else:
            raise ValueError(f"Unknown provider: {self.provider}")


def test_llm_connection(config: Dict) -> bool:
    """
    Test LLM connection.

    Args:
        config: Configuration dictionary

    Returns:
        True if connection successful
    """
    try:
        client = LLMClient(config)
        response = client.generate("Say 'Hello!' in one word.")
        logger.info(f"LLM test successful. Response: {response}")
        return True
    except Exception as e:
        logger.error(f"LLM test failed: {e}")
        return False
