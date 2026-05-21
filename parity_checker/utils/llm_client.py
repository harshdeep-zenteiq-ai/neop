"""LLM Client for Qwen API interaction."""

import json
import requests
from typing import Dict, Any, Optional
import sys

from config import LLM_CONFIG


class QwenLLMClient:
    """Client for interacting with Qwen LLM API."""

    def __init__(self, config: Dict[str, Any] = None):
        """Initialize the LLM client with Qwen configuration.

        Parameters
        ----------
        config : dict, optional
            LLM configuration (uses LLM_CONFIG from config.py if not provided)
        """
        self.config = config or LLM_CONFIG
        self.base_url = self.config["base_url"]
        self.model = self.config["model"]
        self.api_key = self.config["api_key"]
        self.max_tokens = self.config["max_tokens"]
        self.temperature = self.config["temperature"]
        self.timeout = self.config.get("timeout", 300)

    def call(
        self,
        messages: list,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Call the Qwen LLM API.

        Parameters
        ----------
        messages : list
            List of message dictionaries with 'role' and 'content'
        system_prompt : str, optional
            System prompt to prepend
        temperature : float, optional
            Override default temperature
        max_tokens : int, optional
            Override default max_tokens

        Returns
        -------
        str
            Response content from the LLM
        """
        # Build message list
        full_messages = []

        if system_prompt:
            full_messages.append({
                "role": "system",
                "content": system_prompt
            })

        full_messages.extend(messages)

        # Prepare request
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        payload = {
            "model": self.model,
            "messages": full_messages,
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": temperature or self.temperature,
        }

        try:
            # Make API call
            response = requests.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload,
                timeout=self.timeout
            )
            response.raise_for_status()

            # Extract response
            data = response.json()
            return data["choices"][0]["message"]["content"]

        except requests.exceptions.RequestException as e:
            print(f"Error calling Qwen API: {e}", file=sys.stderr)
            raise

    def analyze_json_response(self, response: str) -> Dict[str, Any]:
        """Extract and parse JSON from LLM response.

        Parameters
        ----------
        response : str
            LLM response text

        Returns
        -------
        dict
            Parsed JSON data
        """
        # Try to find JSON in response
        try:
            # First try direct parsing
            return json.loads(response)
        except json.JSONDecodeError:
            # Try to extract JSON from markdown code blocks
            if "```json" in response:
                start = response.find("```json") + 7
                end = response.find("```", start)
                json_str = response[start:end].strip()
                return json.loads(json_str)
            elif "```" in response:
                start = response.find("```") + 3
                end = response.find("```", start)
                json_str = response[start:end].strip()
                return json.loads(json_str)
            else:
                # Try to find JSON object
                start = response.find("{")
                end = response.rfind("}") + 1
                if start >= 0 and end > start:
                    json_str = response[start:end]
                    return json.loads(json_str)
                else:
                    raise ValueError(f"Could not parse JSON from response: {response}")

    def test_connection(self) -> bool:
        """Test connection to Qwen API.

        Returns
        -------
        bool
            True if connection successful
        """
        try:
            response = self.call(
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=50
            )
            return bool(response)
        except Exception as e:
            print(f"Connection test failed: {e}", file=sys.stderr)
            return False
