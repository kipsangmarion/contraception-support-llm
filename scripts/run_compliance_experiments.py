#!/usr/bin/env python3
"""
LLM Compliance Experiments Runner

This script runs the three compliance experiments:
- Experiment 1: SOTA Baseline (neutral prompt)
- Experiment 2: SOTA with Compliance Prompt
- Experiment 3: Open-source with RAG

Usage:
    python scripts/run_compliance_experiments.py --experiment 1 --model gpt-4o
    python scripts/run_compliance_experiments.py --experiment 2 --model claude-3-5-sonnet
    python scripts/run_compliance_experiments.py --experiment 3 --model llama-3.1-70b --rag

Author: Research Team
Date: December 8, 2025
"""

import json
import argparse
import os
import sys
import time
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import anthropic
import openai
from tqdm import tqdm
from dotenv import load_dotenv

# Add project root to Python path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Force UTF-8 encoding for Windows console
if sys.platform == 'win32':
    import io
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

# Load environment variables from .env file
load_dotenv()

# Configure API clients (will be initialized from environment variables)
OPENAI_CLIENT = None
ANTHROPIC_CLIENT = None
GOOGLE_CLIENT = None
XAI_CLIENT = None


class ComplianceExperimentRunner:
    """Runs LLM compliance experiments."""

    def __init__(
        self,
        experiment_num: int,
        model_name: str,
        use_rag: bool = False,
        output_dir: str = "results/compliance_experiments"
    ):
        """
        Initialize experiment runner.

        Args:
            experiment_num: Experiment number (1, 2, or 3)
            model_name: Model identifier (e.g., 'gpt-4o', 'claude-3-5-sonnet')
            use_rag: Whether to use RAG (Experiment 3 only)
            output_dir: Directory to save results
        """
        self.experiment_num = experiment_num
        self.model_name = model_name
        self.use_rag = use_rag
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load test cases
        self.test_cases = self._load_test_cases()

        # Initialize API clients
        self._init_api_clients()

        # Select system prompt based on experiment
        self.system_prompt = self._get_system_prompt()

    def _load_test_cases(self) -> List[Dict]:
        """Load the compliance test dataset."""
        test_file = Path("data/compliance_test_set.json")
        if not test_file.exists():
            raise FileNotFoundError(
                f"Test dataset not found at {test_file}. "
                "Run generate_compliance_dataset.py first."
            )

        with open(test_file, 'r', encoding='utf-8') as f:
            dataset = json.load(f)

        print(f"Loaded {len(dataset['test_cases'])} test cases")
        return dataset['test_cases']

    def _init_api_clients(self):
        """Initialize API clients from environment variables."""
        global OPENAI_CLIENT, ANTHROPIC_CLIENT, GOOGLE_CLIENT, XAI_CLIENT

        # Check for OpenAI models (gpt-*, o1-*, o3-*, chatgpt-*)
        if any(x in self.model_name.lower() for x in ['gpt', 'o1-', 'o3-', 'chatgpt']):
            api_key = os.getenv('OPENAI_API_KEY')
            if not api_key:
                raise ValueError(
                    "OPENAI_API_KEY environment variable not set. "
                    "Run: python setup_env.py"
                )
            try:
                import httpx

                OPENAI_CLIENT = openai.OpenAI(api_key=api_key, http_client=httpx.Client())
                print(f"✓ Initialized OpenAI client for {self.model_name} using explicit httpx client")
            except Exception as e:
                print(f"Warning: explicit httpx client init for OpenAI failed: {e}. Falling back to openai.OpenAI(...)")
                OPENAI_CLIENT = openai.OpenAI(api_key=api_key)
                print(f"✓ Initialized OpenAI client for {self.model_name} (fallback)")

        elif 'claude' in self.model_name.lower():
            api_key = os.getenv('ANTHROPIC_API_KEY')
            if not api_key:
                raise ValueError(
                    "ANTHROPIC_API_KEY environment variable not set. "
                    "Run: python setup_env.py"
                )
            # Some httpx/http client versions can cause a TypeError when the
            # Anthropic package attempts to construct an internal httpx.Client
            # (e.g., unexpected 'proxies' arg). To avoid this, provide an
            # explicit httpx.Client instance when possible so Anthropic will
            # reuse it instead of creating one with incompatible kwargs.
            try:
                import httpx

                ANTHROPIC_CLIENT = anthropic.Anthropic(
                    api_key=api_key,
                    http_client=httpx.Client()
                )
                print(f"✓ Initialized Anthropic client for {self.model_name} using explicit httpx client")
            except Exception as e:
                # Fallback to default initialization and surface useful error
                print(f"Warning: explicit httpx client init failed: {e}. Falling back to anthropic.Anthropic(...)")
                ANTHROPIC_CLIENT = anthropic.Anthropic(api_key=api_key)
                print(f"✓ Initialized Anthropic client for {self.model_name} (fallback)")

        elif 'gemini' in self.model_name.lower():
            api_key = os.getenv('GOOGLE_API_KEY')
            if not api_key:
                raise ValueError(
                    "GOOGLE_API_KEY environment variable not set. "
                    "Run: python setup_env.py"
                )
            try:
                import google.generativeai as genai

                # Configure Gemini API (timeout handled at transport level)
                genai.configure(api_key=api_key)
                GOOGLE_CLIENT = genai
                print(f"✓ Initialized Google Gemini client for {self.model_name}")
            except ImportError:
                raise ImportError(
                    "google-generativeai package not installed. "
                    "Run: pip install google-generativeai"
                )

        elif 'grok' in self.model_name.lower():
            api_key = os.getenv('XAI_API_KEY')
            if not api_key:
                raise ValueError(
                    "XAI_API_KEY environment variable not set. "
                    "Run: python setup_env.py"
                )
            # xAI uses OpenAI-compatible API
            try:
                import httpx

                XAI_CLIENT = openai.OpenAI(
                    api_key=api_key,
                    base_url="https://api.x.ai/v1",
                    http_client=httpx.Client()
                )
                print(f"✓ Initialized xAI client for {self.model_name} using explicit httpx client")
            except Exception as e:
                print(f"Warning: explicit httpx client init for xAI failed: {e}. Falling back to openai.OpenAI(...)")
                XAI_CLIENT = openai.OpenAI(api_key=api_key, base_url="https://api.x.ai/v1")
                print(f"✓ Initialized xAI client for {self.model_name} (fallback)")

        else:
            # Placeholder for other models (Llama via Together AI, etc.)
            print(f"⚠ Model {self.model_name} - client initialization pending")

    def _get_system_prompt(self) -> str:
        """Get system prompt based on experiment number."""

        if self.experiment_num == 1:
            # Experiment 1: Neutral baseline prompt
            return (
                "You are a helpful assistant answering questions about contraception. "
                "Provide accurate, helpful information to users."
            )

        elif self.experiment_num == 2:
            # Experiment 2: Compliance-oriented prompt
            return (
                "You are a family planning counselor assistant. Your answers MUST be "
                "strictly consistent with the 2022 WHO Family Planning Global Handbook, "
                "WHO Medical Eligibility Criteria (MEC), and the Balanced Counseling "
                "Strategy (BCS+) for family planning.\n\n"
                "Guidelines:\n"
                "- Follow WHO MEC categories (1-4) for medical eligibility\n"
                "- Provide accurate effectiveness rates (typical use and perfect use)\n"
                "- Use non-directive, non-judgmental counseling language\n"
                "- Respect client autonomy and support informed choice\n"
                "- Mention side effects, warning signs, and when to seek care\n"
                "- If you are not sure about something, say you don't know and advise "
                "the user to consult a healthcare provider rather than guessing.\n\n"
                "For Rwanda-specific questions, follow Rwanda National Family Planning "
                "Policy (2021), including:\n"
                "- Adolescents can access FP without parental consent\n"
                "- All women have equal access regardless of marital status\n"
                "- Services are free or low-cost at public facilities\n"
                "- Respect cultural context and use accessible language"
            )

        elif self.experiment_num == 3:
            # Experiment 3: RAG-enhanced prompt
            if not self.use_rag:
                raise ValueError("Experiment 3 requires --rag flag")

            return (
                "You are a family planning counselor assistant answering contraception "
                "questions for users in Rwanda. Use ONLY the information provided in the "
                "CONTEXT below to answer questions. If the context is insufficient, say "
                "you don't know and advise seeing a healthcare provider.\n\n"
                "IMPORTANT:\n"
                "- Do not use information outside the provided context\n"
                "- If the context doesn't contain the answer, say so\n"
                "- Cite the source guideline when possible\n"
                "- Use non-directive, respectful language\n"
                "- Support informed, autonomous decision-making\n\n"
                "CONTEXT:\n{context}\n\n"
                "Based on the context above, please answer the following question:"
            )

        else:
            raise ValueError(f"Invalid experiment number: {self.experiment_num}")

    def _query_model(self, user_prompt: str, context: Optional[str] = None) -> Dict:
        """
        Query the LLM with a user prompt.

        Args:
            user_prompt: User's question
            context: Retrieved context (for RAG experiments)

        Returns:
            Dict with response, model, timestamp, etc.
        """
        start_time = time.time()

        # Prepare system prompt
        system_prompt = self.system_prompt
        if context and self.use_rag:
            system_prompt = system_prompt.format(context=context)

        # Call appropriate model API
        # Check for OpenAI models (gpt-*, o1-*, o3-*, chatgpt-*)
        if any(x in self.model_name.lower() for x in ['gpt', 'o1-', 'o3-', 'chatgpt']):
            response_text = self._query_openai(system_prompt, user_prompt)

        elif 'claude' in self.model_name.lower():
            response_text = self._query_anthropic(system_prompt, user_prompt)

        elif 'gemini' in self.model_name.lower():
            response_text = self._query_gemini(system_prompt, user_prompt)

        elif 'grok' in self.model_name.lower():
            response_text = self._query_grok(system_prompt, user_prompt)

        else:
            # Placeholder for other models
            response_text = f"[Model {self.model_name} not yet implemented]"

        latency = time.time() - start_time

        return {
            "model": self.model_name,
            "response": response_text,
            "latency_seconds": round(latency, 2),
            "timestamp": datetime.now().isoformat(),
            "experiment": self.experiment_num,
            "rag_used": self.use_rag,
            "context": context if self.use_rag else None
        }

    def _query_openai(self, system_prompt: str, user_prompt: str) -> str:
        """Query OpenAI API (GPT-4, o1, o3, etc.)."""
        try:
            # o1 and o3 models don't support system messages or temperature
            if 'o1' in self.model_name.lower() or 'o3' in self.model_name.lower():
                # Combine system and user prompts for o-series models
                combined_message = f"{system_prompt}\n\n{user_prompt}"

                response = OPENAI_CLIENT.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "user", "content": combined_message}
                    ],
                    # o-series models don't support temperature or max_tokens parameters
                )
            else:
                # Standard GPT models
                response = OPENAI_CLIENT.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.0,  # Deterministic for reproducibility
                    max_tokens=1000
                )

            # Extract response text
            if response.choices and len(response.choices) > 0:
                content = response.choices[0].message.content
                if content:
                    return content
                else:
                    return "[ERROR: Empty response from OpenAI]"
            else:
                return "[ERROR: No choices in OpenAI response]"

        except Exception as e:
            error_msg = str(e)
            print(f"Error querying OpenAI: {e}")

            # Provide helpful error messages
            if "model" in error_msg.lower() and "does not exist" in error_msg.lower():
                return f"[ERROR: Model '{self.model_name}' not found. Check available models at https://platform.openai.com/docs/models]"
            elif "rate" in error_msg.lower() or "quota" in error_msg.lower():
                return "[ERROR: Rate limit or quota exceeded. Wait and try again.]"
            elif "authentication" in error_msg.lower() or "api key" in error_msg.lower():
                return "[ERROR: Authentication failed. Check your OPENAI_API_KEY.]"

            return f"[ERROR: {error_msg}]"

    def _query_anthropic(self, system_prompt: str, user_prompt: str) -> str:
        """Query Anthropic API."""
        try:
            response = ANTHROPIC_CLIENT.messages.create(
                model=self.model_name,
                system=system_prompt,
                messages=[
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,  # Deterministic for reproducibility
                max_tokens=1000
            )
            return response.content[0].text

        except Exception as e:
            print(f"Error querying Anthropic: {e}")
            return f"[ERROR: {str(e)}]"

    def _query_gemini(self, system_prompt: str, user_prompt: str) -> str:
        if GOOGLE_CLIENT is None:
            msg = "Google Gemini client not initialized"
            print(f"Error querying Google Gemini: {msg}")
            return f"[ERROR: {msg}]"

        import google.generativeai as genai
        from google.api_core import exceptions as google_exceptions

        combined_prompt = system_prompt + "\n\n" + user_prompt

        try:
            # Configure safety settings to allow medical/health content
            safety_settings = {
                genai.types.HarmCategory.HARM_CATEGORY_HARASSMENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                genai.types.HarmCategory.HARM_CATEGORY_HATE_SPEECH: genai.types.HarmBlockThreshold.BLOCK_NONE,
                genai.types.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: genai.types.HarmBlockThreshold.BLOCK_NONE,
                genai.types.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: genai.types.HarmBlockThreshold.BLOCK_NONE,
            }

            model = genai.GenerativeModel(self.model_name)

            try:
                response = model.generate_content(
                    combined_prompt,
                    generation_config={
                        "temperature": 0.0,
                        "max_output_tokens": 1000,
                    },
                    safety_settings=safety_settings
                )
            except google_exceptions.DeadlineExceeded:
                return "[ERROR: Request timed out. Try a shorter prompt or different model.]"
            except google_exceptions.ResourceExhausted:
                return "[ERROR: Rate limit exceeded. Wait a moment and try again.]"
            except google_exceptions.InvalidArgument as e:
                return f"[ERROR: Invalid model or request: {str(e)}. Check model name is correct.]"

            # Handle blocked responses
            if not response.candidates:
                return "[ERROR: Response blocked by Gemini safety filters. Try rephrasing or use a different model.]"

            # Check if response has valid parts
            if not response.parts:
                # Try to get feedback on why it was blocked
                feedback = getattr(response, 'prompt_feedback', None)
                if feedback:
                    return f"[ERROR: Response blocked. Reason: {feedback}]"
                return "[ERROR: Response blocked by Gemini safety filters with no parts returned.]"

            return response.text

        except Exception as e:
            error_msg = str(e)
            # Provide helpful error message for common issues
            if "finish_reason" in error_msg and "SAFETY" in error_msg.upper():
                return "[ERROR: Gemini blocked response due to safety filters. Medical content may trigger false positives.]"
            if "timeout" in error_msg.lower() or "deadline" in error_msg.lower():
                return "[ERROR: Request timed out. Try again or use a different model.]"
            if "model" in error_msg.lower() and "not found" in error_msg.lower():
                return f"[ERROR: Model '{self.model_name}' not found. Check available models at https://ai.google.dev/models/gemini]"
            print(f"Error querying Google Gemini: {e}")
            return f"[ERROR: {error_msg}]"


    def _query_grok(self, system_prompt: str, user_prompt: str) -> str:
        """Query xAI (Grok) via OpenAI-compatible client stored in XAI_CLIENT."""
        try:
            if XAI_CLIENT is None:
                raise ValueError("xAI client not initialized")

            response = XAI_CLIENT.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,
                max_tokens=1000
            )

            # Attempt to parse response similarly to OpenAI client
            try:
                return response.choices[0].message.content
            except Exception:
                # Fallback: try common alternate fields
                if hasattr(response, 'text'):
                    return response.text
                if isinstance(response, dict):
                    # nested access for dict-shaped responses
                    return response.get('choices', [{}])[0].get('message', {}).get('content', str(response))
                return str(response)

        except Exception as e:
            print(f"Error querying xAI/Grok: {e}")
            return f"[ERROR: {str(e)}]"

    def _retrieve_context(self, query: str) -> str:
        """
        Retrieve relevant guideline context for RAG experiments.

        Uses FAISS vector store to retrieve top-k relevant chunks from
        WHO FP Handbook, WHO MEC, BCS+, and Rwanda FP guidelines.

        Args:
            query: User's question

        Returns:
            Retrieved context string
        """
        if not self.use_rag:
            return None

        try:
            # Import RAG components
            from src.rag.retriever import RAGRetriever

            # Initialize retriever (lazy initialization)
            if not hasattr(self, '_retriever'):
                self._retriever = RAGRetriever(
                    vector_store_path="data/processed/vector_store",
                    embeddings_config={
                        'model_name': 'all-MiniLM-L6-v2',
                        'provider': 'sentence-transformers'
                    },
                    relevance_threshold=0.3  # Minimum similarity score
                )

            # Retrieve context
            context = self._retriever.get_context(
                query=query,
                top_k=5,  # Retrieve top 5 most relevant chunks
                max_length=2000  # Limit context to 2000 characters
            )

            if not context:
                print(f"Warning: No relevant context found for query: {query[:100]}...")
                return "[No relevant guideline context found. Please consult a healthcare provider.]"

            return context

        except Exception as e:
            print(f"Error retrieving RAG context: {e}")
            return f"[RAG retrieval error: {str(e)}]"

    def run_experiment(self) -> Dict:
        """
        Run the experiment on all test cases.

        Returns:
            Dict with experiment results
        """
        print(f"\n{'='*60}")
        print(f"Starting Experiment {self.experiment_num}: {self.model_name}")
        print(f"RAG: {'Yes' if self.use_rag else 'No'}")
        print(f"Test cases: {len(self.test_cases)}")
        print(f"{'='*60}\n")

        results = {
            "experiment_metadata": {
                "experiment_number": self.experiment_num,
                "model": self.model_name,
                "rag_used": self.use_rag,
                "system_prompt": self.system_prompt[:200] + "...",  # Truncated
                "num_test_cases": len(self.test_cases),
                "start_time": datetime.now().isoformat()
            },
            "responses": []
        }

        # Milestone: Starting test case processing
        print("\n🚀 MILESTONE: Beginning test case processing...")
        if self.use_rag:
            print("   [RAG will initialize on first retrieval - may take ~5 seconds]")
        print()

        # Run through all test cases
        for i, test_case in enumerate(tqdm(self.test_cases, desc="Processing test cases"), 1):
            # Extract user query from scenario
            user_query = test_case['scenario']

            # Retrieve context if RAG
            context = self._retrieve_context(user_query) if self.use_rag else None

            # Query model
            model_output = self._query_model(user_query, context)

            # Combine test case info with model response
            result = {
                "test_case_id": test_case['id'],
                "category": test_case['category'],
                "severity": test_case.get('severity', 'N/A'),
                "scenario": user_query,
                "ground_truth": test_case.get('ground_truth_answer', 'N/A'),
                "model_response": model_output['response'],
                "latency_seconds": model_output['latency_seconds'],
                "timestamp": model_output['timestamp'],
                "compliant_criteria": test_case.get('compliant_response_criteria', {}),
                "non_compliant_indicators": test_case.get('non_compliant_indicators', []),
                "who_guideline": test_case.get('who_guideline', {}),
                "rwanda_context": test_case.get('rwanda_context', None),
                "rag_context": context if self.use_rag else None
            }

            results["responses"].append(result)

            # Print milestone markers at key intervals
            if i == 1:
                print(f"\n✓ MILESTONE: First test case complete! ({i}/80)")
                print(f"   Remaining cases will process faster (~15s each)\n")
            elif i == 20:
                print(f"\n✓ MILESTONE: 25% complete ({i}/80)")
                print(f"   Estimated time remaining: ~{(80-i)*15//60} minutes\n")
            elif i == 40:
                print(f"\n✓ MILESTONE: 50% complete ({i}/80) - Halfway there!")
                print(f"   Estimated time remaining: ~{(80-i)*15//60} minutes\n")
            elif i == 60:
                print(f"\n✓ MILESTONE: 75% complete ({i}/80)")
                print(f"   Estimated time remaining: ~{(80-i)*15//60} minutes\n")

            # Rate limiting (adjust based on API limits)
            time.sleep(0.5)  # 2 requests/second

        results["experiment_metadata"]["end_time"] = datetime.now().isoformat()

        # Milestone: Saving results
        print(f"\n💾 MILESTONE: All {len(self.test_cases)} test cases complete! Saving results...")

        # Save results
        output_file = self._save_results(results)
        print(f"\n{'='*60}")
        print(f"✅ Experiment {self.experiment_num} Complete!")
        print(f"Results saved to: {output_file}")
        print(f"{'='*60}\n")

        return results

    def _save_results(self, results: Dict) -> Path:
        """Save experiment results to JSON file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = (
            f"exp{self.experiment_num}_{self.model_name.replace('/', '_')}_"
            f"{'rag_' if self.use_rag else ''}{timestamp}.json"
        )
        output_path = self.output_dir / filename

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)

        return output_path


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Run LLM compliance experiments for contraception counseling"
    )
    parser.add_argument(
        "--experiment",
        type=int,
        choices=[1, 2, 3],
        required=True,
        help="Experiment number: 1=Baseline, 2=Prompted, 3=RAG"
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name (e.g., 'gpt-4o', 'claude-3-5-sonnet-20241022', 'llama-3.1-70b')"
    )
    parser.add_argument(
        "--rag",
        action="store_true",
        help="Use RAG (required for Experiment 3)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/compliance_experiments",
        help="Output directory for results"
    )

    args = parser.parse_args()

    # Validate RAG flag
    if args.experiment == 3 and not args.rag:
        print("Error: Experiment 3 requires --rag flag")
        return

    if args.experiment != 3 and args.rag:
        print("Warning: --rag flag only used in Experiment 3")

    # Run experiment
    runner = ComplianceExperimentRunner(
        experiment_num=args.experiment,
        model_name=args.model,
        use_rag=args.rag,
        output_dir=args.output_dir
    )

    results = runner.run_experiment()

    # Print summary statistics
    print("\nSummary Statistics:")
    print(f"  Total responses: {len(results['responses'])}")
    print(f"  Average latency: {sum(r['latency_seconds'] for r in results['responses']) / len(results['responses']):.2f}s")
    print(f"  Errors: {sum(1 for r in results['responses'] if '[ERROR' in r['model_response'])}")
    print("\nResults ready for annotation!")


if __name__ == "__main__":
    main()
