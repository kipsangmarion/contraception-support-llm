"""
Test script to compare Aya vs llama3.2 for Kinyarwanda responses.

This will demonstrate the improvement in Kinyarwanda quality when using
Aya (trained on African languages) vs llama3.2 (Swahili mixing).
"""

import sys
import io
import json
from pathlib import Path

# Fix UTF-8 encoding for Windows console
if sys.platform == 'win32':
    try:
        sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
        sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')
    except:
        pass

from src.rag.rag_pipeline import RAGPipelineWithMemory

def main():
    """Test Kinyarwanda responses with Aya model."""

    print("="*80)
    print("KINYARWANDA QUALITY TEST: Aya vs llama3.2")
    print("="*80)
    print()
    print("Testing Kinyarwanda responses with language-specific model routing:")
    print("- Kinyarwanda → Aya (aya:8b) - Trained on African languages")
    print("- English → llama3.2")
    print("- French → llama3.2")
    print()

    # Load multi-language questions
    multilang_file = Path("data/synthetic/multilang_qa_pairs.json")
    if not multilang_file.exists():
        print("Error: multilang_qa_pairs.json not found!")
        return

    with open(multilang_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    # Get Kinyarwanda questions
    kinyarwanda_questions = [
        q for q in dataset['questions']
        if q.get('language') == 'kinyarwanda'
    ]

    if not kinyarwanda_questions:
        print("Error: No Kinyarwanda questions found!")
        return

    print(f"Found {len(kinyarwanda_questions)} Kinyarwanda questions")
    print()

    # Initialize pipeline (now uses MultiLanguageLLMClient)
    print("Initializing RAG pipeline with multi-language routing...")
    pipeline = RAGPipelineWithMemory('configs/config.yaml')
    print("✓ Pipeline initialized")
    print()

    # Test first 3 questions
    test_questions = kinyarwanda_questions[:3]

    for i, q in enumerate(test_questions, 1):
        print("="*80)
        print(f"TEST {i}/{len(test_questions)}")
        print("="*80)
        print(f"Category: {q.get('category', 'N/A')}")
        print(f"Question: {q['question']}")
        print()
        print(f"Expected: {q.get('ground_truth', 'N/A')[:100]}...")
        print()
        print("-"*80)
        print("GENERATING RESPONSE WITH AYA MODEL...")
        print("-"*80)

        try:
            result = pipeline.query(q['question'], language='kinyarwanda')
            response = result['response']

            print()
            print(f"Response ({len(response)} chars):")
            print(response)
            print()

            # Check for Swahili words (should be reduced/eliminated)
            swahili_markers = [
                'kikamilifu', 'kwa mpangilio', 'kusawazishwa',
                'matibabu', 'mwenyeji', 'vipimo'
            ]
            swahili_found = [word for word in swahili_markers if word in response.lower()]

            print("-"*80)
            print("QUALITY CHECK:")
            print("-"*80)
            print(f"Sources cited: {len(result.get('sources', []))}")
            print(f"Swahili words detected: {len(swahili_found)}")
            if swahili_found:
                print(f"  Swahili words: {', '.join(swahili_found)}")
                print("  ⚠️ Still contains some Swahili")
            else:
                print("  ✓ No Swahili mixing detected!")

            # Check if response contains safety fallback
            safety_keywords = ['muganga', 'ikigo', 'inama']
            has_safety = any(word in response.lower() for word in safety_keywords)
            print(f"Safety keywords: {'✓ Present' if has_safety else '✗ Not detected'}")

            print()

        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()

        if i < len(test_questions):
            print()
            input("Press Enter to continue to next question...")
            print()

    print("="*80)
    print("KINYARWANDA QUALITY TEST COMPLETE")
    print("="*80)
    print()
    print("Summary:")
    print("- Aya model (aya:8b) is now used for Kinyarwanda")
    print("- Should see reduced/eliminated Swahili mixing")
    print("- Better language quality due to African language training")
    print()
    print("Next steps:")
    print("1. Compare this output with previous test_multilang.py results")
    print("2. Document improvement in Kinyarwanda quality")
    print("3. Update MULTILANG_COMPLETE.md with Aya integration")


if __name__ == "__main__":
    main()
