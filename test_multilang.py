"""
Quick test script for multi-language capabilities with UTF-8 support.
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
        pass  # Already wrapped or not applicable

from src.rag.rag_pipeline import RAGPipelineWithMemory

# Initialize pipeline
print("Initializing RAG pipeline...")
pipeline = RAGPipelineWithMemory('configs/config.yaml')

# Load multi-language questions
multilang_file = Path("data/synthetic/multilang_qa_pairs.json")
if multilang_file.exists():
    with open(multilang_file, 'r', encoding='utf-8') as f:
        dataset = json.load(f)

    # Get sample questions from dataset
    french_q = next((q for q in dataset['questions'] if q.get('language') == 'french'), None)
    kinyarwanda_q = next((q for q in dataset['questions'] if q.get('language') == 'kinyarwanda'), None)

    questions = {
        'english': {
            'question': "What is the effectiveness rate of the copper IUD?",
            'expected': "The copper IUD is more than 99% effective..."
        },
        'french': {
            'question': french_q['question'] if french_q else "Quelle est l'efficacité de l'implant contraceptif?",
            'expected': french_q.get('ground_truth', 'N/A')[:100] + "..." if french_q else "N/A"
        },
        'kinyarwanda': {
            'question': kinyarwanda_q['question'] if kinyarwanda_q else "Ni gute 'implant' yo kuboneza urubyaro ikora?",
            'expected': kinyarwanda_q.get('ground_truth', 'N/A')[:100] + "..." if kinyarwanda_q else "N/A"
        }
    }
else:
    # Fallback questions
    questions = {
        'english': {
            'question': "What is the effectiveness rate of the copper IUD?",
            'expected': "The copper IUD is more than 99% effective..."
        },
        'french': {
            'question': "Quelle est l'efficacité de l'implant contraceptif?",
            'expected': "L'implant contraceptif est plus de 99% efficace..."
        },
        'kinyarwanda': {
            'question': "Ni gute 'implant' yo kuboneza urubyaro ikora?",
            'expected': "Implant ikora neza cyane, irenga 99%..."
        }
    }

print("="*80)
print("MULTI-LANGUAGE RESPONSE TEST")
print("="*80)
print(f"\nTesting {len(questions)} languages: English, French, Kinyarwanda")
print(f"Using dataset: {multilang_file if multilang_file.exists() else 'fallback questions'}\n")

for lang, data in questions.items():
    print(f"\n{'='*80}")
    print(f"Language: {lang.upper()}")
    print(f"Question: {data['question']}")
    print(f"Expected: {data['expected']}")
    print(f"{'='*80}")

    try:
        result = pipeline.query(data['question'], language=lang)
        response = result['response']

        print(f"\nResponse ({len(response)} chars):")
        print(response[:600] if len(response) > 600 else response)
        print(f"\n... (truncated)" if len(response) > 600 else "")
        print(f"\nSources: {len(result.get('sources', []))} documents cited")

        # Check if response contains safety fallback
        is_fallback = any(phrase in response.lower() for phrase in ['recommend consulting', 'consulter un professionnel', 'muganga'])
        print(f"Safety fallback: {'Yes' if is_fallback else 'No'}")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)
print("\nFor comprehensive multi-language evaluation, run:")
print("  python experiments/exp1_baseline_knowledge.py --language french")
print("  python experiments/exp1_baseline_knowledge.py --language kinyarwanda")
