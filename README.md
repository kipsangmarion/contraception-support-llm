# AI Contraception Counseling System

An AI-powered contraception counseling system using RAG (Retrieval-Augmented Generation) grounded in WHO Family Planning Handbook 2022 and BCS+ Toolkit guidelines.

## Overview

This project implements a comprehensive AI system for contraception counseling that includes:
- **Guideline-grounded RAG system** for accurate, evidence-based counseling
- **Memory management** for personalized multi-session conversations
- **Adaptive adherence support** using reinforcement learning
- **Comprehensive evaluation** framework for LLM capabilities

## Project Structure

```
contraception-support-llm/
├── data/                      # Data storage
│   ├── who/                   # WHO FP Handbook PDFs
│   ├── bcs/                   # BCS+ Toolkit PDFs
│   ├── synthetic/             # Generated synthetic datasets
│   └── processed/             # Processed chunks and FAISS index
├── src/                       # Source code
│   ├── rag/                   # RAG pipeline components
│   ├── memory/                # Memory and session management
│   ├── adherence/             # RL-based adherence support
│   ├── evaluation/            # Evaluation harness
│   ├── api/                   # FastAPI endpoints
│   └── utils/                 # Utility functions
├── experiments/               # Experiment scripts
├── results/                   # Experiment results
│   ├── tables/                # Results tables (CSV)
│   ├── plots/                 # Visualizations
│   └── logs/                  # Execution logs
├── tests/                     # Unit and integration tests
├── notebooks/                 # Jupyter notebooks for analysis
├── configs/                   # Configuration files
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd contraception-support-llm
```

### 2. Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate (Windows)
venv\Scripts\activate

# Activate (Mac/Linux)
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables

```bash
# Copy the template
cp .env.template .env

# Edit .env and add your API keys
# - OPENAI_API_KEY (for GPT models)
# - ANTHROPIC_API_KEY (for Claude models, optional)
```

### 5. Configure Settings

Edit `configs/config.yaml` to customize:
- Model selection (GPT-4, Claude, etc.)
- RAG parameters (chunk size, top-k, etc.)
- Experiment settings
- Random seeds for reproducibility

## Usage

### Running the RAG System

```bash
# Start the API server
uvicorn src.api.main:app --reload

# Access API at http://localhost:8000
# API docs at http://localhost:8000/docs
```

### Running Experiments

```bash
# Run all experiments
python experiments/run_all_experiments.py

# Run individual experiments
python experiments/exp1_baseline.py
python experiments/exp2_anchored.py
python experiments/exp3_rag_comparison.py
python experiments/exp4a_long_session.py
python experiments/exp4b_multi_session.py
python experiments/exp5_adherence_rl.py
```

### Data Preparation

```bash
# Process WHO and BCS+ documents
python src/rag/preprocess_documents.py

# Generate synthetic datasets
python src/utils/generate_synthetic_data.py
```

## Experiments

### Experiment 1: Baseline LLM Knowledge Test
Tests raw LLM knowledge without RAG or anchoring prompts.

**Metrics:** Accuracy, hallucination rate, latency

### Experiment 2: Anchored Prompt Evaluation
Tests LLM with strict guideline-following prompts.

**Metrics:** Accuracy, safety fallback rate, hallucination reduction

### Experiment 3: RAG vs Non-RAG Comparison
Compares raw LLM, anchored LLM, and full RAG system.

**Metrics:** Accuracy, grounding, latency, safety errors

### Experiment 4A: Long-Session Forgetting Test
Tests memory retention in 20-40 turn conversations.

**Metrics:** Contradiction rate, recall accuracy

### Experiment 4B: Multi-Session Memory Test
Compares different memory strategies across sessions.

**Metrics:** Memory recall accuracy, consistency

### Experiment 5: Adaptive Adherence Support
Compares LinUCB reinforcement learning against baselines.

**Metrics:** Cumulative reward, convergence rate

## Results

Results are saved in the `results/` directory:
- **Tables:** CSV files with detailed metrics
- **Plots:** Visualizations (PNG/SVG)
- **Logs:** Execution logs and debugging info

## Reproducibility

All experiments use fixed random seeds configured in `configs/config.yaml`. Environment information is automatically logged for each experiment.

To reproduce results:
1. Ensure same Python version (3.9+)
2. Install exact package versions from `requirements.txt`
3. Use same configuration in `configs/config.yaml`
4. Run experiments in order

## Testing

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/

# Run specific test file
pytest tests/test_rag.py
```

## Development

### Code Style

```bash
# Format code
black src/ tests/

# Lint code
flake8 src/ tests/

# Type checking
mypy src/
```

### Adding New Features

1. Create feature branch: `git checkout -b feature/your-feature`
2. Implement feature with tests
3. Update documentation
4. Submit pull request

## API Documentation

Once the server is running, access interactive API documentation at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Contributing

See [IMPLEMENTATION_CHECKLIST.md](IMPLEMENTATION_CHECKLIST.md) for development roadmap.

## License

[Your License Here]

## Citation

If you use this system in your research, please cite:

```bibtex
@software{contraception_counseling_ai,
  title = {AI Contraception Counseling System},
  author = {Your Name},
  year = {2025},
  url = {https://github.com/yourusername/contraception-support-llm}
}
```

## Acknowledgments

- WHO Family Planning Handbook 2022
- BCS+ Toolkit
- OpenAI / Anthropic for LLM APIs
- LangChain community

## Contact

For questions or issues, please open an issue on GitHub or contact [your email].
