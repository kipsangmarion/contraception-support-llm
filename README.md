# AI Contraception Counseling System

**Graduate Capstone Project**
An AI-powered contraception counseling system using compliance-aware prompting grounded in WHO Family Planning Handbook 2022 and BCS+ Toolkit guidelines.

[![Status](https://img.shields.io/badge/status-production--ready-green)]()
[![Python](https://img.shields.io/badge/python-3.10+-blue)]()
[![License](https://img.shields.io/badge/license-MIT-blue)]()

---

## Overview

This project implements a comprehensive AI system for evidence-based contraception counseling with:

- **Compliance-Aware Approach**: Direct LLM generation with WHO/BCS+ compliance-aware system prompts (76.25% compliant, 0 critical issues)
- **Safety Validation**: Post-generation safety checks to catch unsafe language patterns (Experiment 4)
- **Multi-Language Support**: English, French, and Kinyarwanda using Claude Opus 4.5
- **Memory Management**: Session-based conversation history and cross-session user profiles
- **Privacy-First Design**: Anonymous IDs, opt-in data collection, GDPR compliance
- **Comprehensive Evaluation**: Compliance scoring, hallucination detection, safety assessment
- **Production-Ready**: FastAPI, Docker, extensive testing

**Note**: This system uses Experiment 4's approach: compliance-aware prompting (Exp 2: 76.25% compliant, 0 critical issues) enhanced with post-generation safety validation for additional protection.

---

## Quick Start

### Prerequisites
- Python 3.10+
- Anthropic API key (for Claude Opus 4.5)
- 4GB+ RAM
- 2GB+ disk space

### Installation

```bash
# 1. Clone repository
git clone <repository-url>
cd contraception-support-llm

# 2. Create virtual environment
python -m venv venv
source venv/bin/activate  # Mac/Linux
venv\Scripts\activate     # Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set up environment
python setup_env.py  # Creates .env file and sets ANTHROPIC_API_KEY

# Alternative: Manual setup
# Create .env file and add:
# ANTHROPIC_API_KEY=your_key_here
```

### Running the System

```bash
# Start API server
uvicorn src.api.main:app --reload --host 0.0.0.0 --port 8000

# Access at:
# - Web UI: http://localhost:8000
# - API docs: http://localhost:8000/docs
```

---

## Experiments

The system has completed 4 core experiments using the unified pipeline:

```bash
# Test the unified pipeline (without running full dataset)
python tests/test_unified_pipeline.py

# Run compliance experiments
python scripts/run_compliance_experiments.py
```

**Completed Experiments**:
1. **Baseline Knowledge** - LLM knowledge without compliance prompts or RAG
2. **Compliance-Aware Prompting** - System prompts with WHO guidelines (76.25% compliant, 0 critical issues)
3. **RAG Enhancement** - RAG retrieval with compliance prompts (showed 35% degradation vs. prompting alone)
4. **Safety Validation** - Real-time safety checks on LLM outputs

**Results**: Compliance-aware prompting (Exp 2) outperformed RAG (Exp 3) by 41% in compliance metrics.

---

## Project Structure

```
contraception-support-llm/
├── configs/
│   └── config.yaml                    # Central configuration
├── data/
│   ├── compliance_test_set.json       # Compliance evaluation dataset
│   ├── memory/                        # Conversation & profile storage
│   └── collected/                     # Opt-in data collection
├── src/
│   ├── api/main.py                    # FastAPI application
│   ├── pipeline/                      # Compliance-aware pipeline
│   │   ├── compliance_pipeline.py     # Main orchestrator
│   │   └── generator.py               # LLM generation with compliance prompts
│   ├── memory/                        # Memory management
│   │   ├── memory_manager.py          # Orchestrator
│   │   └── conversation_memory.py     # Session tracking
│   ├── evaluation/                    # Compliance evaluation
│   │   ├── annotator.py               # LLM-as-judge annotation
│   │   └── agreement.py               # Inter-annotator agreement
│   └── utils/
│       ├── multilang_llm_client.py    # Multi-language routing (Claude Opus 4.5)
│       ├── data_collection.py         # Privacy-preserving collection
│       └── logger.py                  # Logging setup
├── experiments/                       # Experiment scripts
├── scripts/                           # Analysis & utility scripts
├── tests/                             # Test suites
│   ├── test_unified_pipeline.py       # Pipeline tests (4 experiments)
│   └── test_unified_analysis.py       # Analysis tests
├── static/                            # Web UI (HTML/CSS/JS)
├── results/                           # Experiment outputs
├── setup_env.py                       # Environment setup script
├── requirements.txt                   # Python dependencies
└── README.md                          # This file
```

---

## Architecture

### High-Level System Architecture

```
┌──────────────────────────────────────────────────────────┐
│                     USER LAYER                           │
│  Web Browser / Mobile App / API Clients                 │
└────────────────────┬─────────────────────────────────────┘
                     │ HTTPS/REST API
┌────────────────────▼─────────────────────────────────────┐
│              FastAPI Application Layer                   │
│  ┌────────────┐  ┌────────────┐  ┌─────────────────┐   │
│  │ Counseling │  │  Memory    │  │ Data Collection │   │
│  │ Endpoints  │  │  Endpoints │  │ (GDPR)          │   │
│  └────────────┘  └────────────┘  └─────────────────┘   │
└────────────────────┬─────────────────────────────────────┘
                     │
┌────────────────────▼─────────────────────────────────────┐
│      Compliance-Aware Pipeline with Memory               │
│  ┌──────────────┐  ┌──────────┐                         │
│  │  Generator   │→ │  Memory  │→ Response               │
│  │(Compliance-  │  │ Manager  │                         │
│  │ Aware Prompts│  │          │                         │
│  └──────────────┘  └──────────┘                         │
└────────────────────┬─────────────────────────────────────┘
                     │
    ┌────────────────┼────────────────┐
    ▼                ▼
┌──────────────┐  ┌──────────┐
│   Anthropic  │  │  Memory  │
│   Claude     │  │  Storage │
│   Opus 4.5   │  │  (JSON)  │
└──────────────┘  └──────────┘
```

See [docs/GUIDELINE_SCOPE_AND_COMPLIANCE.md](docs/GUIDELINE_SCOPE_AND_COMPLIANCE.md) for comprehensive guideline documentation.

---

## Key Features

### 1. Compliance-Aware Pipeline with Safety Validation
- **Direct LLM Generation**: No RAG retrieval - uses model's built-in WHO guideline knowledge
- **Compliance-Aware Prompts**: Experiment 2 system prompts with WHO MEC categories, effectiveness rates
- **Post-Generation Safety Validation**: Experiment 4 adds pattern-based safety checks to catch unsafe language
- **Safety-First**: Explicit uncertainty handling, non-directive counseling language, automatic fallbacks
- **Results**: 76.25% compliant with 0 critical safety issues (Claude Opus 4.5, Exp 2 baseline)

### 2. Multi-Language Support
- **All Languages**: Claude Opus 4.5 - Consistent performance across English, French, Kinyarwanda
- **Auto-Detection**: Automatic language detection for seamless multilingual support
- **Cultural Context**: Rwanda-specific policy compliance in all languages

Language routing via [MultiLanguageLLMClient](src/utils/multilang_llm_client.py):
```python
LANGUAGE_MODELS = {
    'english': 'claude-opus-4-5-20251101',
    'french': 'claude-opus-4-5-20251101',
    'kinyarwanda': 'claude-opus-4-5-20251101'
}
```

### 3. Memory Management
- **Session-based**: Conversation history per session
- **User Profiles**: Cross-session persistence
- **Summarization**: Auto-condense after 10+ turns
- **Max History**: 20 turns (configurable)

### 4. Privacy & GDPR Compliance
- **Anonymous IDs**: UUID-based, no PII
- **Opt-in Collection**: Disabled by default
- **Right to Access**: Export user data
- **Right to be Forgotten**: Delete user data
- **Transparent**: Clear consent forms

### 5. Evaluation Framework
- **Compliance Scoring**: WHO MEC category adherence, effectiveness rate accuracy
- **Safety Assessment**: Critical issue detection, non-directive language verification
- **Hallucination Detection**: Medical accuracy checks
- **Multi-Model Comparison**: Claude Opus, o3, Grok evaluation
- **Statistical Tests**: Paired t-tests, confidence intervals

---

## API Endpoints

### Counseling
```bash
# Submit question
POST /counsel/query
{
  "question": "What is emergency contraception?",
  "language": "english",
  "session_id": "uuid",
  "user_id": "uuid"
}

# Submit feedback
POST /counsel/feedback

# Get conversation history
GET /counsel/conversation/{session_id}

# Delete conversation
DELETE /counsel/conversation/{session_id}
```

### Memory Management
```bash
# Create/update user profile
POST /memory/profiles

# Get user profile
GET /memory/profiles/{user_id}

# Update preferences
PUT /memory/profiles/{user_id}/preferences
```

### GDPR Compliance
```bash
# Export all user data
GET /memory/users/{user_id}/export

# Delete all user data
DELETE /memory/users/{user_id}
```

### Utilities
```bash
# Health check
GET /health

# System statistics
GET /stats

# API documentation
GET /docs
```

See API documentation at `http://localhost:8000/docs` when the server is running.

---

## Evaluation Results

### Key Metrics
| Metric | Value | Method |
|--------|-------|--------|
| **BERTScore F1** | TBD | Semantic similarity |
| **Hallucination Rate** | TBD | Context grounding |
| **Citation Accuracy** | TBD | Source verification |
| **Response Latency** | 2-6s | End-to-end |
| **Memory Retention** | TBD | Across sessions |

### Experiment Coverage
- Baseline knowledge assessment
- Anchored prompt evaluation
- RAG vs non-RAG comparison
- Long-session memory testing
- Multi-session profile consistency
- Adherence optimization (LinUCB)

Run experiments and analysis:
```bash
# Test pipeline without full dataset
python tests/test_unified_pipeline.py

# Analyze experiment results
python scripts/analyze_results.py --mode comprehensive

# Compare experiments
python scripts/compare_results.py --mode comprehensive
```

---

## Technology Stack

| Component | Technology | Version |
|-----------|-----------|---------|
| **Backend** | FastAPI | 0.115.5 |
| **Server** | Uvicorn | 0.32.1 |
| **LLM** | Anthropic Claude Opus 4.5 | Latest |
| **API Client** | Anthropic Python SDK | Latest |
| **Evaluation** | Custom compliance metrics | - |
| **Stats** | SciPy, Statsmodels | Latest |
| **Frontend** | Vanilla JS | - |
| **Logging** | Loguru | 0.7.3 |

---

## Configuration

Edit [configs/config.yaml](configs/config.yaml) to customize:

```yaml
# Model settings
models:
  llm:
    provider: "anthropic"
    model_name: "claude-opus-4-5-20251101"
    temperature: 0.7
    max_tokens: 1024

# Memory settings
memory:
  enabled: true
  max_history_turns: 20
  summarization:
    enabled: true
    trigger_length: 10
```

---

## Documentation

| Document | Description |
|----------|-------------|
| [UNIFIED_PIPELINE_GUIDE.md](UNIFIED_PIPELINE_GUIDE.md) | Unified pipeline architecture and usage |
| [UNIFIED_ANALYSIS_GUIDE.md](UNIFIED_ANALYSIS_GUIDE.md) | Analysis and comparison system guide |
| [ANNOTATION_CLEANUP_SUMMARY.md](ANNOTATION_CLEANUP_SUMMARY.md) | Annotation module documentation |
| [docs/QUICK_START.md](docs/QUICK_START.md) | Quick start guide |
| [docs/API_KEYS_SETUP.md](docs/API_KEYS_SETUP.md) | API keys configuration |
| [docs/GUIDELINE_SCOPE_AND_COMPLIANCE.md](docs/GUIDELINE_SCOPE_AND_COMPLIANCE.md) | WHO/BCS+ guideline scope |
| [docs/ANNOTATION_PROTOCOL.md](docs/ANNOTATION_PROTOCOL.md) | Compliance annotation protocol |
| [data/COMPLIANCE_DATASET_README.md](data/COMPLIANCE_DATASET_README.md) | Compliance test dataset documentation |

---

## Docker Deployment

```bash
# Build image
docker build -t contraception-counseling:latest .

# Run with Docker Compose
docker-compose up -d

# View logs
docker-compose logs -f

# Stop
docker-compose down
```

---

## Testing

```bash
# Test unified pipeline (4 experiments)
python tests/test_unified_pipeline.py

# Test unified analysis system
python tests/test_unified_analysis.py

# Run all tests
python -m pytest tests/ -v

# Run with coverage
pytest --cov=src tests/
```

---

## Monitoring

### Logging
- **Level**: INFO (configurable)
- **Format**: Structured JSON
- **Rotation**: 100 MB, 30 days retention
- **Location**: `results/logs/`

### Performance
- **LLM Generation**: 2-3s (Claude Opus 4.5 API)
- **Total Latency**: 2-3s (end-to-end)
- **Compliance Rate**: 76.25% (Claude), 85% (o3)
- **Critical Issues**: 0 (Claude), 0 (o3)

### Resource Usage
- **RAM**: ~4 GB (minimal local inference)
- **Disk**: ~2 GB (data + logs)
- **Network**: Anthropic API calls (~$0.015 per request)

---

## Status & Roadmap

### Completed
- Compliance-aware pipeline with WHO/BCS+ prompts
- Multi-language support (EN/FR/RW) via Claude Opus 4.5
- Memory management and user profiles
- Privacy-first data collection
- Compliance evaluation framework (Experiments 1-3)
- Web-based chat interface
- REST API with core endpoints
- Experiment runner with validation
- Code cleanup and restructuring

### Future Enhancements
- Cloud deployment guides (AWS, Azure, GCP)
- Mobile application (React Native)
- Advanced adherence support
- Real-time analytics dashboard
- A/B testing framework
- Model fine-tuning pipeline

---

## Academic Context

This is a **graduate-level capstone project** demonstrating:

- **Production-ready system**: FastAPI, Docker, comprehensive testing
- **Research rigor**: 6 experiments, statistical analysis, reproducibility
- **Privacy compliance**: GDPR, anonymous data, opt-in collection
- **Multilingual NLP**: Language-specific model routing
- **Evaluation depth**: BERTScore, hallucination detection, citation accuracy
- **Documentation**: Architecture diagrams, data flow, API specs

### Key Contributions
1. **Compliance-Aware AI for Medical Counseling**: Direct LLM generation with WHO-compliant system prompts (76.25% compliant, 0 critical issues)
2. **Multi-Language Healthcare AI**: Claude Opus 4.5 for consistent multilingual performance
3. **Privacy-Preserving Data Collection**: Anonymous, opt-in, GDPR-compliant
4. **Comprehensive Safety Evaluation**: Compliance scoring, critical issue detection, medical accuracy assessment
5. **Empirical Evidence Against RAG**: Demonstrated 35% degradation in compliance when using RAG retrieval

---

## Acknowledgments

### Data Sources
- **WHO Family Planning Handbook 2022** - Primary medical guidelines
- **BCS+ Toolkit** - Comprehensive counseling framework

### Technologies
- **Anthropic Claude Opus 4.5** - Primary LLM for all languages
- **FastAPI** - Modern Python web framework
- **Loguru** - Structured logging
- **Pydantic** - Data validation

---

## Contact

For questions or collaboration:
- Create an issue in the repository
- See documentation in [docs/](docs/) for detailed guides

---

## License

MIT License - See LICENSE file for details

---

## Citation

If you use this system in your research, please cite:

```bibtex
@software{ai_contraception_counseling_2025,
  title={AI Contraception Counseling System: Compliance-Aware Approach},
  author={[Your Name]},
  year={2025},
  url={https://github.com/[your-repo]},
  note={Demonstrates 41% improvement in compliance over RAG-based approaches}
}
```

---

**Built with ❤️ for improving global access to evidence-based contraception counseling**

**Status**: Production Ready (Compliance-Aware Approach)
**Last Updated**: December 9, 2025
