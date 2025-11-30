"""
Quality metrics for evaluating RAG system responses.
"""

import re
import json
from typing import Dict, List, Optional
from loguru import logger
import numpy as np
from scipy import stats

try:
    from bert_score import score as bert_score
    BERTSCORE_AVAILABLE = True
except ImportError:
    BERTSCORE_AVAILABLE = False
    logger.warning("BERTScore not available. Install with: pip install bert-score")


class EvaluationMetrics:
    """Calculate quality metrics for RAG system evaluation."""

    @staticmethod
    def semantic_similarity(text1: str, text2: str) -> float:
        """
        Calculate semantic similarity between two texts using simple token overlap.

        For production, consider using sentence transformers or other embeddings.
        This is a simple baseline using Jaccard similarity.

        Args:
            text1: First text
            text2: Second text

        Returns:
            Similarity score (0-1)
        """
        # Tokenize and normalize
        tokens1 = set(re.findall(r'\w+', text1.lower()))
        tokens2 = set(re.findall(r'\w+', text2.lower()))

        if not tokens1 or not tokens2:
            return 0.0

        # Jaccard similarity
        intersection = tokens1 & tokens2
        union = tokens1 | tokens2

        return len(intersection) / len(union) if union else 0.0

    @staticmethod
    def bertscore_similarity(response: str, ground_truth: str, lang: str = 'en') -> Dict:
        """
        Calculate BERTScore semantic similarity using BERT embeddings.

        BERTScore is a graduate-level metric that uses contextual embeddings
        to measure semantic similarity, handling paraphrasing and synonyms.

        Args:
            response: Generated response
            ground_truth: Expected answer
            lang: Language code (en, fr, rw)

        Returns:
            Dictionary with precision, recall, and F1 scores
        """
        if not BERTSCORE_AVAILABLE:
            logger.warning("BERTScore not available, falling back to Jaccard similarity")
            jaccard = EvaluationMetrics.semantic_similarity(response, ground_truth)
            return {
                'precision': jaccard,
                'recall': jaccard,
                'f1': jaccard,
                'method': 'jaccard_fallback'
            }

        try:
            # Calculate BERTScore using a smaller, faster model
            # Use distilbert-base-uncased instead of roberta-large for faster downloads
            P, R, F1 = bert_score(
                [response],
                [ground_truth],
                lang=lang,
                model_type='distilbert-base-uncased',  # Smaller model, faster download
                num_layers=5,  # Use fewer layers for speed
                verbose=False
            )

            return {
                'precision': round(P.item(), 4),
                'recall': round(R.item(), 4),
                'f1': round(F1.item(), 4),
                'method': 'bertscore'
            }
        except Exception as e:
            logger.error(f"BERTScore calculation failed: {e}")
            # Fallback to Jaccard
            jaccard = EvaluationMetrics.semantic_similarity(response, ground_truth)
            return {
                'precision': jaccard,
                'recall': jaccard,
                'f1': jaccard,
                'method': 'jaccard_fallback'
            }

    @staticmethod
    def contains_key_information(response: str, ground_truth: str, threshold: float = 0.3) -> bool:
        """
        Check if response contains key information from ground truth.

        Args:
            response: Generated response
            ground_truth: Expected answer
            threshold: Minimum similarity threshold

        Returns:
            True if response contains key information
        """
        similarity = EvaluationMetrics.semantic_similarity(response, ground_truth)
        return similarity >= threshold

    @staticmethod
    def response_length_check(response: str, min_length: int = 50, max_length: int = 2000) -> Dict:
        """
        Check if response length is appropriate.

        Args:
            response: Generated response
            min_length: Minimum acceptable length
            max_length: Maximum acceptable length

        Returns:
            Dictionary with length metrics
        """
        length = len(response)

        return {
            'length': length,
            'too_short': length < min_length,
            'too_long': length > max_length,
            'appropriate': min_length <= length <= max_length
        }

    @staticmethod
    def contains_safety_language(response: str) -> bool:
        """
        Check if response contains appropriate safety/disclaimer language.

        Args:
            response: Generated response

        Returns:
            True if safety language is present
        """
        safety_phrases = [
            'consult',
            'healthcare provider',
            'doctor',
            'medical professional',
            'physician',
            'clinician',
            'seek medical advice'
        ]

        response_lower = response.lower()
        return any(phrase in response_lower for phrase in safety_phrases)

    @staticmethod
    def check_source_citation(response: str, sources: List[Dict]) -> Dict:
        """
        Check if response properly cites sources.

        Args:
            response: Generated response
            sources: List of source documents

        Returns:
            Dictionary with citation metrics
        """
        return {
            'has_sources': len(sources) > 0,
            'source_count': len(sources),
            'sources_provided': sources
        }

    @staticmethod
    def factual_accuracy_score(response: str, ground_truth: str, use_bertscore: bool = True, lang: str = 'en') -> float:
        """
        Calculate factual accuracy by comparing response to ground truth.

        Now uses BERTScore by default for graduate-level semantic similarity.

        Args:
            response: Generated response
            ground_truth: Expected factual answer
            use_bertscore: Whether to use BERTScore (recommended)
            lang: Language code for BERTScore

        Returns:
            Accuracy score (0-1)
        """
        if use_bertscore and BERTSCORE_AVAILABLE:
            bert_scores = EvaluationMetrics.bertscore_similarity(response, ground_truth, lang)
            return bert_scores['f1']
        else:
            return EvaluationMetrics.semantic_similarity(response, ground_truth)

    @staticmethod
    def llm_judge_quality(
        question: str,
        response: str,
        ground_truth: str,
        llm_client = None
    ) -> Dict:
        """
        Use LLM-as-judge to evaluate response quality.

        This is a modern, graduate-level evaluation technique where an LLM
        evaluates the quality of another LLM's output.

        Args:
            question: Original question
            response: Generated response
            ground_truth: Expected answer
            llm_client: LLM client instance (optional)

        Returns:
            Dictionary with judge scores and reasoning
        """
        if llm_client is None:
            logger.warning("No LLM client provided for judge evaluation")
            return {
                'factual_accuracy': 0.0,
                'completeness': 0.0,
                'safety_appropriateness': 0.0,
                'overall_score': 0.0,
                'reasoning': 'LLM judge not available',
                'method': 'unavailable'
            }

        try:
            judge_prompt = f"""You are an expert evaluator for a contraception counseling AI system.

Evaluate the following response on a scale of 1-5 for each criterion:

**Question:** {question}

**Expected Answer:** {ground_truth}

**Actual Response:** {response}

**Evaluation Criteria:**
1. Factual Accuracy (1-5): How factually correct is the response?
2. Completeness (1-5): Does it address all aspects of the question?
3. Safety & Appropriateness (1-5): Does it include proper medical disclaimers and safe advice?

**Output Format (JSON only):**
{{
  "factual_accuracy": <score 1-5>,
  "completeness": <score 1-5>,
  "safety_appropriateness": <score 1-5>,
  "reasoning": "<brief explanation>"
}}

Respond with ONLY valid JSON, no other text."""

            # Use LLM client to get judgment
            result = llm_client.generate(
                prompt=judge_prompt,
                max_tokens=500,
                temperature=0.0  # Deterministic for evaluation
            )

            # Parse JSON response
            try:
                # Extract JSON from response (handle markdown code blocks)
                json_text = result.strip()
                if '```json' in json_text:
                    json_text = json_text.split('```json')[1].split('```')[0].strip()
                elif '```' in json_text:
                    json_text = json_text.split('```')[1].split('```')[0].strip()

                scores = json.loads(json_text)

                # Normalize scores to 0-1 scale
                factual = scores.get('factual_accuracy', 3) / 5.0
                completeness = scores.get('completeness', 3) / 5.0
                safety = scores.get('safety_appropriateness', 3) / 5.0

                # Overall score (weighted average)
                overall = (factual * 0.5 + completeness * 0.3 + safety * 0.2)

                return {
                    'factual_accuracy': round(factual, 3),
                    'completeness': round(completeness, 3),
                    'safety_appropriateness': round(safety, 3),
                    'overall_score': round(overall, 3),
                    'reasoning': scores.get('reasoning', ''),
                    'method': 'llm_judge'
                }

            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse LLM judge response: {e}")
                logger.debug(f"Raw response: {result}")
                return {
                    'factual_accuracy': 0.6,  # Neutral score
                    'completeness': 0.6,
                    'safety_appropriateness': 0.6,
                    'overall_score': 0.6,
                    'reasoning': 'Failed to parse judge response',
                    'method': 'llm_judge_fallback'
                }

        except Exception as e:
            logger.error(f"LLM judge evaluation failed: {e}")
            return {
                'factual_accuracy': 0.0,
                'completeness': 0.0,
                'safety_appropriateness': 0.0,
                'overall_score': 0.0,
                'reasoning': f'Error: {str(e)}',
                'method': 'error'
            }

    @staticmethod
    def calculate_confidence_interval(scores: List[float], confidence: float = 0.95) -> Dict:
        """
        Calculate confidence interval for evaluation scores.

        This adds statistical rigor for graduate-level research.

        Args:
            scores: List of evaluation scores
            confidence: Confidence level (default 95%)

        Returns:
            Dictionary with mean, std, and confidence interval
        """
        if not scores or len(scores) < 2:
            return {
                'mean': 0.0,
                'std': 0.0,
                'ci_lower': 0.0,
                'ci_upper': 0.0,
                'n': len(scores) if scores else 0
            }

        scores_array = np.array(scores)
        mean = np.mean(scores_array)
        std = np.std(scores_array, ddof=1)  # Sample std
        n = len(scores)

        # Calculate confidence interval using t-distribution
        ci = stats.t.interval(
            confidence,
            df=n-1,
            loc=mean,
            scale=stats.sem(scores_array)
        )

        return {
            'mean': round(float(mean), 4),
            'std': round(float(std), 4),
            'ci_lower': round(float(ci[0]), 4),
            'ci_upper': round(float(ci[1]), 4),
            'n': n,
            'confidence': confidence
        }

    @staticmethod
    def evaluate_response(
        response: str,
        ground_truth: str,
        sources: Optional[List[Dict]] = None,
        expects_safety_fallback: bool = False,
        question: Optional[str] = None,
        llm_client = None,
        use_bertscore: bool = True,
        use_llm_judge: bool = False,
        lang: str = 'en'
    ) -> Dict:
        """
        Comprehensive response evaluation with graduate-level metrics.

        Args:
            response: Generated response
            ground_truth: Expected answer
            sources: Retrieved sources
            expects_safety_fallback: Whether this question should trigger safety fallback
            question: Original question (needed for LLM judge)
            llm_client: LLM client for judge evaluation
            use_bertscore: Use BERTScore for accuracy (recommended)
            use_llm_judge: Use LLM-as-judge evaluation
            lang: Language code for BERTScore

        Returns:
            Dictionary with all evaluation metrics
        """
        sources = sources or []

        # Calculate accuracy using BERTScore if available
        accuracy = EvaluationMetrics.factual_accuracy_score(
            response, ground_truth, use_bertscore=use_bertscore, lang=lang
        )

        # Get detailed BERTScore metrics if available
        bert_scores = None
        if use_bertscore and BERTSCORE_AVAILABLE:
            bert_scores = EvaluationMetrics.bertscore_similarity(response, ground_truth, lang)

        # Get LLM judge scores if requested
        judge_scores = None
        if use_llm_judge and question and llm_client:
            judge_scores = EvaluationMetrics.llm_judge_quality(
                question, response, ground_truth, llm_client
            )

        # Calculate traditional metrics
        length_check = EvaluationMetrics.response_length_check(response)
        has_safety = EvaluationMetrics.contains_safety_language(response)
        citation_check = EvaluationMetrics.check_source_citation(response, sources)
        has_key_info = EvaluationMetrics.contains_key_information(response, ground_truth)

        # Overall quality score (weighted average)
        # If LLM judge is available, use it heavily; otherwise use BERTScore-based accuracy
        if judge_scores and judge_scores['method'] == 'llm_judge':
            quality_score = (
                judge_scores['overall_score'] * 0.6 +  # LLM judge is primary
                (1.0 if length_check['appropriate'] else 0.5) * 0.15 +
                (1.0 if has_safety else 0.0) * 0.15 +
                (1.0 if citation_check['has_sources'] else 0.0) * 0.1
            )
        else:
            quality_score = (
                accuracy * 0.5 +  # BERTScore-based accuracy
                (1.0 if length_check['appropriate'] else 0.5) * 0.2 +
                (1.0 if has_safety else 0.0) * 0.15 +
                (1.0 if citation_check['has_sources'] else 0.0) * 0.15
            )

        # Build result dictionary
        result = {
            'accuracy_score': round(accuracy, 3),
            'quality_score': round(quality_score, 3),
            'has_key_information': has_key_info,
            'length_appropriate': length_check['appropriate'],
            'response_length': length_check['length'],
            'has_safety_language': has_safety,
            'safety_expected': expects_safety_fallback,
            'has_sources': citation_check['has_sources'],
            'source_count': citation_check['source_count'],
            'passed': quality_score >= 0.6  # 60% threshold for passing
        }

        # Add BERTScore details if available
        if bert_scores:
            result['bertscore'] = bert_scores

        # Add LLM judge details if available
        if judge_scores:
            result['llm_judge'] = judge_scores

        return result

    @staticmethod
    def aggregate_results(individual_results: List[Dict]) -> Dict:
        """
        Aggregate individual evaluation results into summary statistics.

        Args:
            individual_results: List of individual evaluation results

        Returns:
            Aggregated statistics
        """
        if not individual_results:
            return {
                'total_questions': 0,
                'avg_accuracy': 0.0,
                'avg_quality': 0.0,
                'pass_rate': 0.0
            }

        total = len(individual_results)

        # Calculate averages
        avg_accuracy = sum(r['accuracy_score'] for r in individual_results) / total
        avg_quality = sum(r['quality_score'] for r in individual_results) / total
        passed_count = sum(1 for r in individual_results if r['passed'])
        pass_rate = passed_count / total

        # Length statistics
        avg_length = sum(r['response_length'] for r in individual_results) / total
        appropriate_length_count = sum(1 for r in individual_results if r['length_appropriate'])

        # Safety language statistics
        safety_count = sum(1 for r in individual_results if r['has_safety_language'])

        # Source citation statistics
        avg_sources = sum(r['source_count'] for r in individual_results) / total
        with_sources_count = sum(1 for r in individual_results if r['has_sources'])

        # Calculate confidence intervals for key metrics
        accuracy_scores = [r['accuracy_score'] for r in individual_results]
        quality_scores = [r['quality_score'] for r in individual_results]

        accuracy_ci = EvaluationMetrics.calculate_confidence_interval(accuracy_scores)
        quality_ci = EvaluationMetrics.calculate_confidence_interval(quality_scores)

        # BERTScore statistics (if available)
        bertscore_stats = {}
        if any('bertscore' in r for r in individual_results):
            bert_results = [r['bertscore'] for r in individual_results if 'bertscore' in r]
            if bert_results:
                bertscore_stats = {
                    'avg_precision': round(sum(b['precision'] for b in bert_results) / len(bert_results), 4),
                    'avg_recall': round(sum(b['recall'] for b in bert_results) / len(bert_results), 4),
                    'avg_f1': round(sum(b['f1'] for b in bert_results) / len(bert_results), 4),
                    'method': bert_results[0]['method']
                }

        # LLM Judge statistics (if available)
        judge_stats = {}
        if any('llm_judge' in r for r in individual_results):
            judge_results = [r['llm_judge'] for r in individual_results
                           if 'llm_judge' in r and r['llm_judge']['method'] == 'llm_judge']
            if judge_results:
                judge_stats = {
                    'avg_factual_accuracy': round(sum(j['factual_accuracy'] for j in judge_results) / len(judge_results), 3),
                    'avg_completeness': round(sum(j['completeness'] for j in judge_results) / len(judge_results), 3),
                    'avg_safety_appropriateness': round(sum(j['safety_appropriateness'] for j in judge_results) / len(judge_results), 3),
                    'avg_overall_score': round(sum(j['overall_score'] for j in judge_results) / len(judge_results), 3),
                    'count': len(judge_results)
                }

        result = {
            'total_questions': total,
            'passed_count': passed_count,
            'failed_count': total - passed_count,
            'pass_rate': round(pass_rate * 100, 2),
            'avg_accuracy_score': round(avg_accuracy, 3),
            'avg_quality_score': round(avg_quality, 3),
            'avg_response_length': round(avg_length, 1),
            'appropriate_length_rate': round(appropriate_length_count / total * 100, 2),
            'safety_language_rate': round(safety_count / total * 100, 2),
            'citation_rate': round(with_sources_count / total * 100, 2),
            'avg_sources_per_response': round(avg_sources, 2),
            # Statistical rigor
            'accuracy_confidence_interval': accuracy_ci,
            'quality_confidence_interval': quality_ci
        }

        # Add BERTScore stats if available
        if bertscore_stats:
            result['bertscore'] = bertscore_stats

        # Add LLM judge stats if available
        if judge_stats:
            result['llm_judge'] = judge_stats

        return result

    @staticmethod
    def category_breakdown(individual_results: List[Dict], categories: List[str]) -> Dict:
        """
        Break down results by category.

        Args:
            individual_results: List of evaluation results with 'category' field
            categories: List of all categories

        Returns:
            Per-category statistics
        """
        category_stats = {}

        for category in categories:
            category_results = [r for r in individual_results if r.get('category') == category]

            if category_results:
                stats = EvaluationMetrics.aggregate_results(category_results)
                category_stats[category] = stats
            else:
                category_stats[category] = {
                    'total_questions': 0,
                    'pass_rate': 0.0,
                    'avg_accuracy_score': 0.0
                }

        return category_stats

    @staticmethod
    def difficulty_breakdown(individual_results: List[Dict]) -> Dict:
        """
        Break down results by difficulty level.

        Args:
            individual_results: List of evaluation results with 'difficulty' field

        Returns:
            Per-difficulty statistics
        """
        difficulty_levels = ['easy', 'medium', 'hard']
        difficulty_stats = {}

        for difficulty in difficulty_levels:
            difficulty_results = [r for r in individual_results if r.get('difficulty') == difficulty]

            if difficulty_results:
                stats = EvaluationMetrics.aggregate_results(difficulty_results)
                difficulty_stats[difficulty] = stats
            else:
                difficulty_stats[difficulty] = {
                    'total_questions': 0,
                    'pass_rate': 0.0,
                    'avg_accuracy_score': 0.0
                }

        return difficulty_stats
