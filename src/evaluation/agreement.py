"""
Inter-Annotator Agreement Metrics

Calculate agreement between multiple annotators (human or LLM)
using Cohen's kappa and other metrics.
"""

import numpy as np
from typing import List, Dict, Tuple
from collections import Counter


class AgreementMetrics:
    """Calculate inter-annotator agreement metrics."""

    @staticmethod
    def cohens_kappa(annotations1: List[int], annotations2: List[int]) -> float:
        """
        Calculate Cohen's kappa for two annotators.

        Args:
            annotations1: First annotator's scores (list of 0/1/2)
            annotations2: Second annotator's scores (list of 0/1/2)

        Returns:
            Cohen's kappa value (-1 to 1)
            - < 0: Less than chance agreement
            - 0.01-0.20: Slight agreement
            - 0.21-0.40: Fair agreement
            - 0.41-0.60: Moderate agreement
            - 0.61-0.80: Substantial agreement
            - 0.81-1.00: Almost perfect agreement
        """
        if len(annotations1) != len(annotations2):
            raise ValueError("Annotation lists must have same length")

        if len(annotations1) == 0:
            return 0.0

        # Convert to numpy arrays
        a1 = np.array(annotations1)
        a2 = np.array(annotations2)

        # Observed agreement
        po = np.mean(a1 == a2)

        # Expected agreement by chance
        # Get unique categories
        categories = sorted(set(list(a1) + list(a2)))

        pe = 0
        for cat in categories:
            p1 = np.mean(a1 == cat)
            p2 = np.mean(a2 == cat)
            pe += p1 * p2

        # Cohen's kappa
        if pe == 1.0:
            return 1.0

        kappa = (po - pe) / (1 - pe)
        return kappa

    @staticmethod
    def fleiss_kappa(annotations: List[List[int]]) -> float:
        """
        Calculate Fleiss' kappa for multiple annotators.

        Args:
            annotations: List of annotation lists, one per annotator
                        Each list contains scores for the same items

        Returns:
            Fleiss' kappa value
        """
        # Convert to matrix: rows = items, cols = annotators
        n_items = len(annotations[0])
        n_annotators = len(annotations)

        # Validate
        for ann_list in annotations:
            if len(ann_list) != n_items:
                raise ValueError("All annotators must rate same number of items")

        # Get all unique categories
        all_ratings = [score for ann_list in annotations for score in ann_list]
        categories = sorted(set(all_ratings))
        n_categories = len(categories)

        # Build rating matrix
        rating_matrix = np.zeros((n_items, n_categories))

        for item_idx in range(n_items):
            for ann_list in annotations:
                score = ann_list[item_idx]
                cat_idx = categories.index(score)
                rating_matrix[item_idx, cat_idx] += 1

        # Calculate P_i (proportion of agreement for each item)
        P_i = np.sum(rating_matrix ** 2, axis=1) - n_annotators
        P_i = P_i / (n_annotators * (n_annotators - 1))

        # Mean proportion of agreement
        P_bar = np.mean(P_i)

        # Calculate P_j (proportion of ratings in each category)
        P_j = np.sum(rating_matrix, axis=0) / (n_items * n_annotators)

        # Expected agreement by chance
        P_e = np.sum(P_j ** 2)

        # Fleiss' kappa
        if P_e == 1.0:
            return 1.0

        kappa = (P_bar - P_e) / (1 - P_e)
        return kappa

    @staticmethod
    def percent_agreement(annotations1: List[int], annotations2: List[int]) -> float:
        """
        Calculate simple percent agreement.

        Args:
            annotations1: First annotator's scores
            annotations2: Second annotator's scores

        Returns:
            Percent agreement (0-100)
        """
        if len(annotations1) != len(annotations2):
            raise ValueError("Annotation lists must have same length")

        if len(annotations1) == 0:
            return 0.0

        a1 = np.array(annotations1)
        a2 = np.array(annotations2)

        return np.mean(a1 == a2) * 100

    @staticmethod
    def confusion_matrix(annotations1: List[int], annotations2: List[int]) -> np.ndarray:
        """
        Create confusion matrix between two annotators.

        Args:
            annotations1: First annotator's scores
            annotations2: Second annotator's scores

        Returns:
            Confusion matrix (numpy array)
        """
        if len(annotations1) != len(annotations2):
            raise ValueError("Annotation lists must have same length")

        # Get all categories
        categories = sorted(set(list(annotations1) + list(annotations2)))
        n_categories = len(categories)

        # Build confusion matrix
        matrix = np.zeros((n_categories, n_categories), dtype=int)

        for a1, a2 in zip(annotations1, annotations2):
            i = categories.index(a1)
            j = categories.index(a2)
            matrix[i, j] += 1

        return matrix

    @staticmethod
    def interpret_kappa(kappa: float) -> str:
        """
        Interpret Cohen's kappa value.

        Args:
            kappa: Kappa value

        Returns:
            Interpretation string
        """
        if kappa < 0:
            return "Less than chance agreement"
        elif kappa < 0.01:
            return "Slight agreement"
        elif kappa < 0.21:
            return "Fair agreement"
        elif kappa < 0.41:
            return "Moderate agreement"
        elif kappa < 0.61:
            return "Substantial agreement"
        elif kappa < 0.81:
            return "Almost perfect agreement"
        else:
            return "Perfect agreement"

    @staticmethod
    def calculate_agreement_metrics(
        annotations1: List[int],
        annotations2: List[int]
    ) -> Dict:
        """
        Calculate all agreement metrics.

        Args:
            annotations1: First annotator's scores
            annotations2: Second annotator's scores

        Returns:
            Dictionary with all metrics
        """
        kappa = AgreementMetrics.cohens_kappa(annotations1, annotations2)
        percent = AgreementMetrics.percent_agreement(annotations1, annotations2)
        confusion = AgreementMetrics.confusion_matrix(annotations1, annotations2)

        return {
            'cohens_kappa': kappa,
            'interpretation': AgreementMetrics.interpret_kappa(kappa),
            'percent_agreement': percent,
            'confusion_matrix': confusion.tolist(),
            'n_items': len(annotations1)
        }
