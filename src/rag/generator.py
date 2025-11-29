"""
RAG Generator Component

Generates responses using LLM and retrieved context.
"""

from typing import Dict, List, Optional
from loguru import logger

from src.utils.llm_client import LLMClient


class RAGGenerator:
    """
    Generates counseling responses using LLM.

    Features:
    - Context-aware response generation
    - Multi-language support
    - Safety fallbacks
    - Citation extraction
    """

    # System prompts for different languages
    SYSTEM_PROMPTS = {
        'english': """You are a professional contraception counselor. Your role is to provide accurate, evidence-based information about family planning methods.

Guidelines:
- Use the provided context from WHO and BCS+ guidelines
- Be empathetic and non-judgmental
- Provide balanced information about benefits and risks
- Respect user autonomy and choices
- If unsure, recommend consulting a healthcare provider
- Never provide medical diagnoses or treatment decisions

Format your response clearly and cite sources when possible.""",

        'french': """Vous êtes un conseiller professionnel en contraception. Votre rôle est de fournir des informations précises et fondées sur des preuves concernant les méthodes de planification familiale.

Directives:
- Utilisez le contexte fourni par les directives de l'OMS et BCS+
- Soyez empathique et sans jugement
- Fournissez des informations équilibrées sur les avantages et les risques
- Respectez l'autonomie et les choix de l'utilisateur
- En cas de doute, recommandez de consulter un professionnel de santé
- Ne fournissez jamais de diagnostics médicaux ou de décisions de traitement

Formatez votre réponse clairement et citez les sources si possible.""",

        'kinyarwanda': """Uri umujyanama wabigize umwuga mu by'kuboneza urubyaro. Uruhare rwawe ni ugutanga amakuru yizewe, yashingiye ku bimenyetso bijyanye n'uburyo bwo gutegura urubyaro.

Amabwiriza:
- Koresha imiterere yatanzwe n'amabwiriza ya WHO na BCS+
- Ba impuhwe kandi ntumaganye
- Tanga amakuru aringaniye ku byiza n'ingaruka
- Ubahiriza ubwigenge n'amahitamo y'ukoresha
- Niba utagishidikanya, saba gusaba inama ku muganga
- Ntuhe isuzuma ry'ubuvuzi cyangwa ibyemezo by'ubuvuzi

Hindura igisubizo cyawe neza kandi uvuge inkomoko iyo bishoboka."""
    }

    # Response templates for safety fallbacks
    SAFETY_FALLBACKS = {
        'english': """I don't have enough information in the guidelines to answer that question confidently.

I recommend:
1. Consulting a healthcare provider for personalized advice
2. Visiting a family planning clinic
3. Asking a more specific question about contraception methods

Is there a specific contraception method you'd like to know more about?""",

        'french': """Je n'ai pas suffisamment d'informations dans les directives pour répondre à cette question avec confiance.

Je recommande:
1. Consulter un professionnel de santé pour des conseils personnalisés
2. Visiter une clinique de planification familiale
3. Poser une question plus spécifique sur les méthodes de contraception

Y a-t-il une méthode de contraception spécifique sur laquelle vous aimeriez en savoir plus?""",

        'kinyarwanda': """Nta makuru ahagije mfite mu mabwiriza kugira ngo nsubize icyo kibazo nizeye.

Ndasaba:
1. Gusaba inama ku muganga kugira ngo ubone inama zihariye
2. Gusura ikigo cy'ubuvuzi bwo gutegura urubyaro
3. Kubaza ikibazo kigaragara cyane ku buryo bwo kuboneza urubyaro

Hari uburyo bwo kuboneza urubyaro ushaka kumenya byinshi?"""
    }

    def __init__(self, llm_config: Dict):
        """
        Initialize generator.

        Args:
            llm_config: LLM configuration (provider, model, etc.)
        """
        self.llm_client = LLMClient(llm_config)
        logger.info(f"RAGGenerator initialized with {llm_config['provider']} provider")

    def _build_prompt(
        self,
        query: str,
        context: str,
        language: str = 'english',
        conversation_history: Optional[List[Dict]] = None
    ) -> str:
        """
        Build prompt for LLM.

        Args:
            query: User's question
            context: Retrieved context from guidelines
            language: Response language
            conversation_history: Previous conversation turns

        Returns:
            Formatted prompt
        """
        # Get system prompt
        system_prompt = self.SYSTEM_PROMPTS.get(
            language.lower(),
            self.SYSTEM_PROMPTS['english']
        )

        # Build context section
        context_section = f"""CONTEXT FROM GUIDELINES:
{context}

---"""

        # Build conversation history
        history_section = ""
        if conversation_history:
            history_parts = []
            for turn in conversation_history[-3:]:  # Last 3 turns
                history_parts.append(f"User: {turn['query']}")
                history_parts.append(f"Assistant: {turn['response']}")

            if history_parts:
                history_section = "CONVERSATION HISTORY:\n" + "\n".join(history_parts) + "\n\n---\n\n"

        # Build query section
        query_section = f"""CURRENT QUESTION:
{query}

Please provide a helpful, accurate response based on the context above. If the context doesn't contain relevant information, use the safety fallback guidance."""

        # Combine all sections
        full_prompt = f"""{context_section}

{history_section}{query_section}"""

        return full_prompt

    def generate(
        self,
        query: str,
        context: str,
        language: str = 'english',
        conversation_history: Optional[List[Dict]] = None,
        temperature: float = 0.7,
        max_tokens: int = 500
    ) -> Dict:
        """
        Generate response using LLM.

        Args:
            query: User's question
            context: Retrieved context
            language: Response language
            conversation_history: Previous conversation
            temperature: LLM temperature (0-1)
            max_tokens: Maximum response length

        Returns:
            Dictionary with response, sources, and metadata
        """
        logger.debug(f"Generating response for query in {language}")

        # Get system prompt
        system_prompt = self.SYSTEM_PROMPTS.get(
            language.lower(),
            self.SYSTEM_PROMPTS['english']
        )

        # Build prompt
        prompt = self._build_prompt(
            query=query,
            context=context,
            language=language,
            conversation_history=conversation_history
        )

        # Generate response
        try:
            response = self.llm_client.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens
            )

            # Check if response is empty or too short (potential failure)
            if not response or len(response.strip()) < 20:
                logger.warning("LLM returned empty/short response, using fallback")
                response = self._get_safety_fallback(language)

            # Extract citations (sources mentioned in response)
            citations = self._extract_citations(response)

            result = {
                'response': response.strip(),
                'citations': citations,
                'language': language,
                'metadata': {
                    'temperature': temperature,
                    'max_tokens': max_tokens,
                    'context_length': len(context)
                }
            }

            logger.debug(f"Generated response with {len(response)} chars, {len(citations)} citations")
            return result

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {
                'response': self._get_safety_fallback(language),
                'citations': [],
                'language': language,
                'metadata': {'error': str(e)}
            }

    def _get_safety_fallback(self, language: str) -> str:
        """
        Get safety fallback response.

        Args:
            language: Response language

        Returns:
            Safety fallback message
        """
        return self.SAFETY_FALLBACKS.get(
            language.lower(),
            self.SAFETY_FALLBACKS['english']
        )

    def _extract_citations(self, response: str) -> List[str]:
        """
        Extract source citations from response.

        Args:
            response: Generated response

        Returns:
            List of cited sources
        """
        citations = []

        # Look for citation patterns like [Source 1: ...]
        import re
        pattern = r'\[Source \d+: ([^\]]+)\]'
        matches = re.findall(pattern, response)

        citations.extend(matches)

        return citations

    def validate_response(self, response: str, query: str) -> Dict:
        """
        Validate response quality.

        Args:
            response: Generated response
            query: Original query

        Returns:
            Validation metrics
        """
        validation = {
            'is_valid': True,
            'warnings': [],
            'score': 1.0
        }

        # Check minimum length
        if len(response) < 50:
            validation['warnings'].append("Response too short")
            validation['score'] -= 0.3

        # Check for safety keywords
        safety_keywords = ['healthcare provider', 'doctor', 'clinic', 'professionnel de santé', 'muganga']
        has_safety = any(keyword.lower() in response.lower() for keyword in safety_keywords)

        # Check for medical disclaimers
        disclaimer_keywords = ['recommend consulting', 'suggest consulting', 'should consult', 'consulter']
        has_disclaimer = any(keyword.lower() in response.lower() for keyword in disclaimer_keywords)

        # Check for harmful advice patterns
        harmful_patterns = [
            'definitely',
            'certainly will',
            'guaranteed',
            'you must',
            'you should definitely'
        ]
        has_harmful = any(pattern.lower() in response.lower() for pattern in harmful_patterns)

        if has_harmful:
            validation['warnings'].append("Response contains overly definitive language")
            validation['score'] -= 0.2

        # Check if response addresses query
        query_terms = set(query.lower().split())
        response_terms = set(response.lower().split())
        overlap = len(query_terms & response_terms) / len(query_terms) if query_terms else 0

        if overlap < 0.2:
            validation['warnings'].append("Response may not address query")
            validation['score'] -= 0.2

        # Overall validity
        validation['is_valid'] = validation['score'] >= 0.5

        return validation


class MultilingualGenerator(RAGGenerator):
    """
    Enhanced generator with automatic language detection.
    """

    def detect_language(self, text: str) -> str:
        """
        Detect language of input text.

        Args:
            text: Input text

        Returns:
            Detected language code
        """
        # Simple keyword-based detection
        # Can be enhanced with langdetect library

        french_keywords = ['bonjour', 'contraception', 'méthode', 'comment', 'pourquoi']
        kinyarwanda_keywords = ['mwaramutse', 'kuboneza', 'uburyo', 'gute', 'kuki']

        text_lower = text.lower()

        # Count language-specific keywords
        french_count = sum(1 for kw in french_keywords if kw in text_lower)
        kinyarwanda_count = sum(1 for kw in kinyarwanda_keywords if kw in text_lower)

        if french_count > 0:
            return 'french'
        elif kinyarwanda_count > 0:
            return 'kinyarwanda'
        else:
            return 'english'  # Default

    def generate(
        self,
        query: str,
        context: str,
        language: Optional[str] = None,
        **kwargs
    ) -> Dict:
        """
        Generate with automatic language detection.

        Args:
            query: User's question
            context: Retrieved context
            language: Response language (auto-detected if None)
            **kwargs: Additional arguments

        Returns:
            Response dictionary
        """
        # Auto-detect language if not provided
        if language is None:
            language = self.detect_language(query)
            logger.debug(f"Detected language: {language}")

        return super().generate(
            query=query,
            context=context,
            language=language,
            **kwargs
        )


# Example usage
if __name__ == "__main__":
    from src.utils.logger import setup_logger
    import yaml

    # Setup logging
    logger = setup_logger()

    # Load config
    with open("configs/config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    # Initialize generator
    generator = RAGGenerator(llm_config=config['models']['llm'])

    # Example context
    context = """[Source 1: WHO FP Handbook 2022, Page 45]
DMPA (Depo-Provera) is a progestin-only injectable contraceptive given every 3 months.

Common side effects:
- Irregular bleeding or spotting
- Weight gain (average 2-3 kg)
- Headaches
- Mood changes

Benefits:
- Highly effective (99% with perfect use)
- Private method
- Does not interfere with sex
- May reduce menstrual cramps

[Source 2: BCS+ Toolkit, Page 12]
DMPA is safe for most women, including breastfeeding mothers."""

    # Example query
    query = "What are the side effects of DMPA injection?"

    # Generate response
    result = generator.generate(
        query=query,
        context=context,
        language='english'
    )

    print(f"\nQuery: {query}")
    print(f"\nResponse:\n{result['response']}")
    print(f"\nCitations: {result['citations']}")
    print(f"\nMetadata: {result['metadata']}")

    # Validate response
    validation = generator.validate_response(result['response'], query)
    print(f"\nValidation:")
    print(f"  Valid: {validation['is_valid']}")
    print(f"  Score: {validation['score']}")
    print(f"  Warnings: {validation['warnings']}")
