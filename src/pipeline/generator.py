"""
Compliance-Aware Generator Component

Generates responses using LLM with compliance-aware prompting.
Based on Experiment 2 approach (no RAG retrieval).
"""

from typing import Dict, List, Optional
import re
from loguru import logger

from src.utils.multilang_llm_client import MultiLanguageLLMClient


class SafetyValidator:
    """
    Lightweight post-generation safety validator.

    Performs simple pattern-based checks to catch unsafe language patterns
    without the performance degradation of RAG-based constraints.

    Based on "Trust but Verify" approach:
    - Trust compliance-aware prompts for primary safety (76.25% compliant)
    - Verify with lightweight checks for obvious unsafe patterns
    - Log concerning responses without blocking (avoid false positives)
    """

    # Unsafe definitive language patterns (multi-language)
    UNSAFE_PATTERNS = {
        'english': [
            r'definitely will not get pregnant',
            r'definitely won\'?t get pregnant',
            r'100% effective',
            r'guaranteed to (work|prevent)',
            r'never (need|have) to (see|consult)',
            r'don\'?t need (to see|a) (doctor|healthcare)',
            r'stop taking.*without',
            r'must use',
            r'you should definitely',
        ],
        'french': [
            r'certainement ne tombera pas enceinte',
            r'100% efficace',
            r'garanti de',
            r'jamais besoin de (voir|consulter)',
            r'pas besoin de (voir|consulter)',
        ],
        'kinyarwanda': [
            r'nta mahirwe yo gutwita',
            r'100% inoze',
        ]
    }

    # Medical context keywords (questions requiring healthcare provider mention)
    MEDICAL_KEYWORDS = {
        'english': ['medical condition', 'disease', 'medication', 'pregnant', 'breastfeeding',
                   'side effect', 'bleeding', 'pain', 'illness', 'surgery'],
        'french': ['maladie', 'médicament', 'enceinte', 'allaitement', 'effet secondaire',
                  'saignement', 'douleur'],
        'kinyarwanda': ['indwara', 'imiti', 'inda', 'konsa', 'amaraso']
    }

    # Required safety disclaimers
    SAFETY_DISCLAIMERS = {
        'english': [r'consult.*provider', r'see.*doctor', r'healthcare provider',
                   r'speak with.*provider', r'talk to.*doctor'],
        'french': [r'consulter.*professionnel', r'voir.*médecin', r'parler.*médecin'],
        'kinyarwanda': [r'ugomba.*muganga', r'baza.*muganga']
    }

    def __init__(self):
        """Initialize safety validator."""
        logger.info("SafetyValidator initialized (lightweight post-generation checks)")

    def validate(self, response: str, query: str, language: str = 'english') -> Dict:
        """
        Validate response for safety issues.

        Args:
            response: Generated response text
            query: User's question
            language: Response language

        Returns:
            Validation result with safety flags and recommendations
        """
        issues = []
        severity = 'low'

        # Check 1: Unsafe definitive language
        unsafe_found = self._check_unsafe_patterns(response, language)
        if unsafe_found:
            issues.extend(unsafe_found)
            severity = 'high'

        # Check 2: Medical questions missing provider disclaimer
        if self._is_medical_question(query, language):
            if not self._has_provider_disclaimer(response, language):
                issues.append("Medical question missing healthcare provider recommendation")
                severity = 'medium' if severity == 'low' else severity

        # Check 3: Response too short (potential failure)
        if len(response.strip()) < 50:
            issues.append("Response unusually short (possible generation failure)")
            severity = 'medium' if severity == 'low' else severity

        return {
            'is_safe': len(issues) == 0,
            'issues': issues,
            'severity': severity,
            'requires_disclaimer': self._is_medical_question(query, language) and not self._has_provider_disclaimer(response, language)
        }

    def _check_unsafe_patterns(self, response: str, language: str) -> List[str]:
        """Check for unsafe definitive language patterns."""
        issues = []
        patterns = self.UNSAFE_PATTERNS.get(language, self.UNSAFE_PATTERNS['english'])

        for pattern in patterns:
            if re.search(pattern, response, re.IGNORECASE):
                issues.append(f"Unsafe definitive language detected: '{pattern}'")

        return issues

    def _is_medical_question(self, query: str, language: str) -> bool:
        """Check if query is medical in nature (requires healthcare disclaimer)."""
        keywords = self.MEDICAL_KEYWORDS.get(language, self.MEDICAL_KEYWORDS['english'])
        query_lower = query.lower()

        return any(keyword in query_lower for keyword in keywords)

    def _has_provider_disclaimer(self, response: str, language: str) -> bool:
        """Check if response includes healthcare provider recommendation."""
        disclaimers = self.SAFETY_DISCLAIMERS.get(language, self.SAFETY_DISCLAIMERS['english'])

        return any(re.search(pattern, response, re.IGNORECASE) for pattern in disclaimers)

    def add_disclaimer(self, response: str, language: str = 'english') -> str:
        """
        Add safety disclaimer to response.

        Args:
            response: Original response
            language: Response language

        Returns:
            Response with disclaimer added
        """
        disclaimers = {
            'english': "\n\n⚠️ **Note:** For personalized medical advice, please consult a qualified healthcare provider.",
            'french': "\n\n⚠️ **Note:** Pour des conseils médicaux personnalisés, veuillez consulter un professionnel de santé qualifié.",
            'kinyarwanda': "\n\n⚠️ **Icyitonderwa:** Kugira ngo ubone inama z'ubuvuzi zihariye, nyamuneka baza muganga."
        }

        disclaimer = disclaimers.get(language, disclaimers['english'])
        return response + disclaimer


class ComplianceGenerator:
    """
    Generates counseling responses using LLM with compliance-aware prompting.

    Features:
    - Compliance-aware response generation
    - Multi-language support
    - Safety fallbacks
    - Citation extraction
    """

    # System prompts for different languages
    # Based on Experiment 2 (Compliance-Aware Prompting) which achieved:
    # - Claude Opus 4.5: 76.25% compliant, 0 critical issues
    # - o3: 85% compliant, 0 critical issues
    SYSTEM_PROMPTS = {
        'english': """You are a family planning counselor assistant. Your answers MUST be strictly consistent with the 2022 WHO Family Planning Global Handbook, WHO Medical Eligibility Criteria (MEC), and the Balanced Counseling Strategy (BCS+) for family planning.

Guidelines:
- Follow WHO MEC categories (1-4) for medical eligibility
- Provide accurate effectiveness rates (typical use and perfect use)
- Use non-directive, non-judgmental counseling language
- Respect client autonomy and support informed choice
- Mention side effects, warning signs, and when to seek care
- If you are not sure about something, say you don't know and advise the user to consult a healthcare provider rather than guessing.

For Rwanda-specific questions, follow Rwanda National Family Planning Policy (2021), including:
- Adolescents can access FP without parental consent
- All women have equal access regardless of marital status
- Services are free or low-cost at public facilities
- Respect cultural context and use accessible language""",

        'french': """Vous êtes un assistant conseiller en planification familiale. Vos réponses DOIVENT être strictement cohérentes avec le Manuel mondial de planification familiale de l'OMS 2022, les critères d'éligibilité médicale (MEC) de l'OMS et la stratégie de conseil équilibré (BCS+) pour la planification familiale.

Directives:
- Suivez les catégories MEC de l'OMS (1-4) pour l'éligibilité médicale
- Fournissez des taux d'efficacité précis (utilisation typique et utilisation parfaite)
- Utilisez un langage de conseil non directif et sans jugement
- Respectez l'autonomie du client et soutenez le choix éclairé
- Mentionnez les effets secondaires, les signes d'alerte et quand consulter
- Si vous n'êtes pas sûr de quelque chose, dites que vous ne savez pas et conseillez à l'utilisateur de consulter un professionnel de santé plutôt que de deviner.

Pour les questions spécifiques au Rwanda, suivez la Politique nationale de planification familiale du Rwanda (2021), y compris:
- Les adolescents peuvent accéder à la PF sans consentement parental
- Toutes les femmes ont un accès égal quel que soit leur statut matrimonial
- Les services sont gratuits ou à faible coût dans les établissements publics
- Respectez le contexte culturel et utilisez un langage accessible""",

        'kinyarwanda': """Uri umufasha w'umujyanama mu gutegura urubyaro. Ibisubizo byawe BIGOMBA kuba byuzuye n'Igitabo cy'Isi yose cyo Gutegura Urubyaro cya WHO 2022, Ibipimo by'Ubuzima bwo Kwemererwa (MEC) bya WHO, n'Ingamba zo Kugira Inama Ziringaniye (BCS+) zo gutegura urubyaro.

Amabwiriza:
- Kurikiza ibyiciro bya WHO MEC (1-4) ku bwemerewe bw'ubuvuzi
- Tanga igipimo cy'imikorere nyayo (ikoreshwa risanzwe n'ikoreshwa ryuzuye)
- Koresha imvugo yo kugira inama idahuza abantu kandi itagira icyiza
- Ubahirize ubwigenge bw'umukiriya kandi ushyigikire guhitamo kubimenya
- Vuga ingaruka z'ingaruka, ibimenyetso byo kuburira, n'igihe cyo gusaba ubufasha
- Niba utagishidikanya ku kintu, vuga ko utazi kandi ugire inama umukoresha gusaba ubufasha ku muganga aho gutekereza.

Ku bibazo byihariye by'u Rwanda, kurikiza Politiki y'Igihugu yo Gutegura Urubyaro y'u Rwanda (2021), harimo:
- Abangavu bashobora kubona serivisi zo gutegura urubyaro nta kwemera kw'ababyeyi
- Abagore bose bafite uburenganzira bungana kutitaye ku buryo bw'ubukwe
- Serivisi ni ubuntu cyangwa ihendutse mu bigo by'ubuvuzi bya Leta
- Kubahiriza imico n'imvugo yoroshye"""
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

    def __init__(self, llm_config: Dict, enable_safety_validation: bool = True):
        """
        Initialize generator.

        Args:
            llm_config: LLM configuration (provider, model, etc.)
            enable_safety_validation: Enable post-generation safety checks
        """
        self.llm_client = MultiLanguageLLMClient(llm_config)
        self.enable_safety_validation = enable_safety_validation
        self.safety_validator = SafetyValidator() if enable_safety_validation else None

        logger.info(f"ComplianceGenerator initialized with multi-language routing: {MultiLanguageLLMClient.LANGUAGE_MODELS}")
        if enable_safety_validation:
            logger.info("✓ Safety validation enabled (post-generation checks)")

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
            context: Retrieved context from guidelines (optional, for reference only)
            language: Response language
            conversation_history: Previous conversation turns

        Returns:
            Formatted prompt
        """
        # Build conversation history
        history_section = ""
        if conversation_history:
            history_parts = []
            for turn in conversation_history[-3:]:  # Last 3 turns
                history_parts.append(f"User: {turn['query']}")
                history_parts.append(f"Assistant: {turn['response']}")

            if history_parts:
                history_section = "CONVERSATION HISTORY:\n" + "\n".join(history_parts) + "\n\n---\n\n"

        # Build query section - NO CONTEXT CONSTRAINT (based on Exp2 success)
        # Exp3 showed that forcing "ONLY use context" degrades performance by 35%
        query_section = f"""CURRENT QUESTION:
{query}

Please provide a helpful, accurate response based on WHO guidelines and family planning best practices."""

        # Combine sections (no context section - model uses its training data)
        full_prompt = f"""{history_section}{query_section}"""

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

        # Generate response with language-specific model
        try:
            response = self.llm_client.generate(
                prompt=prompt,
                system_prompt=system_prompt,
                temperature=temperature,
                max_tokens=max_tokens,
                language=language  # Route to appropriate model
            )

            # Check if response is empty or too short (potential failure)
            if not response or len(response.strip()) < 20:
                logger.warning("LLM returned empty/short response, using fallback")
                response = self._get_safety_fallback(language)

            # SAFETY VALIDATION: Post-generation checks (Trust but Verify)
            safety_validation = None
            if self.enable_safety_validation and self.safety_validator:
                safety_validation = self.safety_validator.validate(
                    response=response,
                    query=query,
                    language=language
                )

                # Log safety issues (but don't block response)
                if not safety_validation['is_safe']:
                    logger.warning(f"Safety validation flagged issues (severity={safety_validation['severity']}): {safety_validation['issues']}")

                    # High severity: Log for immediate review
                    if safety_validation['severity'] == 'high':
                        logger.error(f"HIGH SEVERITY safety issue detected. Query: {query[:100]}... Response: {response[:200]}...")

                # Add disclaimer if medical question missing provider mention
                if safety_validation.get('requires_disclaimer'):
                    logger.info("Adding safety disclaimer to medical question response")
                    response = self.safety_validator.add_disclaimer(response, language)

            # Extract citations (sources mentioned in response)
            citations = self._extract_citations(response)

            result = {
                'response': response.strip(),
                'citations': citations,
                'language': language,
                'metadata': {
                    'temperature': temperature,
                    'max_tokens': max_tokens,
                    'context_length': len(context),
                    'safety_validation': safety_validation
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


class MultilingualGenerator(ComplianceGenerator):
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
    generator = ComplianceGenerator(llm_config=config['models']['llm'])

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
