"""
Generate Multi-Language Q&A Pairs

Creates French and Kinyarwanda questions with ground truth answers
for contraception counseling evaluation.
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any

# Set seed for reproducibility
random.seed(42)


class MultiLanguageQAGenerator:
    """Generate multi-language Q&A pairs for evaluation."""

    def __init__(self):
        self.french_questions = self._create_french_questions()
        self.kinyarwanda_questions = self._create_kinyarwanda_questions()

    def _create_french_questions(self) -> List[Dict[str, Any]]:
        """Create French questions based on WHO materials."""

        questions = [
            # Effectiveness questions
            {
                "question": "Quelle est l'efficacité de l'implant contraceptif?",
                "ground_truth": "L'implant contraceptif est plus de 99% efficace. C'est l'une des méthodes les plus efficaces disponibles.",
                "category": "effectiveness",
                "difficulty": "easy",
                "language": "french"
            },
            {
                "question": "Le DIU au cuivre peut-il être utilisé comme contraception d'urgence?",
                "ground_truth": "Oui, le DIU au cuivre peut être inséré jusqu'à 5 jours après un rapport non protégé et est plus de 99% efficace comme contraception d'urgence.",
                "category": "emergency_contraception",
                "difficulty": "medium",
                "language": "french"
            },
            {
                "question": "Quelle est la différence entre l'efficacité théorique et l'efficacité pratique?",
                "ground_truth": "L'efficacité théorique est l'efficacité en utilisation parfaite, tandis que l'efficacité pratique tient compte de l'utilisation typique avec des erreurs possibles.",
                "category": "effectiveness",
                "difficulty": "medium",
                "language": "french"
            },

            # Side effects questions
            {
                "question": "Quels sont les effets secondaires courants de la pilule contraceptive?",
                "ground_truth": "Les effets secondaires courants incluent les nausées, les maux de tête, les changements d'humeur, et parfois des saignements irréguliers, surtout pendant les premiers mois.",
                "category": "side_effects",
                "difficulty": "easy",
                "language": "french"
            },
            {
                "question": "L'injection DMPA provoque-t-elle des changements de poids?",
                "ground_truth": "Certaines femmes peuvent prendre du poids avec l'injection DMPA, généralement entre 1-2 kg par an. Cependant, cela varie selon les individus.",
                "category": "side_effects",
                "difficulty": "medium",
                "language": "french"
            },
            {
                "question": "Les saignements irréguliers sont-ils dangereux avec le DIU hormonal?",
                "ground_truth": "Les saignements irréguliers sont courants avec le DIU hormonal, surtout dans les premiers mois. Ils ne sont généralement pas dangereux et diminuent avec le temps.",
                "category": "side_effects",
                "difficulty": "medium",
                "language": "french"
            },

            # Fertility return questions
            {
                "question": "Combien de temps faut-il pour que la fertilité revienne après l'arrêt de l'implant?",
                "ground_truth": "La fertilité revient généralement rapidement après le retrait de l'implant, habituellement dans un délai de 1 à 3 mois.",
                "category": "fertility_return",
                "difficulty": "easy",
                "language": "french"
            },
            {
                "question": "La pilule contraceptive affecte-t-elle la fertilité à long terme?",
                "ground_truth": "Non, la pilule n'affecte pas la fertilité à long terme. La fertilité revient généralement rapidement après l'arrêt de la pilule.",
                "category": "fertility_return",
                "difficulty": "easy",
                "language": "french"
            },
            {
                "question": "Après l'injection DMPA, combien de temps faut-il pour retrouver la fertilité?",
                "ground_truth": "Après l'arrêt de l'injection DMPA, il peut falloir entre 3 et 12 mois pour que la fertilité revienne complètement, parfois plus longtemps.",
                "category": "fertility_return",
                "difficulty": "medium",
                "language": "french"
            },

            # Usage questions
            {
                "question": "Que faire si j'oublie une pilule contraceptive?",
                "ground_truth": "Si vous oubliez une pilule, prenez-la dès que vous vous en souvenez. Si plus de 24 heures se sont écoulées, consultez les instructions spécifiques selon le type de pilule.",
                "category": "usage",
                "difficulty": "medium",
                "language": "french"
            },
            {
                "question": "À quelle fréquence dois-je recevoir l'injection DMPA?",
                "ground_truth": "L'injection DMPA doit être administrée tous les 3 mois (12-13 semaines) pour maintenir son efficacité contraceptive.",
                "category": "usage",
                "difficulty": "easy",
                "language": "french"
            },
            {
                "question": "Combien de temps puis-je garder un DIU?",
                "ground_truth": "Cela dépend du type: le DIU au cuivre peut rester en place 10-12 ans, le DIU hormonal 3-7 ans selon le modèle.",
                "category": "usage",
                "difficulty": "medium",
                "language": "french"
            },

            # Safety questions
            {
                "question": "Qui ne devrait pas utiliser la pilule contraceptive combinée?",
                "ground_truth": "Les femmes qui fument et ont plus de 35 ans, celles ayant des antécédents de caillots sanguins, d'hypertension non contrôlée, ou certaines conditions médicales ne devraient pas utiliser la pilule combinée.",
                "category": "contraindications",
                "difficulty": "hard",
                "language": "french"
            },
            {
                "question": "Le DIU peut-il causer une infection?",
                "ground_truth": "Le risque d'infection est très faible et se limite généralement aux premières semaines après l'insertion. Avec une insertion appropriée, le DIU est très sûr.",
                "category": "safety",
                "difficulty": "medium",
                "language": "french"
            },
            {
                "question": "Les préservatifs protègent-ils contre les IST?",
                "ground_truth": "Oui, les préservatifs (masculins et féminins) sont la seule méthode contraceptive qui protège également contre les infections sexuellement transmissibles (IST).",
                "category": "protection",
                "difficulty": "easy",
                "language": "french"
            },

            # Method comparison
            {
                "question": "Quelle est la différence entre le DIU hormonal et le DIU au cuivre?",
                "ground_truth": "Le DIU hormonal libère des hormones (progestatif) et peut réduire les règles, tandis que le DIU au cuivre ne contient pas d'hormones et peut rendre les règles plus abondantes.",
                "category": "method_comparison",
                "difficulty": "medium",
                "language": "french"
            },
            {
                "question": "Quelle méthode contraceptive est la plus efficace?",
                "ground_truth": "Les méthodes les plus efficaces sont l'implant, le DIU (hormonal et cuivre), et la stérilisation, tous avec une efficacité supérieure à 99%.",
                "category": "effectiveness",
                "difficulty": "easy",
                "language": "french"
            },

            # Breastfeeding compatibility
            {
                "question": "Puis-je utiliser la pilule contraceptive pendant l'allaitement?",
                "ground_truth": "La pilule progestative seule est compatible avec l'allaitement. La pilule combinée n'est généralement pas recommandée avant 6 mois après l'accouchement si vous allaitez.",
                "category": "breastfeeding",
                "difficulty": "medium",
                "language": "french"
            },
            {
                "question": "L'injection DMPA est-elle sûre pendant l'allaitement?",
                "ground_truth": "Oui, l'injection DMPA est considérée comme sûre pendant l'allaitement et n'affecte pas la qualité ou la quantité du lait maternel.",
                "category": "breastfeeding",
                "difficulty": "easy",
                "language": "french"
            },

            # Age-related questions
            {
                "question": "À quel âge peut-on commencer à utiliser la contraception?",
                "ground_truth": "La contraception peut être utilisée dès le début de l'activité sexuelle, quel que soit l'âge. Toutes les méthodes peuvent être utilisées par les adolescentes, selon leurs besoins.",
                "category": "age_appropriateness",
                "difficulty": "easy",
                "language": "french"
            },
            {
                "question": "Peut-on utiliser un DIU si on n'a jamais eu d'enfants?",
                "ground_truth": "Oui, le DIU peut être utilisé par les femmes qui n'ont jamais eu d'enfants. C'est sûr et efficace pour toutes les femmes en âge de procréer.",
                "category": "eligibility",
                "difficulty": "medium",
                "language": "french"
            },

            # Access and cost
            {
                "question": "Où puis-je obtenir gratuitement une contraception?",
                "ground_truth": "Les centres de santé publics, les cliniques de planification familiale, et certaines ONG offrent souvent une contraception gratuite ou à faible coût. Renseignez-vous auprès des services de santé locaux.",
                "category": "access",
                "difficulty": "easy",
                "language": "french"
            },
            {
                "question": "Les méthodes contraceptives à longue durée d'action sont-elles plus économiques?",
                "ground_truth": "Oui, bien que le coût initial puisse être plus élevé, les méthodes comme le DIU et l'implant sont plus économiques à long terme car elles durent plusieurs années.",
                "category": "cost_effectiveness",
                "difficulty": "medium",
                "language": "french"
            },

            # Myths and misconceptions
            {
                "question": "La contraception cause-t-elle l'infertilité?",
                "ground_truth": "Non, c'est un mythe. La contraception moderne ne cause pas d'infertilité. La fertilité revient après l'arrêt de la méthode contraceptive.",
                "category": "myths",
                "difficulty": "easy",
                "language": "french"
            },
            {
                "question": "Les hormones contraceptives sont-elles dangereuses pour la santé?",
                "ground_truth": "Pour la plupart des femmes, les hormones contraceptives sont sûres. Les avantages dépassent largement les risques. Cependant, certaines conditions médicales peuvent nécessiter des méthodes non hormonales.",
                "category": "safety",
                "difficulty": "medium",
                "language": "french"
            },

            # Emergency contraception
            {
                "question": "Dans quel délai dois-je prendre la contraception d'urgence?",
                "ground_truth": "La contraception d'urgence doit être prise le plus tôt possible après un rapport non protégé, idéalement dans les 72-120 heures selon le type, mais elle est plus efficace dans les premières 24 heures.",
                "category": "emergency_contraception",
                "difficulty": "medium",
                "language": "french"
            },
            {
                "question": "La contraception d'urgence provoque-t-elle un avortement?",
                "ground_truth": "Non, la contraception d'urgence empêche la grossesse en retardant l'ovulation. Elle ne provoque pas d'avortement et n'affecte pas une grossesse existante.",
                "category": "emergency_contraception",
                "difficulty": "medium",
                "language": "french"
            },

            # Partner involvement
            {
                "question": "Ai-je besoin du consentement de mon partenaire pour utiliser la contraception?",
                "ground_truth": "Non, vous avez le droit de choisir votre méthode contraceptive sans le consentement de votre partenaire. La décision vous appartient.",
                "category": "rights",
                "difficulty": "easy",
                "language": "french"
            },
            {
                "question": "Comment puis-je parler de contraception avec mon partenaire?",
                "ground_truth": "Choisissez un moment calme pour discuter, exprimez vos préoccupations et besoins, écoutez votre partenaire, et prenez une décision ensemble qui respecte vos choix.",
                "category": "communication",
                "difficulty": "medium",
                "language": "french"
            },

            # Switching methods
            {
                "question": "Puis-je changer de méthode contraceptive?",
                "ground_truth": "Oui, vous pouvez changer de méthode à tout moment. Consultez un professionnel de santé pour assurer une transition en douceur et maintenir une protection contraceptive continue.",
                "category": "switching_methods",
                "difficulty": "easy",
                "language": "french"
            }
        ]

        # Add question IDs
        for i, q in enumerate(questions):
            q["question_id"] = f"fr_{i:04d}"

        return questions

    def _create_kinyarwanda_questions(self) -> List[Dict[str, Any]]:
        """Create Kinyarwanda questions based on WHO materials."""

        questions = [
            # Effectiveness questions
            {
                "question": "Ni gute 'implant' yo kuboneza urubyaro ikora?",
                "ground_truth": "Implant ikora neza cyane, irenga 99% mu gukumira inda. Ni kimwe mu buryo bwizewe cyane buboneka.",
                "category": "effectiveness",
                "difficulty": "easy",
                "language": "kinyarwanda"
            },
            {
                "question": "IUD yo ku munyu irashobora gukoreshwa nk'ubuzima bw'ihutirwa?",
                "ground_truth": "Yego, IUD yo ku munyu irashobora gushyirwa mu minsi 5 nyuma y'imibonano mpuzabitsina itarinze kandi ikora neza cyane nk'ubuzima bw'ihutirwa.",
                "category": "emergency_contraception",
                "difficulty": "medium",
                "language": "kinyarwanda"
            },

            # Side effects
            {
                "question": "Ni izihe ngaruka zisanzwe z'inkingo ya DMPA?",
                "ground_truth": "Ingaruka zisanzwe zirimo guhinduka kw'ibiro (kuzamuka cyangwa kugabanuka), amaraso atunguranye, no guhora wumva ubabaye. Ibi bikunze kugabanuka nyuma y'amezi menshi.",
                "category": "side_effects",
                "difficulty": "easy",
                "language": "kinyarwanda"
            },
            {
                "question": "Inkingo ya DMPA iterwa ryari?",
                "ground_truth": "Inkingo ya DMPA iterwa buri mezi atatu (ibyumweru 12-13) kugira ngo ikomeze gukora neza mu gukumira inda.",
                "category": "usage",
                "difficulty": "easy",
                "language": "kinyarwanda"
            },

            # Fertility return
            {
                "question": "Ni ryari uburumbuke bugaruka nyuma yo guhagarika implant?",
                "ground_truth": "Uburumbuke busanzwe bugaruka vuba nyuma yo gukuramo implant, mubisanzwe mu mezi 1 kugeza 3.",
                "category": "fertility_return",
                "difficulty": "easy",
                "language": "kinyarwanda"
            },
            {
                "question": "Pillule yo kuboneza urubyaro ihora itesha agaciro uburumbuke?",
                "ground_truth": "Oya, pillule ntihora itesha agaciro uburumbuke. Uburumbuke busanzwe bugaruka vuba nyuma yo guhagarika pillule.",
                "category": "fertility_return",
                "difficulty": "easy",
                "language": "kinyarwanda"
            },

            # Method comparison
            {
                "question": "Ni ubuhe buryo bwo kuboneza urubyaro bukora neza cyane?",
                "ground_truth": "Uburyo bukora neza cyane ni implant, IUD (ya hormone no ku munyu), na sterilization, byose hejuru ya 99% mu gukora neza.",
                "category": "effectiveness",
                "difficulty": "easy",
                "language": "kinyarwanda"
            },
            {
                "question": "Ni iyihe tofali itandukanye hagati ya IUD ya hormone na IUD yo ku munyu?",
                "ground_truth": "IUD ya hormone irekura imiti ya hormone (progestatif) kandi ishobora kugabanya imihango, mugihe IUD yo ku munyu idafite hormone kandi ishobora kongera imihango.",
                "category": "method_comparison",
                "difficulty": "medium",
                "language": "kinyarwanda"
            },

            # Safety
            {
                "question": "Kapoti zikarinda indwara zandurira mu mibonano mpuzabitsina?",
                "ground_truth": "Yego, kapoti (z'abagabo n'abagore) ni yo buryo bwo kuboneza urubyaro burinda kandi indwara zandurira mu mibonano mpuzabitsina (STI).",
                "category": "protection",
                "difficulty": "easy",
                "language": "kinyarwanda"
            },
            {
                "question": "IUD irashobora gutera ubwandu?",
                "ground_truth": "Ibyago byo kwandura ni bike cyane kandi birashoboka mu byumweru bya mbere nyuma yo gushyirwa. Iyo yashyizwe neza, IUD ni umutekano cyane.",
                "category": "safety",
                "difficulty": "medium",
                "language": "kinyarwanda"
            },

            # Breastfeeding
            {
                "question": "Nshobora gukoresha pillule yo kuboneza urubyaro mugihe nkonsa?",
                "ground_truth": "Pillule ya progestatif yonyine irahuje no konsa. Pillule ifatanije ntabwo isanzwe irasabwa mbere y'amezi 6 nyuma yo kubyara niba ukonsa.",
                "category": "breastfeeding",
                "difficulty": "medium",
                "language": "kinyarwanda"
            },
            {
                "question": "Inkingo ya DMPA ni umutekano mugihe nkonsa?",
                "ground_truth": "Yego, inkingo ya DMPA ifatwa nk'umutekano mugihe nkonsa kandi ntigira ingaruka ku mwiza cyangwa ubwinshi bw'amata.",
                "category": "breastfeeding",
                "difficulty": "easy",
                "language": "kinyarwanda"
            },

            # Age and eligibility
            {
                "question": "Ni ryari ushobora gutangira gukoresha ubuzima bwo kuboneza urubyaro?",
                "ground_truth": "Ubuzima bwo kuboneza urubyaro bushobora gukoreshwa uhereye mugihe utangiye imibonano mpuzabitsina, ari uko wabaye wangahe. Uburyo bwose bushobora gukoreshwa n'ingimbi, ukurikije ibikenewe byabo.",
                "category": "age_appropriateness",
                "difficulty": "easy",
                "language": "kinyarwanda"
            },
            {
                "question": "Nshobora gukoresha IUD niba ntarigeze mbyara?",
                "ground_truth": "Yego, IUD irashobora gukoreshwa n'abagore batarihora byara. Ni umutekano kandi ikora neza ku bagore bose mu myaka yo kubyara.",
                "category": "eligibility",
                "difficulty": "medium",
                "language": "kinyarwanda"
            },

            # Access
            {
                "question": "Ni he nashobora kubona ubuzima bwo kuboneza urubyaro kubuntu?",
                "ground_truth": "Ibigo by'ubuzima bwa leta, amaklinike yo gutegura urubyaro, n'imiryango itemewe bikunze gutanga ubuzima bwo kuboneza urubyaro kubuntu cyangwa ku giciro gito. Wibaze ku masoko y'ubuzima yo mu karere kawe.",
                "category": "access",
                "difficulty": "easy",
                "language": "kinyarwanda"
            },

            # Myths
            {
                "question": "Ubuzima bwo kuboneza urubyaro butera ubumuge?",
                "ground_truth": "Oya, ibi ni ibinyoma. Ubuzima bugezweho bwo kuboneza urubyaro ntibuzana ubumuge. Uburumbuke bugaruka nyuma yo guhagarika uburyo bwo kuboneza urubyaro.",
                "category": "myths",
                "difficulty": "easy",
                "language": "kinyarwanda"
            },

            # Emergency contraception
            {
                "question": "Ni mugihe kingana iki nkwiye gufata ubuzima bw'ihutirwa?",
                "ground_truth": "Ubuzima bw'ihutirwa bugomba gufatwa ako kanya bishoboka nyuma y'imibonano mpuzabitsina itarinze, nibura mu masaha 72-120 ukurikije ubwoko, ariko bukora neza cyane mu masaha ya mbere 24.",
                "category": "emergency_contraception",
                "difficulty": "medium",
                "language": "kinyarwanda"
            },

            # Rights and autonomy
            {
                "question": "Nkeneye uruhushya rw'umukunzi wanjye kugira ngo nkoreshe ubuzima bwo kuboneza urubyaro?",
                "ground_truth": "Oya, ufite uburenganzira bwo guhitamo uburyo bwo kuboneza urubyaro udakeneye uruhushya rw'umukunzi wawe. Icyemezo kiri kuri wewe.",
                "category": "rights",
                "difficulty": "easy",
                "language": "kinyarwanda"
            },

            # Switching methods
            {
                "question": "Nshobora guhindura uburyo bwo kuboneza urubyaro?",
                "ground_truth": "Yego, urashobora guhindura uburyo igihe icyo aricyo cyose. Gusaba inama ku muganga kugirango ubashe inzibacyuho yoroshye kandi ukomeze kurindwa neza.",
                "category": "switching_methods",
                "difficulty": "easy",
                "language": "kinyarwanda"
            },

            # Side effect management
            {
                "question": "Niba nkagize ikibazo cyangwa amaraso atunguranye, niki nkwiye gukora?",
                "ground_truth": "Amaraso atunguranye ni asanzwe mugihe cya mbere cyo gukoresha uburyo bwo kuboneza urubyaro. Niba aho bigumye cyangwa ubabaye cyane, saba inama ku muganga.",
                "category": "side_effect_management",
                "difficulty": "medium",
                "language": "kinyarwanda"
            },

            # Partner communication
            {
                "question": "Ni gute nshobora kuvugana n'umukunzi wanjye ku bijyanye no kuboneza urubyaro?",
                "ground_truth": "Hitamo igihe cyo gutuza kugirango muganire, vuga impungenge zawe n'ibyo ukeneye, umve umukunzi wawe, kandi mufate icyemezo hamwe gishyira mu gaciro amahitamo yawe.",
                "category": "communication",
                "difficulty": "medium",
                "language": "kinyarwanda"
            },

            # Cost considerations
            {
                "question": "Uburyo burambye bwo kuboneza urubyaro ni buhenze?",
                "ground_truth": "Yego, nubwo ikiguzi cya mbere gishobora kuba kinini, uburyo nka IUD na implant ni buhenze mugihe kirekire kuko bumara imyaka myinshi.",
                "category": "cost_effectiveness",
                "difficulty": "medium",
                "language": "kinyarwanda"
            },

            # Medical conditions
            {
                "question": "Niba mfite indwara, ni ubuhe buryo bwo kuboneza urubyaro nshobora gukoresha?",
                "ground_truth": "Ibi biterwa n'indwara wafite. Abantu benshi bashobora gukoresha uburyo bwinshi bwo kuboneza urubyaro nta kibazo. Saba inama ku muganga kugirango ubashe uburyo bukwiriye wewe.",
                "category": "medical_conditions",
                "difficulty": "hard",
                "language": "kinyarwanda"
            },

            # Dual protection
            {
                "question": "Nshobora gukoresha kapoti hamwe n'ubundi buryo?",
                "ground_truth": "Yego! Gukoresha kapoti hamwe n'ubundi buryo (nka pillule cyangwa IUD) ni ngombwa cyane kuko bitanga kurinda kabiri: gukumira inda no kurinda indwara.",
                "category": "dual_protection",
                "difficulty": "easy",
                "language": "kinyarwanda"
            },

            # Long-acting methods
            {
                "question": "Ni izihe nyungu z'uburyo burambye nka implant na IUD?",
                "ground_truth": "Uburyo burambye butanga kurindwa igihe kirekire (imyaka 3-12), ntubakeneye kwibuka buri munsi, ni buhenze mugihe kirekire, kandi bukora neza cyane.",
                "category": "long_acting_methods",
                "difficulty": "medium",
                "language": "kinyarwanda"
            },

            # Insertion and removal
            {
                "question": "Bigabanya kugirango bashyireho IUD cyangwa implant?",
                "ground_truth": "Ugushyiraho ni uburyo bworoshye kandi butera igihe gito (iminota 5-15). Hakoreshwa ubuvuzi bwo ku rubuga kubabaza kandi abantu benshi bavuga ko babaye ubabaye buke cyangwa nta babaye.",
                "category": "procedures",
                "difficulty": "medium",
                "language": "kinyarwanda"
            },

            # Natural methods
            {
                "question": "Uburyo bwa kamere bwo kuboneza urubyaro bukora neza?",
                "ground_truth": "Uburyo bwa kamere (nko kubara iminsi) burashobora gukora ariko bukeneye kuba umwumvikana cyane no kwiyumvisha neza igihe wese. Ntabwo bukora neza nkuburyo bugezweho.",
                "category": "natural_methods",
                "difficulty": "medium",
                "language": "kinyarwanda"
            }
        ]

        # Add question IDs
        for i, q in enumerate(questions):
            q["question_id"] = f"kw_{i:04d}"

        return questions

    def generate_multilang_dataset(self, output_file: str):
        """Generate complete multi-language Q&A dataset."""

        # Load existing English questions
        english_file = Path("data/synthetic/qa_pairs.json")
        if english_file.exists():
            with open(english_file, 'r', encoding='utf-8') as f:
                english_questions = json.load(f)
                # Add language tag if not present
                for q in english_questions:
                    if 'language' not in q:
                        q['language'] = 'english'
        else:
            english_questions = []

        # Combine all questions
        all_questions = english_questions + self.french_questions + self.kinyarwanda_questions

        # Create dataset
        dataset = {
            "metadata": {
                "total_questions": len(all_questions),
                "english_questions": len(english_questions),
                "french_questions": len(self.french_questions),
                "kinyarwanda_questions": len(self.kinyarwanda_questions),
                "languages": ["english", "french", "kinyarwanda"],
                "purpose": "Multi-language contraception counseling evaluation"
            },
            "questions": all_questions
        }

        # Save dataset
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, indent=2, ensure_ascii=False)

        print(f"Multi-language dataset saved to: {output_path}")
        print(f"Total questions: {len(all_questions)}")
        print(f"  English: {len(english_questions)}")
        print(f"  French: {len(self.french_questions)}")
        print(f"  Kinyarwanda: {len(self.kinyarwanda_questions)}")

        return dataset


def main():
    """Generate multi-language Q&A dataset."""
    print("Generating Multi-Language Q&A Dataset...")
    print()

    generator = MultiLanguageQAGenerator()
    dataset = generator.generate_multilang_dataset("data/synthetic/multilang_qa_pairs.json")

    print("\nDataset generation complete!")
    print(f"French questions: {len(generator.french_questions)}")
    print(f"Kinyarwanda questions: {len(generator.kinyarwanda_questions)}")


if __name__ == "__main__":
    main()
