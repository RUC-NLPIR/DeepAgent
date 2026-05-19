"""
Content Bias Detector

Detects and mitigates bias in AI-generated content:
- Demographic bias (gender, race, age, etc.)
- Cultural insensitivity
- Stereotyping
- Inclusive language checking
- Representation analysis
- Harmful content detection
"""

import re
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime
from collections import defaultdict

from ..utils.logger import get_logger


class ContentBiasDetector:
    """
    Agent for detecting bias in AI-generated content

    Addresses fairness and inclusion challenges:
    - Stereotypical associations
    - Exclusionary language
    - Demographic imbalances
    - Cultural assumptions
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize Content Bias Detector"""
        self.config = config or {}
        self.logger = get_logger("system")

        # Gender bias indicators
        self.gendered_terms = {
            "male": ["he", "him", "his", "himself", "man", "men", "male", "guy", "father", "son", "brother", "husband"],
            "female": ["she", "her", "hers", "herself", "woman", "women", "female", "gal", "mother", "daughter", "sister", "wife"]
        }

        # Problematic terms
        self.non_inclusive_terms = {
            "blacklist": "blocklist",
            "whitelist": "allowlist",
            "master/slave": "primary/replica",
            "guys": "everyone/folks/team",
            "mankind": "humankind/humanity",
            "manpower": "workforce/personnel",
            "chairman": "chairperson/chair",
            "policeman": "police officer",
            "fireman": "firefighter",
            "crazy": "unexpected/surprising",
            "insane": "incredible/amazing",
            "blind spot": "gap/oversight",
            "tone-deaf": "insensitive/unaware"
        }

        # Stereotypical associations to flag
        self.stereotype_patterns = {
            "gender_profession": [
                (r'\b(nurse|secretary|teacher)\b.*\bshe\b', "female"),
                (r'\b(engineer|doctor|CEO|programmer)\b.*\bhe\b', "male"),
            ],
            "age": [
                (r'\b(young|millennial)\b.*\b(tech-savvy|innovative)\b', "age_assumption"),
                (r'\b(old|elderly|senior)\b.*\b(confused|slow)\b', "age_stereotype"),
            ],
            "cultural": [
                (r'\b(exotic|oriental)\b', "cultural_othering"),
            ]
        }

        # Inclusive language alternatives
        self.inclusive_alternatives = {
            "disabled people": "people with disabilities",
            "handicapped": "person with a disability",
            "suffers from": "has/lives with",
            "confined to a wheelchair": "uses a wheelchair",
            "normal": "typical/common",
            "abnormal": "atypical/uncommon"
        }

        self.logger.info("ContentBiasDetector initialized")

    async def detect_bias(
        self,
        content: str,
        check_types: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Comprehensive bias detection across multiple dimensions

        Args:
            content: Content to analyze
            check_types: Specific checks to run (default: all)

        Returns:
            Bias analysis with detected issues and recommendations
        """
        if check_types is None:
            check_types = [
                "gender",
                "inclusive_language",
                "stereotypes",
                "representation",
                "cultural"
            ]

        issues = []
        warnings = []
        recommendations = []

        # Run requested checks
        if "gender" in check_types:
            gender_issues = await self.check_gender_bias(content)
            if gender_issues["bias_detected"]:
                issues.extend(gender_issues["issues"])
                recommendations.extend(gender_issues["recommendations"])

        if "inclusive_language" in check_types:
            inclusive_issues = await self.check_inclusive_language(content)
            if inclusive_issues["issues_found"]:
                issues.extend(inclusive_issues["issues"])
                recommendations.extend(inclusive_issues["recommendations"])

        if "stereotypes" in check_types:
            stereotype_issues = await self.check_stereotypes(content)
            if stereotype_issues["stereotypes_detected"]:
                issues.extend(stereotype_issues["issues"])
                recommendations.extend(stereotype_issues["recommendations"])

        if "representation" in check_types:
            rep_analysis = await self.analyze_representation(content)
            if rep_analysis["imbalance_detected"]:
                warnings.extend(rep_analysis["warnings"])
                recommendations.extend(rep_analysis["recommendations"])

        if "cultural" in check_types:
            cultural_issues = await self.check_cultural_sensitivity(content)
            if cultural_issues["issues_detected"]:
                issues.extend(cultural_issues["issues"])
                recommendations.extend(cultural_issues["recommendations"])

        # Calculate overall bias score
        total_checks = len(check_types)
        issues_found = len(issues)
        bias_score = min(issues_found / max(total_checks, 1), 1.0)

        fairness_score = 1.0 - bias_score

        return {
            "bias_score": round(bias_score, 2),
            "fairness_score": round(fairness_score, 2),
            "total_issues": len(issues),
            "total_warnings": len(warnings),
            "issues": issues,
            "warnings": warnings,
            "recommendations": recommendations,
            "content_length": len(content),
            "checks_performed": check_types,
            "timestamp": datetime.now().isoformat()
        }

    async def check_gender_bias(self, content: str) -> Dict[str, Any]:
        """
        Check for gender bias in content

        Returns:
            Gender bias analysis
        """
        content_lower = content.lower()

        # Count gendered terms
        male_count = sum(content_lower.count(term) for term in self.gendered_terms["male"])
        female_count = sum(content_lower.count(term) for term in self.gendered_terms["female"])

        total_gendered = male_count + female_count

        issues = []
        recommendations = []

        # Check for significant imbalance
        if total_gendered > 5:
            ratio = male_count / total_gendered if total_gendered > 0 else 0

            if ratio > 0.8:
                issues.append({
                    "type": "gender_imbalance",
                    "severity": "high",
                    "description": f"Heavily male-skewed language ({male_count} male vs {female_count} female terms)",
                    "location": "overall"
                })
                recommendations.append("Balance gender representation or use gender-neutral language")

            elif ratio < 0.2:
                issues.append({
                    "type": "gender_imbalance",
                    "severity": "high",
                    "description": f"Heavily female-skewed language ({female_count} female vs {male_count} male terms)",
                    "location": "overall"
                })
                recommendations.append("Balance gender representation or use gender-neutral language")

        # Check for gendered job titles
        gendered_titles = [
            "businessman", "businesswoman", "congressman", "congresswoman",
            "policeman", "policewoman", "fireman", "firewoman"
        ]

        for title in gendered_titles:
            if title in content_lower:
                neutral = title.replace("man", "person").replace("woman", "person")
                issues.append({
                    "type": "gendered_title",
                    "severity": "medium",
                    "description": f"Gendered job title: '{title}'",
                    "location": content_lower.find(title)
                })
                recommendations.append(f"Use gender-neutral alternative: '{neutral}'")

        return {
            "bias_detected": len(issues) > 0,
            "male_term_count": male_count,
            "female_term_count": female_count,
            "gender_ratio": round(ratio if total_gendered > 0 else 0.5, 2),
            "issues": issues,
            "recommendations": recommendations
        }

    async def check_inclusive_language(self, content: str) -> Dict[str, Any]:
        """
        Check for non-inclusive language

        Returns:
            Inclusive language analysis
        """
        content_lower = content.lower()
        issues = []
        recommendations = []

        # Check non-inclusive terms
        for term, alternative in self.non_inclusive_terms.items():
            if term.lower() in content_lower:
                issues.append({
                    "type": "non_inclusive_term",
                    "severity": "medium",
                    "description": f"Non-inclusive term: '{term}'",
                    "location": content_lower.find(term.lower())
                })
                recommendations.append(f"Replace '{term}' with '{alternative}'")

        # Check disability language
        for term, alternative in self.inclusive_alternatives.items():
            if term.lower() in content_lower:
                issues.append({
                    "type": "disability_language",
                    "severity": "high",
                    "description": f"Non-person-first language: '{term}'",
                    "location": content_lower.find(term.lower())
                })
                recommendations.append(f"Use person-first language: '{alternative}'")

        return {
            "issues_found": len(issues) > 0,
            "total_issues": len(issues),
            "issues": issues,
            "recommendations": recommendations
        }

    async def check_stereotypes(self, content: str) -> Dict[str, Any]:
        """
        Check for stereotypical associations

        Returns:
            Stereotype detection analysis
        """
        content_lower = content.lower()
        issues = []
        recommendations = []

        # Check stereotype patterns
        for category, patterns in self.stereotype_patterns.items():
            for pattern, stereotype_type in patterns:
                matches = re.finditer(pattern, content_lower, re.IGNORECASE)
                for match in matches:
                    issues.append({
                        "type": f"stereotype_{category}",
                        "severity": "high",
                        "description": f"Potential stereotype: {stereotype_type}",
                        "location": match.start(),
                        "matched_text": match.group(0)
                    })

                    if category == "gender_profession":
                        recommendations.append("Avoid gendered assumptions about professions")
                    elif category == "age":
                        recommendations.append("Avoid age-based stereotypes")
                    elif category == "cultural":
                        recommendations.append("Use culturally respectful language")

        return {
            "stereotypes_detected": len(issues) > 0,
            "total_stereotypes": len(issues),
            "issues": issues,
            "recommendations": recommendations
        }

    async def analyze_representation(self, content: str) -> Dict[str, Any]:
        """
        Analyze demographic representation in content

        Returns:
            Representation analysis
        """
        # This is a simplified version - production would use NER and more sophisticated analysis
        content_lower = content.lower()
        warnings = []
        recommendations = []

        # Check for people mentions
        people_indicators = ["person", "people", "individual", "user", "customer", "employee"]
        has_people = any(indicator in content_lower for indicator in people_indicators)

        if has_people:
            # Check for diversity mentions
            diversity_terms = [
                "diverse", "diversity", "inclusion", "inclusive",
                "accessibility", "accessible", "equity", "equitable"
            ]
            has_diversity = any(term in content_lower for term in diversity_terms)

            if not has_diversity and len(content.split()) > 100:
                warnings.append({
                    "type": "representation",
                    "description": "Content discusses people but doesn't mention diversity/inclusion",
                    "severity": "low"
                })
                recommendations.append("Consider explicitly addressing diversity and inclusion")

        return {
            "imbalance_detected": len(warnings) > 0,
            "warnings": warnings,
            "recommendations": recommendations
        }

    async def check_cultural_sensitivity(self, content: str) -> Dict[str, Any]:
        """
        Check for cultural sensitivity issues

        Returns:
            Cultural sensitivity analysis
        """
        content_lower = content.lower()
        issues = []
        recommendations = []

        # Cultural appropriation terms
        appropriation_terms = ["spirit animal", "tribe", "pow wow", "guru"]

        for term in appropriation_terms:
            if term in content_lower:
                issues.append({
                    "type": "cultural_appropriation",
                    "severity": "high",
                    "description": f"Potentially appropriative term: '{term}'",
                    "location": content_lower.find(term)
                })
                recommendations.append(f"Avoid cultural appropriation - reconsider use of '{term}'")

        # Western-centric assumptions
        western_assumptions = [
            (r'\b(christmas|easter)\b', "religious_assumption"),
            (r'\b(obviously|clearly|everyone knows)\b', "assumed_knowledge"),
        ]

        for pattern, assumption_type in western_assumptions:
            matches = re.finditer(pattern, content_lower, re.IGNORECASE)
            for match in matches:
                issues.append({
                    "type": assumption_type,
                    "severity": "medium",
                    "description": f"Potential cultural assumption: {match.group(0)}",
                    "location": match.start()
                })
                recommendations.append("Avoid assuming shared cultural context")

        return {
            "issues_detected": len(issues) > 0,
            "total_issues": len(issues),
            "issues": issues,
            "recommendations": recommendations
        }

    async def suggest_inclusive_alternatives(
        self,
        content: str
    ) -> Dict[str, Any]:
        """
        Suggest inclusive alternatives for biased content

        Returns:
            Content with suggested replacements
        """
        suggestions = []
        modified_content = content

        # Apply all known replacements
        replacements_made = []

        for term, alternative in self.non_inclusive_terms.items():
            if term.lower() in modified_content.lower():
                # Case-preserving replacement
                pattern = re.compile(re.escape(term), re.IGNORECASE)
                modified_content = pattern.sub(alternative, modified_content)
                replacements_made.append({
                    "original": term,
                    "replacement": alternative,
                    "reason": "inclusive_language"
                })

        for term, alternative in self.inclusive_alternatives.items():
            if term.lower() in modified_content.lower():
                pattern = re.compile(re.escape(term), re.IGNORECASE)
                modified_content = pattern.sub(alternative, modified_content)
                replacements_made.append({
                    "original": term,
                    "replacement": alternative,
                    "reason": "person_first_language"
                })

        return {
            "original_content": content,
            "inclusive_content": modified_content,
            "replacements_made": replacements_made,
            "improvement_count": len(replacements_made),
            "timestamp": datetime.now().isoformat()
        }

    async def generate_bias_report(
        self,
        content: str
    ) -> Dict[str, Any]:
        """
        Generate comprehensive bias report

        Returns:
            Full bias analysis report
        """
        # Run all checks
        bias_analysis = await self.detect_bias(content)

        # Generate summary
        if bias_analysis["total_issues"] == 0:
            summary = "✅ No significant bias detected. Content appears fair and inclusive."
            grade = "A"
        elif bias_analysis["total_issues"] <= 2:
            summary = "⚠️  Minor bias issues detected. Review and address flagged concerns."
            grade = "B"
        elif bias_analysis["total_issues"] <= 5:
            summary = "⚠️  Moderate bias detected. Significant revisions recommended."
            grade = "C"
        else:
            summary = "❌ High bias detected. Major revisions required for fairness."
            grade = "F"

        # Get inclusive alternatives
        alternatives = await self.suggest_inclusive_alternatives(content)

        return {
            "summary": summary,
            "grade": grade,
            "bias_analysis": bias_analysis,
            "inclusive_alternatives": alternatives,
            "actionable_steps": self._get_actionable_steps(bias_analysis),
            "timestamp": datetime.now().isoformat()
        }

    def _get_actionable_steps(self, bias_analysis: Dict[str, Any]) -> List[str]:
        """Generate actionable steps from bias analysis"""
        steps = []

        if bias_analysis["total_issues"] > 0:
            steps.append(f"Review and address {bias_analysis['total_issues']} flagged issues")

        # Get unique recommendation types
        unique_recommendations = list(set(bias_analysis["recommendations"]))
        steps.extend(unique_recommendations[:5])  # Top 5 recommendations

        if not steps:
            steps.append("Continue following inclusive language best practices")

        return steps
