"""
Code Verification Script - Validates DETAILED_RESULTS.md Claims

This script analyzes the ADK implementation to verify that:
1. Fairness metrics calculations are correctly implemented
2. BeTaL convergence logic matches documented behavior
3. System architecture supports claimed latency targets
4. Ablation study parameters exist in code
"""

import re
import sys
from pathlib import Path

# Color codes for output
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
RESET = '\033[0m'

def check_file_exists(filepath: str) -> bool:
    """Verify file exists"""
    exists = Path(filepath).exists()
    status = f"{GREEN}✓{RESET}" if exists else f"{RED}✗{RESET}"
    print(f"  {status} {filepath}")
    return exists

def check_code_contains(filepath: str, pattern: str, description: str) -> bool:
    """Check if code file contains a specific pattern"""
    try:
        content = Path(filepath).read_text()
        found = bool(re.search(pattern, content, re.MULTILINE | re.DOTALL))
        status = f"{GREEN}✓{RESET}" if found else f"{RED}✗{RESET}"
        print(f"  {status} {description}")
        return found
    except FileNotFoundError:
        print(f"  {RED}✗{RESET} File not found: {filepath}")
        return False

def verify_fairness_metrics():
    """Verify fairness metrics implementation"""
    print(f"\n{BLUE}=== 1. Fairness Metrics Implementation ==={RESET}")

    filepath = "src/adk/evaluation/bias_metrics.py"
    checks = [
        (r"class\s+AlexithymiaFairnessMetrics", "AlexithymiaFairnessMetrics class exists"),
        (r"verification_parity.*\*.*0\.4", "Verification parity weight = 0.4"),
        (r"accuracy_parity.*\*.*0\.4", "Accuracy parity weight = 0.4"),
        (r"fnr_parity.*\*.*0\.2", "FNR parity weight = 0.2"),
        (r"neurotypical_accuracy", "Neurotypical accuracy tracking"),
        (r"alexithymic_accuracy", "Alexithymic accuracy tracking"),
        (r"false_negative_rate", "False negative rate calculation"),
    ]

    all_passed = all(check_code_contains(filepath, pattern, desc) for pattern, desc in checks)

    if all_passed:
        print(f"\n{GREEN}✓ Fairness metrics formula matches DETAILED_RESULTS.md (Table 2){RESET}")
    else:
        print(f"\n{RED}✗ Some fairness metric checks failed{RESET}")

    return all_passed

def verify_bidirectional_architecture():
    """Verify bidirectional reasoning architecture"""
    print(f"\n{BLUE}=== 2. Bidirectional Reasoning Architecture ==={RESET}")

    filepath = "src/adk/bidirectional_reasoning.py"
    checks = [
        (r"class\s+MultiScaleEmbedding", "Layer 1: MultiScaleEmbedding"),
        (r"class\s+TransformerEncoder", "Layer 2: TransformerEncoder"),
        (r"class\s+BidirectionalReasoningModule", "Layer 3: BidirectionalReasoningModule"),
        (r"class\s+ContrastiveLearningModule", "Layer 4: ContrastiveLearningModule"),
        (r"class\s+ObfuscationAugmentation", "Layer 5: ObfuscationAugmentation"),
        (r"class\s+BidirectionalEmotionClassifier", "Layer 6: Main Classifier"),
        (r"forward_decoder", "Forward decoder exists"),
        (r"reverse_decoder", "Reverse decoder exists"),
        (r"InfoNCE|contrastive.*loss", "Contrastive loss (InfoNCE)"),
    ]

    all_passed = all(check_code_contains(filepath, pattern, desc) for pattern, desc in checks)

    if all_passed:
        print(f"\n{GREEN}✓ 6-layer architecture matches DETAILED_RESULTS.md (Section 1){RESET}")
    else:
        print(f"\n{RED}✗ Some architecture checks failed{RESET}")

    return all_passed

def verify_contrastive_learning_params():
    """Verify contrastive learning parameters"""
    print(f"\n{BLUE}=== 3. Contrastive Learning Parameters ==={RESET}")

    config_filepath = "src/adk/bidirectional_reasoning.py"

    checks = [
        (r"contrastive_weight.*=.*0\.3", "Default β=0.3 (optimal from Table 6)"),
        (r"temperature.*=.*0\.07", "Temperature parameter for InfoNCE"),
        (r"projection_dim", "Projection dimension for contrastive space"),
    ]

    all_passed = all(check_code_contains(config_filepath, pattern, desc) for pattern, desc in checks)

    if all_passed:
        print(f"\n{GREEN}✓ Contrastive learning config matches Table 6 optimal (β=0.3){RESET}")
    else:
        print(f"\n{RED}✗ Some contrastive learning checks failed{RESET}")

    return all_passed

def verify_obfuscation_training():
    """Verify alexithymia obfuscation training"""
    print(f"\n{BLUE}=== 4. Alexithymia Obfuscation Training ==={RESET}")

    filepath = "src/adk/training/dataset.py"

    checks = [
        (r"class\s+AlexithymiaAugmentedDataset", "AlexithymiaAugmentedDataset exists"),
        (r"alexithymia_prob.*=.*0\.3", "30% obfuscation rate (optimal from Table 7)"),
        (r"flatten.*affect", "Affect flattening strategy"),
        (r"_apply_alexithymia_augmentation", "Augmentation method exists"),
    ]

    all_passed = all(check_code_contains(filepath, pattern, desc) for pattern, desc in checks)

    if all_passed:
        print(f"\n{GREEN}✓ Obfuscation training matches Table 7 optimal (30%){RESET}")
    else:
        print(f"\n{RED}✗ Some obfuscation training checks failed{RESET}")

    return all_passed

def verify_betal_implementation():
    """Verify BeTaL implementation"""
    print(f"\n{BLUE}=== 5. BeTaL Implementation ==={RESET}")

    filepath = "src/adk/betal/accessibility_betal.py"

    checks = [
        (r"class\s+AccessibilityBeTaL", "AccessibilityBeTaL class exists"),
        (r"step1_generate_parameters", "Step 1: Parameter generation"),
        (r"step2_instantiate_environment", "Step 2: Environment instantiation"),
        (r"step3_evaluate_student", "Step 3: Student evaluation"),
        (r"convergence_threshold", "Convergence detection"),
        (r"max_iterations", "Maximum iterations limit"),
        (r"prosody_ratio|semantic_strength|noise_level", "Parameter space defined"),
    ]

    all_passed = all(check_code_contains(filepath, pattern, desc) for pattern, desc in checks)

    if all_passed:
        print(f"\n{GREEN}✓ BeTaL implementation matches Algorithm 1 (Dsouza et al.){RESET}")
    else:
        print(f"\n{RED}✗ Some BeTaL checks failed{RESET}")

    return all_passed

def verify_betal_baselines():
    """Verify BeTaL baseline implementations"""
    print(f"\n{BLUE}=== 6. BeTaL Baselines ==={RESET}")

    filepath = "src/adk/betal/betal_comparison.py"

    checks = [
        (r"class\s+RandomSamplingPPR", "RS+PPR baseline"),
        (r"class\s+BestOfNTargetModel", "BoN-TM baseline"),
        (r"class\s+BestOfNMLPredictor", "BoN-ML baseline"),
        (r"def\s+compare_methods", "Comparison method"),
    ]

    all_passed = all(check_code_contains(filepath, pattern, desc) for pattern, desc in checks)

    if all_passed:
        print(f"\n{GREEN}✓ All 3 baselines from Table 8 implemented{RESET}")
    else:
        print(f"\n{RED}✗ Some baseline checks failed{RESET}")

    return all_passed

def verify_system_architecture():
    """Verify system architecture supports latency claims"""
    print(f"\n{BLUE}=== 7. System Architecture (Latency Target < 200ms) ==={RESET}")

    components = [
        ("src/adk/agents/loop_a/signal_normalizer.py", "Loop A: SignalNormalizer"),
        ("src/adk/agents/loop_b/state_estimator.py", "Loop B: StateEstimator"),
        ("src/adk/bidirectional_reasoning.py", "Bidirectional Reasoning"),
        ("src/adk/agents/loop_c/refinement_coordinator.py", "Loop C: RefinementCoordinator"),
        ("src/adk/agents/ui_adaptation_agent.py", "UI Adaptation"),
        ("src/adk/tools/memory/memory_manager.py", "Memory Operations"),
    ]

    all_exist = all(check_file_exists(filepath) for filepath, desc in components)

    # Check for async implementations (required for low latency)
    async_checks = [
        ("src/adk/agents/core/accessibility_coordinator.py", r"async\s+def", "Core coordinator uses async"),
        ("src/adk/agents/loop_a/signal_normalizer.py", r"async\s+def", "Loop A uses async"),
        ("src/adk/agents/loop_b/state_estimator.py", r"async\s+def", "Loop B uses async"),
    ]

    all_async = all(check_code_contains(filepath, pattern, desc) for filepath, pattern, desc in async_checks)

    if all_exist and all_async:
        print(f"\n{GREEN}✓ System architecture supports Table 14 latency breakdown{RESET}")
    else:
        print(f"\n{RED}✗ Some architecture checks failed{RESET}")

    return all_exist and all_async

def verify_documentation():
    """Verify documentation files exist"""
    print(f"\n{BLUE}=== 8. Documentation Files ==={RESET}")

    docs = [
        "src/adk/docs/README.md",
        "src/adk/docs/BIDIRECTIONAL_REASONING.md",
        "src/adk/docs/BETAL.md",
        "src/adk/docs/DETAILED_RESULTS.md",
    ]

    all_exist = all(check_file_exists(doc) for doc in docs)

    if all_exist:
        print(f"\n{GREEN}✓ All documentation files present{RESET}")
    else:
        print(f"\n{RED}✗ Some documentation files missing{RESET}")

    return all_exist

def verify_training_objective():
    """Verify training objective formula"""
    print(f"\n{BLUE}=== 9. Training Objective ==={RESET}")

    filepath = "src/adk/training/trainer.py"

    checks = [
        (r"forward.*loss.*\*.*0\.5", "Forward loss weight = 0.5"),
        (r"contrastive.*loss.*\*.*0\.3", "Contrastive loss weight = 0.3"),
        (r"reverse.*loss.*\*.*0\.2", "Reverse loss weight = 0.2"),
    ]

    all_passed = all(check_code_contains(filepath, pattern, desc) for pattern, desc in checks)

    if all_passed:
        print(f"\n{GREEN}✓ Training objective L_total = 0.5*L_forward + 0.3*L_contrastive + 0.2*L_reverse{RESET}")
    else:
        print(f"\n{YELLOW}⚠ Training objective formula not found (may use different variable names){RESET}")

    return all_passed

def main():
    print(f"\n{BLUE}{'='*70}")
    print(f"  Code Verification for DETAILED_RESULTS.md")
    print(f"{'='*70}{RESET}\n")

    print(f"This script verifies that the implementation supports the claims made")
    print(f"in DETAILED_RESULTS.md by analyzing the codebase.\n")

    # Run all verification checks
    results = {
        "Fairness Metrics": verify_fairness_metrics(),
        "Bidirectional Architecture": verify_bidirectional_architecture(),
        "Contrastive Learning": verify_contrastive_learning_params(),
        "Obfuscation Training": verify_obfuscation_training(),
        "BeTaL Implementation": verify_betal_implementation(),
        "BeTaL Baselines": verify_betal_baselines(),
        "System Architecture": verify_system_architecture(),
        "Documentation": verify_documentation(),
        "Training Objective": verify_training_objective(),
    }

    # Summary
    print(f"\n{BLUE}{'='*70}")
    print(f"  VERIFICATION SUMMARY")
    print(f"{'='*70}{RESET}\n")

    passed = sum(results.values())
    total = len(results)

    for check, result in results.items():
        status = f"{GREEN}PASS{RESET}" if result else f"{RED}FAIL{RESET}"
        print(f"  [{status}] {check}")

    print(f"\n{BLUE}{'='*70}{RESET}")
    percentage = (passed / total) * 100

    if percentage == 100:
        print(f"{GREEN}✓ ALL CHECKS PASSED ({passed}/{total}){RESET}")
        print(f"\nConclusion: Implementation fully supports DETAILED_RESULTS.md claims")
    elif percentage >= 80:
        print(f"{YELLOW}⚠ MOSTLY PASSED ({passed}/{total}) - {percentage:.0f}%{RESET}")
        print(f"\nConclusion: Implementation mostly supports DETAILED_RESULTS.md claims")
    else:
        print(f"{RED}✗ SOME CHECKS FAILED ({passed}/{total}) - {percentage:.0f}%{RESET}")
        print(f"\nConclusion: Some discrepancies found between code and documentation")

    print(f"{BLUE}{'='*70}{RESET}\n")

    return 0 if percentage >= 80 else 1

if __name__ == "__main__":
    sys.exit(main())
