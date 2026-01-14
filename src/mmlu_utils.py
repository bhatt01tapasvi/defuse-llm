import os
import torch
import numpy as np
from datasets import load_dataset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MMLU_DATASETS_DIR = os.path.join(PROJECT_ROOT, "datasets", "mmlu")
os.makedirs(MMLU_DATASETS_DIR, exist_ok=True)

MMLU_SUBJECTS = [
    "abstract_algebra", "anatomy", "astronomy", "business_ethics", "clinical_knowledge",
    "college_biology", "college_chemistry", "college_computer_science", "college_mathematics",
    "college_medicine", "college_physics", "computer_security", "conceptual_physics",
    "econometrics", "electrical_engineering", "elementary_mathematics", "formal_logic",
    "global_facts", "high_school_biology", "high_school_chemistry", "high_school_computer_science",
    "high_school_european_history", "high_school_geography", "high_school_government_and_politics",
    "high_school_macroeconomics", "high_school_mathematics", "high_school_microeconomics",
    "high_school_physics", "high_school_psychology", "high_school_statistics",
    "high_school_us_history", "high_school_world_history", "human_aging", "human_sexuality",
    "international_law", "jurisprudence", "logical_fallacies", "machine_learning", "management",
    "marketing", "medical_genetics", "miscellaneous", "moral_disputes", "moral_scenarios",
    "nutrition", "philosophy", "prehistory", "professional_accounting", "professional_law",
    "professional_medicine", "professional_psychology", "public_relations", "security_studies",
    "sociology", "us_foreign_policy", "virology", "world_religions"
]

CHOICES = ["A", "B", "C", "D"]

def get_mmlu_prompt(subject, max_samples=None):
    """
    Get MMLU prompts as a LIST of individual question strings.
    Each element is one complete Q&A pair.
    
    Returns:
        list[str]: List of formatted question-answer pairs
    """
    try:
        # Load the dataset
        dataset = load_dataset("cais/mmlu", subject, split="test")
        
        if max_samples:
            dataset = dataset.select(range(min(max_samples, len(dataset))))

        prompts = []  # Return a list instead of concatenated string
        
        for i in range(len(dataset)):
            item = dataset[i]
            question = item["question"]
            choices = item["choices"]
            answer_idx = item["answer"]
            
            # Build individual prompt
            prompt = f"Question: {question}\n"
            for j, choice in enumerate(choices):
                prompt += f"{'ABCD'[j]}. {choice}\n"
            
            answer_letter = 'ABCD'[answer_idx]
            answer_text = choices[answer_idx]
            prompt += f"Answer: {answer_letter}. {answer_text}"
            
            prompts.append(prompt)
        
        return prompts
    
    except Exception as e:
        print(f"Warning: Could not load MMLU dataset for subject '{subject}': {e}")
        return []


def get_mmlu_prompt_concat(subject, max_samples=None):
    """
    Get MMLU prompts as a SINGLE concatenated string.
    Supports multiple space-separated subjects.
    
    Args:
        subject: MMLU subject name(s). Can be a single subject or space-separated subjects
                 e.g., "abstract_algebra" or "abstract_algebra anatomy astronomy"
        max_samples: Maximum number of questions to include per subject
    
    Returns:
        str: Concatenated prompts from all specified subjects
    """
    # Check if multiple subjects are provided (space-separated)
    if ' ' in subject:
        # Handle multiple subjects
        subjects = subject.split()
        all_prompts = []
        for subj in subjects:
            subj = subj.strip()
            if subj:  # Skip empty strings
                prompts = get_mmlu_prompt(subj, max_samples)
                all_prompts.extend(prompts)
        concatenated = "\n\n".join(all_prompts)
    else:
        # Handle single subject (backward compatible)
        prompts = get_mmlu_prompt(subject, max_samples)
        concatenated = "\n\n".join(prompts)
    
    return concatenated
