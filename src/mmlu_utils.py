import os
import torch
import numpy as np
from datasets import load_dataset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
MMLU_DATASETS_DIR = os.path.join(PROJECT_ROOT, "datasets", "mmlu")
os.makedirs(MMLU_DATASETS_DIR, exist_ok=True)

SUBJECTS = [
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

def get_mmlu_prompt(subject):
    try:
        
        # Load the dataset
        dataset = load_dataset("cais/mmlu", subject, split="test")
        
        for i in range(len(dataset)):
            item = dataset[i]
            question = item["question"]
            choices = item["choices"]
            answer_idx = item["answer"]
            
            prompt += f"Question {i+1}: {question}\n"
            for j, choice in enumerate(choices):
                prompt += f"{'ABCD'[j]}. {choice}\n"
            
            answer_letter = 'ABCD'[answer_idx]
            answer_text = choices[answer_idx]
            prompt += f"Answer: {answer_letter}. {answer_text}\n"
            
            prompt += "\n"
        
        return prompt.strip()
    
    except Exception as e:
        print(f"Warning: Could not load MMLU dataset for subject '{subject}': {e}")
        return f"MMLU prompt for subject: {subject}"

def format_example(question, choices, answer=None):
    """Format a single example (with or without answer)."""
    prompt = f"Question: {question}\n"
    for i, choice in enumerate(choices):
        prompt += f"{CHOICES[i]}. {choice}\n"
    prompt += "Answer:"
    if answer is not None:
        prompt += f" {CHOICES[answer]}\n\n"
    return prompt


def format_prompt(question, choices, few_shot_examples=None):
    """Format prompt with optional few-shot examples."""
    prompt = ""
    
    # Add few-shot examples if provided
    if few_shot_examples:
        for ex in few_shot_examples:
            prompt += format_example(ex["question"], ex["choices"], ex["answer"])
    
    # Add the actual question (without answer)
    prompt += format_example(question, choices, answer=None)
    return prompt


def get_few_shot_examples(subject, shots):
    """Get few-shot examples from the dev split."""
    if shots == 0:
        return None
    
    dev_dataset = load_dataset("cais/mmlu", subject, split="dev", cache_dir=MMLU_DATASETS_DIR)
    
    # Take up to 'shots' examples from dev set
    num_examples = min(shots, len(dev_dataset))
    examples = []
    for i in range(num_examples):
        examples.append({
            "question": dev_dataset[i]["question"],
            "choices": dev_dataset[i]["choices"],
            "answer": dev_dataset[i]["answer"]
        })
    return examples

def get_answer_logprobs(model, tokenizer, prompt, device):
    """Get log probabilities for each answer choice (A, B, C, D)."""
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[:, -1, :]
        probs = torch.softmax(logits, dim=-1)
    
    choice_probs = []
    for choice in CHOICES:
        token_id = tokenizer.encode(choice, add_special_tokens=False)[0]
        print(f"Token id for choice '{choice}': {token_id}")
        choice_probs.append(probs[0, token_id].item())

    max_prob = np.argmax(probs[0].cpu().numpy())
    print(f"========models max prob token id: {max_prob}")
    return choice_probs

def evaluate_subject(model, tokenizer, subject, device, max_samples=None, shots=0):
    """Evaluate model on a single MMLU subject."""
    dataset = load_dataset("cais/mmlu", subject, split="test", cache_dir=MMLU_DATASETS_DIR)
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    # Get few-shot examples from dev set
    few_shot_examples = get_few_shot_examples(subject, shots)
    
    correct = 0
    total = 0
    
    for item in dataset:
        question = item["question"]
        choices = item["choices"]
        label = item["answer"]
        
        prompt = format_prompt(question, choices, few_shot_examples)
        print(prompt)
        choice_probs = get_answer_logprobs(model, tokenizer, prompt, device)
        
        predicted = np.argmax(choice_probs)
        print(predicted, label)
        if predicted == label:
            correct += 1
        total += 1
    
    accuracy = correct / total if total > 0 else 0
    return accuracy, correct, total


def evaluate_mmlu(model, tokenizer, device, subjects=None, max_samples=None, shots=0):
    """Evaluate model on MMLU benchmark."""
    if subjects is None:
        subjects = SUBJECTS
    
    results = {}
    total_correct = 0
    total_questions = 0
    
    for subject in subjects:
        accuracy, correct, total = evaluate_subject(model, tokenizer, subject, device, max_samples, shots)
        results[subject] = {"accuracy": accuracy, "correct": correct, "total": total}
        total_correct += correct
        total_questions += total
        
    overall_accuracy = total_correct / total_questions if total_questions > 0 else 0
    results["overall"] = {
        "accuracy": overall_accuracy,
        "correct": total_correct,
        "total": total_questions
    }
    results["shots"] = shots
    
    return results