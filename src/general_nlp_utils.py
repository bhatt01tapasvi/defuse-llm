import os
import torch
import numpy as np
from datasets import load_dataset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(SCRIPT_DIR)
GENERAL_DATASETS_DIR = os.path.join(PROJECT_ROOT, "datasets", "general_nlp")
os.makedirs(GENERAL_DATASETS_DIR, exist_ok=True)

def evaluate_boolq(model, tokenizer, device, max_samples=None):
    """Evaluate on BoolQ (binary yes/no questions)."""
    dataset = load_dataset("boolq", split="validation", cache_dir=GENERAL_DATASETS_DIR)
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    correct = 0
    total = 0
    
    for item in dataset:
        passage = item["passage"]
        question = item["question"]
        label = item["answer"]  # True or False
        
        prompt = f"Passage: {passage}\nQuestion: {question}\nAnswer:"
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
        
        # Get probabilities for "yes" and "no"
        yes_tokens = tokenizer.encode(" yes", add_special_tokens=False)
        no_tokens = tokenizer.encode(" no", add_special_tokens=False)
        
        yes_prob = probs[0, yes_tokens[0]].item()
        no_prob = probs[0, no_tokens[0]].item()
        
        predicted = yes_prob > no_prob
        if predicted == label:
            correct += 1
        total += 1
    
    accuracy = correct / total if total > 0 else 0
    return accuracy, correct, total


def evaluate_rte(model, tokenizer, device, max_samples=None):
    """Evaluate on RTE (Recognizing Textual Entailment)."""
    dataset = load_dataset("glue", "rte", split="validation", cache_dir=GENERAL_DATASETS_DIR)
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    correct = 0
    total = 0
    
    for item in dataset:
        premise = item["sentence1"]
        hypothesis = item["sentence2"]
        label = item["label"]  # 0 = entailment, 1 = not_entailment
        
        prompt = f"Premise: {premise}\nHypothesis: {hypothesis}\nDoes the premise entail the hypothesis? Answer:"
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
        
        # Get probabilities for "yes" and "no"
        yes_tokens = tokenizer.encode(" yes", add_special_tokens=False)
        no_tokens = tokenizer.encode(" no", add_special_tokens=False)
        
        yes_prob = probs[0, yes_tokens[0]].item()
        no_prob = probs[0, no_tokens[0]].item()
        
        predicted = 0 if yes_prob > no_prob else 1
        if predicted == label:
            correct += 1
        total += 1
    
    accuracy = correct / total if total > 0 else 0
    return accuracy, correct, total


def evaluate_hellaswag(model, tokenizer, device, max_samples=None):
    """Evaluate on HellaSwag (commonsense reasoning)."""
    dataset = load_dataset("hellaswag", split="validation", cache_dir=GENERAL_DATASETS_DIR)
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    correct = 0
    total = 0
    
    for item in dataset:
        context = item["ctx"]
        endings = item["endings"]
        label = int(item["label"])
        
        # Score each ending
        scores = []
        for ending in endings:
            prompt = context + " " + ending
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
            
            with torch.no_grad():
                outputs = model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss.item()
                scores.append(-loss)  # Lower loss = higher score
        
        predicted = np.argmax(scores)
        if predicted == label:
            correct += 1
        total += 1
    
    accuracy = correct / total if total > 0 else 0
    return accuracy, correct, total


def evaluate_winogrande(model, tokenizer, device, max_samples=None):
    """Evaluate on WinoGrande (pronoun resolution)."""
    dataset = load_dataset("winogrande", "winogrande_xl", split="validation", cache_dir=GENERAL_DATASETS_DIR)
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    correct = 0
    total = 0
    
    for item in dataset:
        sentence = item["sentence"]
        option1 = item["option1"]
        option2 = item["option2"]
        label = int(item["answer"])  # 1 or 2
        
        # Replace _ with each option
        prompt1 = sentence.replace("_", option1)
        prompt2 = sentence.replace("_", option2)
        
        # Score each completion
        scores = []
        for prompt in [prompt1, prompt2]:
            inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
            
            with torch.no_grad():
                outputs = model(**inputs, labels=inputs["input_ids"])
                loss = outputs.loss.item()
                scores.append(-loss)
        
        predicted = 1 if scores[0] > scores[1] else 2
        if predicted == label:
            correct += 1
        total += 1
    
    accuracy = correct / total if total > 0 else 0
    return accuracy, correct, total


def evaluate_arc(model, tokenizer, device, difficulty="easy", max_samples=None):
    """Evaluate on ARC (AI2 Reasoning Challenge)."""
    split_name = "ARC-Easy" if difficulty == "easy" else "ARC-Challenge"
    dataset = load_dataset("ai2_arc", split_name, split="test", cache_dir=GENERAL_DATASETS_DIR)
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    correct = 0
    total = 0
    
    for item in dataset:
        question = item["question"]
        choices = item["choices"]["text"]
        labels = item["choices"]["label"]
        answer_key = item["answerKey"]
        
        # Find correct answer index
        try:
            correct_idx = labels.index(answer_key)
        except ValueError:
            # Some datasets use numeric labels
            correct_idx = int(answer_key) - 1 if answer_key.isdigit() else 0
        
        prompt = f"Question: {question}\n"
        for i, choice in enumerate(choices):
            prompt += f"{labels[i]}. {choice}\n"
        prompt += "Answer:"
        
        # Get log probs for each choice label
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
        
        choice_probs = []
        for label in labels:
            token_id = tokenizer.encode(" " + label, add_special_tokens=False)[0]
            choice_probs.append(probs[0, token_id].item())
        
        predicted = np.argmax(choice_probs)
        if predicted == correct_idx:
            correct += 1
        total += 1
    
    accuracy = correct / total if total > 0 else 0
    return accuracy, correct, total


def evaluate_openbookqa(model, tokenizer, device, max_samples=None):
    """Evaluate on OpenbookQA (open book question answering)."""
    dataset = load_dataset("openbookqa", "main", split="test", cache_dir=GENERAL_DATASETS_DIR)
    
    if max_samples:
        dataset = dataset.select(range(min(max_samples, len(dataset))))
    
    correct = 0
    total = 0
    
    for item in dataset:
        question = item["question_stem"]
        choices = item["choices"]["text"]
        labels = item["choices"]["label"]
        answer_key = item["answerKey"]
        
        # Find correct answer index
        try:
            correct_idx = labels.index(answer_key)
        except ValueError:
            correct_idx = ord(answer_key) - ord('A')
        
        prompt = f"Question: {question}\n"
        for i, choice in enumerate(choices):
            prompt += f"{labels[i]}. {choice}\n"
        prompt += "Answer:"
        
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits[:, -1, :]
            probs = torch.softmax(logits, dim=-1)
        
        choice_probs = []
        for label in labels:
            token_id = tokenizer.encode(" " + label, add_special_tokens=False)[0]
            choice_probs.append(probs[0, token_id].item())
        
        predicted = np.argmax(choice_probs)
        if predicted == correct_idx:
            correct += 1
        total += 1
    
    accuracy = correct / total if total > 0 else 0
    return accuracy, correct, total


def evaluate_general_nlp(model, tokenizer, device, max_samples=None, datasets=None):
    """Evaluate on zero-shot datasets."""
    results = {}
    
    # If no specific datasets provided, run all datasets
    if datasets is None:
        datasets = ['boolq', 'rte', 'hellaswag', 'winogrande', 'arc_easy', 'arc_challenge', 'openbookqa']
    
    # Run specified datasets
    if 'boolq' in datasets:
        accuracy, correct, total = evaluate_boolq(model, tokenizer, device, max_samples)
        results["boolq"] = {"accuracy": accuracy, "correct": correct, "total": total}
    
    if 'rte' in datasets:
        accuracy, correct, total = evaluate_rte(model, tokenizer, device, max_samples)
        results["rte"] = {"accuracy": accuracy, "correct": correct, "total": total}
    
    if 'hellaswag' in datasets:
        accuracy, correct, total = evaluate_hellaswag(model, tokenizer, device, max_samples)
        results["hellaswag"] = {"accuracy": accuracy, "correct": correct, "total": total}
    
    if 'winogrande' in datasets:
        accuracy, correct, total = evaluate_winogrande(model, tokenizer, device, max_samples)
        results["winogrande"] = {"accuracy": accuracy, "correct": correct, "total": total}
    
    if 'arc_easy' in datasets:
        accuracy, correct, total = evaluate_arc(model, tokenizer, device, "easy", max_samples)
        results["arc_easy"] = {"accuracy": accuracy, "correct": correct, "total": total}
    
    if 'arc_challenge' in datasets:
        accuracy, correct, total = evaluate_arc(model, tokenizer, device, "challenge", max_samples)
        results["arc_challenge"] = {"accuracy": accuracy, "correct": correct, "total": total}
    
    if 'openbookqa' in datasets:
        accuracy, correct, total = evaluate_openbookqa(model, tokenizer, device, max_samples)
        results["openbookqa"] = {"accuracy": accuracy, "correct": correct, "total": total}
    
    # Calculate average
    total_correct = sum([results[task]["correct"] for task in results])
    total_questions = sum([results[task]["total"] for task in results])
    avg_accuracy = total_correct / total_questions if total_questions > 0 else 0
    
    results["overall"] = {
        "accuracy": avg_accuracy,
        "correct": total_correct,
        "total": total_questions
    }
    
    return results