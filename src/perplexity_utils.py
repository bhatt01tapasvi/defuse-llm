import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from datasets import load_dataset, load_from_disk
import math
import os

from src.mmlu_utils import get_mmlu_prompt

def load_corpus(dataset_path: str, text_column: str = "text", max_samples: int = None):
    """Load corpus from disk or text file."""
    texts = []
    
    # Try loading as HuggingFace dataset
    if os.path.isdir(dataset_path):
        try:
            dataset = load_from_disk(dataset_path)
            texts = dataset[text_column] if text_column in dataset.column_names else dataset["text"]
        except Exception as e:
            print(f"Failed to load as HF dataset: {e}")
    
    # # Try loading as text file
    # elif dataset_path.endswith('.txt'):
    #     with open(dataset_path, 'r', encoding='utf-8') as f:
    #         content = f.read()
    #         texts = [p.strip() for p in content.split('\n\n') if p.strip()]
    
    # Try loading from datasets directory by name
    else:
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    if max_samples:
        texts = texts[:max_samples]
    
    return texts

def calculate_perplexity(model, tokenizer, texts: list[str], max_length: int = 512, stride: int = 256, device: str = "cuda"):
    """Calculate perplexity using sliding window approach for long texts."""
    model.eval()
    
    total_loss = 0.0
    total_tokens = 0
    
    for text in texts:
        if not text.strip():
            continue
        
        encodings = tokenizer(text, return_tensors="pt", truncation=False, add_special_tokens=True)
        input_ids = encodings.input_ids[0]
        seq_len = input_ids.size(0)
        
        if seq_len < 2:
            continue
        
        prev_end = 0
        for begin in range(0, seq_len, stride):
            end = min(begin + max_length, seq_len)
            chunk_ids = input_ids[begin:end].unsqueeze(0).to(device)
            
            trg_start = max(0, prev_end - begin) if begin > 0 else 0
            
            with torch.no_grad():
                outputs = model(chunk_ids, labels=chunk_ids)
                logits = outputs.logits[:, :-1, :]
                labels = chunk_ids[:, 1:]
                
                loss_fct = torch.nn.CrossEntropyLoss(reduction='none')
                token_losses = loss_fct(logits.reshape(-1, logits.size(-1)), labels.reshape(-1))
                
                if begin > 0 and trg_start > 0:
                    token_losses = token_losses[trg_start:]
                
                total_loss += token_losses.sum().item()
                total_tokens += token_losses.size(0)
            
            prev_end = end
            if end >= seq_len:
                break
    
    if total_tokens == 0:
        return float('inf')
    
    return math.exp(total_loss / total_tokens)

def calculate_perplexity_builtin(model, tokenizer, texts: list[str], max_length: int = 512, stride: int = 256, device: str = "cuda"):
    """
    Calculate perplexity using model's built-in loss computation.
    This is simpler and more efficient than manual loss calculation.
    """
    model.eval()
    
    total_loss = 0.0
    total_tokens = 0
    
    for text in texts:
        if not text.strip():
            continue
        
        # Tokenize without truncation
        encodings = tokenizer(text, return_tensors="pt", truncation=False, add_special_tokens=True)
        input_ids = encodings.input_ids[0]
        seq_len = input_ids.size(0)
        
        if seq_len < 2:
            continue
        
        prev_end = 0
        for begin in range(0, seq_len, stride):
            end = min(begin + max_length, seq_len)
            chunk_ids = input_ids[begin:end].unsqueeze(0).to(device)
            
            # Calculate how many tokens to actually count (avoid double-counting overlaps)
            trg_len = end - prev_end if begin > 0 else end - begin
            
            # Prepare target labels (same as input for causal LM)
            target_ids = chunk_ids.clone()
            
            # Mask out tokens that were already counted in previous window
            if begin > 0:
                overlap = prev_end - begin
                if overlap > 0:
                    target_ids[:, :overlap] = -100  # -100 is ignored by CrossEntropyLoss
            
            with torch.no_grad():
                # Model computes loss internally when labels are provided
                outputs = model(chunk_ids, labels=target_ids)
                loss = outputs.loss  # Already averaged per token
                
                # Count only the non-masked tokens
                valid_tokens = (target_ids != -100).sum().item()
                
                # Accumulate loss (multiply by number of tokens to get total)
                total_loss += loss.item() * valid_tokens
                total_tokens += valid_tokens
            
            prev_end = end
            if end >= seq_len:
                break
    
    if total_tokens == 0:
        return float('inf')
    
    # Perplexity = exp(average loss)
    return math.exp(total_loss / total_tokens)

def evaluate_on_datasets(
    model,
    tokenizer,
    datasets: list[tuple[str, str, str]],
    device: str = "cuda",
    max_samples: int = None,
    max_length: int = 512
):
    """Evaluate perplexity across multiple datasets."""
    results = {}
    
    for dataset in datasets:
        
        try:
            if "custom" == dataset[0]:
                dataset_path = dataset[2]
                texts = load_corpus(dataset_path)
                print(f"Loaded {len(texts)} documents")
            elif "mmlu" == dataset[0]:
                texts = get_mmlu_prompt(dataset[1])
                print(f"Loaded {len(texts)} MMLU questions for subject: {dataset[1]}")

            manual_ppl = calculate_perplexity(
                model, tokenizer, texts,
                max_length=max_length,
                device=device
            )

            builtin_ppl = calculate_perplexity_builtin(
                model, tokenizer, texts,
                max_length=max_length,
                device=device
            )
            
            results[dataset[0]+"_"+dataset[1]] = {
                "manual": manual_ppl,
                "builtin": builtin_ppl,
                "difference": abs(manual_ppl - builtin_ppl)
            }
            
            print(f"Perplexity: {manual_ppl:.2f}")
            print(f"Perplexity (builtin): {builtin_ppl:.2f}")
            
        except Exception as e:
            print(f"Error evaluating {dataset[0]}: {e}")
            results[dataset[0]] = {'error': str(e)}
    
    return results