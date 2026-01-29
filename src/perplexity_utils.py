import torch
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
    
    # Try loading as text file
    elif dataset_path.endswith('.txt'):
        with open(dataset_path, 'r', encoding='utf-8') as f:
            content = f.read()
            texts = [p.strip() for p in content.split('\n\n') if p.strip()]
    
    # Try loading from datasets directory by name
    else:
        raise FileNotFoundError(f"Dataset not found: {dataset_path}")
    
    if max_samples:
        texts = texts[:max_samples]
    
    return texts

def calculate_perplexity(model, tokenizer, texts: list[str], max_length: int = 512, stride: int = 256, device: str = "cuda"):
    """Calculate perplexity using sliding window approach for long texts."""
    model.eval()
    
    # Join all texts into one continuous sequence (standard evaluation protocol)
    # Filter out None and empty texts before processing
    valid_texts = [text for text in texts if text is not None]
    full_text = "\n\n".join([text.strip() for text in valid_texts if text.strip()])
    
    # Tokenize the full text
    encodings = tokenizer(full_text, return_tensors="pt", truncation=False, add_special_tokens=True)
    input_ids = encodings.input_ids[0].to(device)
    seq_len = input_ids.size(0)
    
    if seq_len < 2:
        return float('inf')
    
    total_loss = 0.0
    total_tokens = 0
    
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
    
    # Join all texts into one continuous sequence (standard evaluation protocol)
    # Filter out None and empty texts before processing
    valid_texts = [text for text in texts if text is not None]
    full_text = "\n\n".join([text.strip() for text in valid_texts if text.strip()])
    
    # Tokenize the full text
    encodings = tokenizer(full_text, return_tensors="pt", truncation=False, add_special_tokens=True)
    input_ids = encodings.input_ids[0].to(device)
    seq_len = input_ids.size(0)

    if seq_len < 2:
        return float('inf')
    
    total_loss = 0.0
    total_tokens = 0
    
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
    model,                              # Pre-trained language model (e.g., GPT-2, LLaMA) for evaluation
    tokenizer,                          # Tokenizer corresponding to the model (for encoding text)
    datasets: list[tuple[str, str, str]],  # List of tuples: [(dataset_type, subject/name, path), ...]
                                        # - dataset_type (str): "custom" for local datasets, "mmlu" for MMLU benchmark
                                        # - subject/name (str): Dataset identifier (e.g., "imc", "college_computer_science")
                                        # - path (str): File system path to dataset (used only for "custom" type)
                                        # Example: [("custom", "imc", "/path/to/imc_dataset"), 
                                        #           ("mmlu", "college_computer_science", "")]
    device: str = "cuda",               # Device to run computations on ("cuda" for GPU, "cpu" for CPU)
    max_samples: int = None,            # Maximum number of samples/documents to evaluate per dataset
                                        # None = evaluate all available samples
                                        # Useful for quick testing or limiting computation time
    max_length_ppl: int = 512           # Maximum sequence length (in tokens) for each evaluation chunk
                                        # Longer sequences are split into overlapping windows
                                        # Should not exceed model's max_position_embeddings
                                        # Typical values: 512 (GPT-2), 2048 (LLaMA-1), 4096 (LLaMA-2)
):
    """
    Evaluate perplexity across multiple datasets.
    
    Args:
        model: Pre-trained language model for evaluation
        tokenizer: Tokenizer for encoding text into tokens
        datasets: List of dataset tuples, where each tuple contains:
            - dataset_type: "custom" (local dataset) or "mmlu" (MMLU benchmark)
            - subject: Dataset identifier/name (e.g., "imc", "college_computer_science")
            - path: File path (required for "custom", empty string for "mmlu")
        device: Computation device ("cuda" or "cpu")
        max_samples: Maximum number of samples to evaluate per dataset (None = all)
        max_length_ppl: Maximum token length for perplexity calculation chunks
    
    Returns:
        dict: Results dictionary mapping dataset names to perplexity metrics
            {
                "custom_imc": {
                    "manual": 15.23,
                    "builtin": 15.25,
                    "difference": 0.02
                },
                "mmlu_college_computer_science": {...}
            }
    
    Example:
        >>> datasets = [
        ...     ("custom", "imc", "/path/to/imc_dataset"),
        ...     ("mmlu", "college_computer_science", "")
        ... ]
        >>> results = evaluate_on_datasets(
        ...     model, tokenizer, datasets,
        ...     device="cuda", max_samples=50, max_length_ppl=512
        ... )
    """
    results = {}
        
    if max_length_ppl > model.config.max_position_embeddings:
        raise ValueError(f"max_length_ppl {max_length_ppl} exceeds model's max_position_embeddings {model.config.max_position_embeddings}")

    for dataset in datasets:
        texts = None
        try:
            dataset_type = dataset[0]
            dataset_name = dataset[1]
            dataset_path = dataset[2] if len(dataset) > 2 else ""
            
            # Load texts based on dataset type
            if dataset_type == "custom":
                texts = load_corpus(dataset_path, max_samples=max_samples)
                print(f"Loaded {len(texts)} documents from {dataset_name}")
            
            elif dataset_type == "mmlu":
                texts = get_mmlu_prompt(dataset_name, max_samples=max_samples)
                print(f"Loaded {len(texts)} MMLU questions for subject: {dataset_name}")
            
            elif dataset_type in ["wikitext2", "wikitext-2"]:
                print(f"Loading WikiText-2 dataset...")
                testdata = load_dataset('wikitext', 'wikitext-2-raw-v1', split='test')
                # FIXED: Filter out None values
                texts = [
                    text for text in testdata['text'] 
                    if text is not None and text.strip()  # ← ADD None check
                ]
                if max_samples:
                    texts = texts[:max_samples]
                print(f"Loaded {len(texts)} documents from WikiText-2")
                        
            elif dataset_type in ["c4", "c4-validation"]:
                print(f"Loading C4 validation dataset...")
                testdata = load_dataset('allenai/c4', 'realnewslike', split='validation', streaming=True)
                texts = []
                for i, item in enumerate(testdata):
                    if max_samples and i >= max_samples:
                        break
                    # FIXED: Check if text exists and is not None
                    if 'text' in item and item['text'] is not None and item['text'].strip():
                        texts.append(item['text'])
                print(f"Loaded {len(texts)} documents from C4")
            
            else:
                raise ValueError(f"Unknown dataset type: {dataset_type}")

            manual_ppl = calculate_perplexity(
                model, tokenizer, texts,
                max_length=max_length_ppl,
                device=device
            )

            builtin_ppl = calculate_perplexity_builtin(
                model, tokenizer, texts,
                max_length= max_length_ppl,
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