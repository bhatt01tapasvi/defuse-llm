# """
# Text Similarity Evaluation Script - Enhanced Version
# Evaluates semantic and lexical drift in pruned language models by comparing
# outputs against unpruned baseline using BLEU, BERTScore, and sentence embeddings.

# Includes:
# - IDF weighting for BERTScore
# - Sentence-level matching for coverage and coherence analysis
# - Chunking for very long texts
# """

# import evaluate
# from sentence_transformers import SentenceTransformer, util
# import torch
# from typing import List, Dict, Tuple
# import numpy as np
# import warnings
# from collections import Counter
# import math
# import nltk
# import json
# import os

# # Download sentence tokenizer if needed
# try:
#     nltk.data.find('tokenizers/punkt_tab')
# except LookupError:
#     print("Downloading NLTK punkt_tab tokenizer...")
#     nltk.download('punkt_tab')


# class TextSimilarityEvaluator:
#     """
#     Enhanced evaluator with:
#     - BERTScore with optional IDF weighting
#     - Sentence-level matching for coverage/coherence
#     - Document-level embedding similarity
#     - Chunking support for long texts
#     """
    
#     def __init__(
#         self, 
#         embedding_model: str = 'all-mpnet-base-v2',
#         bertscore_model: str = 'microsoft/deberta-xlarge-mnli',
#         use_baseline_rescaling: bool = False,
#         use_idf: bool = True,
#         max_chunk_tokens: int = 512,
#         verbose: bool = True
#     ):
#         """
#         Initialize evaluator with controlled model specifications.
        
#         Args:
#             embedding_model: Sentence-transformers model name
#             bertscore_model: Specific BERT model for BERTScore
#             use_baseline_rescaling: Whether to use baseline rescaling in BERTScore
#             use_idf: Whether to use IDF weighting in BERTScore
#             max_chunk_tokens: Maximum tokens per chunk for long texts
#             verbose: Print loading messages
#         """
#         self.verbose = verbose
#         self.bertscore_model = bertscore_model
#         self.use_baseline_rescaling = use_baseline_rescaling
#         self.use_idf = use_idf
#         self.max_chunk_tokens = max_chunk_tokens
        
#         if self.verbose:
#             print("Loading evaluation metrics...")
#             print(f"  - BLEU (diagnostic metric for lexical drift)")
#             print(f"  - BERTScore (model: {bertscore_model}, IDF: {use_idf})")
#             print(f"  - Sentence embeddings (model: {embedding_model})")
#             print(f"  - Max chunk size: {max_chunk_tokens} tokens")
        
#         self.bleu_metric = evaluate.load("bleu")
#         self.bertscore_metric = evaluate.load("bertscore")
#         self.sentence_model = SentenceTransformer(embedding_model)
        
#         # For IDF computation
#         self.idf_dict = None
        
#         if self.verbose:
#             print("All models loaded successfully!\n")
    
#     def compute_idf(self, texts: List[str]) -> Dict[str, float]:
#         """
#         Compute IDF weights from a corpus of texts.
        
#         Args:
#             texts: List of texts to compute IDF from
            
#         Returns:
#             Dictionary mapping tokens to IDF weights
#         """
#         from collections import defaultdict
#         doc_freq = defaultdict(int)
#         total_docs = len(texts)
        
#         for text in texts:
#             tokens = set(text.lower().split())
#             for token in tokens:
#                 doc_freq[token] += 1
        
#         # Compute IDF: log(N / df)
#         idf_dict = {}
#         for token, freq in doc_freq.items():
#             idf_dict[token] = math.log(total_docs / freq)
        
#         return idf_dict
    
#     def compute_bleu(
#         self, 
#         predictions: List[str], 
#         references: List[str]
#     ) -> Dict:
#         """
#         Compute BLEU score between predictions and references.
        
#         NOTE: BLEU is designed for corpus-level MT evaluation and is used here
#         only as a diagnostic signal for catastrophic lexical drift.
        
#         Args:
#             predictions: List of generated texts
#             references: List of reference texts
            
#         Returns:
#             Dictionary with BLEU scores and statistics
#         """
#         references_formatted = [[ref] for ref in references]
#         result = self.bleu_metric.compute(
#             predictions=predictions,
#             references=references_formatted
#         )
        
#         if result['bleu'] == 0.0:
#             warnings.warn(
#                 "BLEU score is 0.0, which may indicate severe lexical drift "
#                 "or very short/paraphrased outputs."
#             )
        
#         return result
    
#     def chunk_text(self, text: str, max_tokens: int = 512) -> List[str]:
#         """
#         Split text into chunks if it's very long.
        
#         Args:
#             text: Input text
#             max_tokens: Maximum tokens per chunk (approximate)
            
#         Returns:
#             List of text chunks
#         """
#         # Rough approximation: 1 token ≈ 4 characters
#         max_chars = max_tokens * 4
        
#         if len(text) <= max_chars:
#             return [text]
        
#         # Split into sentences first
#         sentences = nltk.sent_tokenize(text)
        
#         chunks = []
#         current_chunk = []
#         current_length = 0
        
#         for sentence in sentences:
#             sentence_length = len(sentence)
            
#             if current_length + sentence_length > max_chars and current_chunk:
#                 chunks.append(' '.join(current_chunk))
#                 current_chunk = [sentence]
#                 current_length = sentence_length
#             else:
#                 current_chunk.append(sentence)
#                 current_length += sentence_length
        
#         if current_chunk:
#             chunks.append(' '.join(current_chunk))
        
#         return chunks
    
#     def compute_bertscore(
#         self, 
#         predictions: List[str], 
#         references: List[str],
#         lang: str = "en"
#     ) -> Dict:
#         """
#         Compute BERTScore with optional IDF weighting and chunking.
        
#         Args:
#             predictions: List of generated texts
#             references: List of reference texts
#             lang: Language code (default: "en")
            
#         Returns:
#             Dictionary with precision, recall, F1 scores (mean and std)
#         """
#         all_precision = []
#         all_recall = []
#         all_f1 = []
        
#         # Compute IDF if enabled and not already computed
#         if self.use_idf and self.idf_dict is None:
#             if self.verbose:
#                 print("  Computing IDF weights from corpus...")
#             all_texts = predictions + references
#             self.idf_dict = self.compute_idf(all_texts)
        
#         # Process each prediction-reference pair
#         for pred, ref in zip(predictions, references):
#             # Check if texts need chunking
#             pred_chunks = self.chunk_text(pred, self.max_chunk_tokens)
#             ref_chunks = self.chunk_text(ref, self.max_chunk_tokens)
            
#             if len(pred_chunks) > 1 or len(ref_chunks) > 1:
#                 if self.verbose and len(all_precision) == 0:
#                     print(f"  Text requires chunking: {len(pred_chunks)} pred chunks, {len(ref_chunks)} ref chunks")
                
#                 # Compute BERTScore for each chunk pair and average
#                 chunk_scores = []
#                 for p_chunk in pred_chunks:
#                     for r_chunk in ref_chunks:
#                         result = self.bertscore_metric.compute(
#                             predictions=[p_chunk],
#                             references=[r_chunk],
#                             lang=lang,
#                             model_type=self.bertscore_model,
#                             idf=self.use_idf,
#                             rescale_with_baseline=self.use_baseline_rescaling
#                         )
#                         chunk_scores.append({
#                             'precision': result['precision'][0],
#                             'recall': result['recall'][0],
#                             'f1': result['f1'][0]
#                         })
                
#                 # Average across chunks
#                 avg_precision = np.mean([s['precision'] for s in chunk_scores])
#                 avg_recall = np.mean([s['recall'] for s in chunk_scores])
#                 avg_f1 = np.mean([s['f1'] for s in chunk_scores])
#             else:
#                 # No chunking needed
#                 result = self.bertscore_metric.compute(
#                     predictions=[pred],
#                     references=[ref],
#                     lang=lang,
#                     model_type=self.bertscore_model,
#                     idf=self.use_idf,
#                     rescale_with_baseline=self.use_baseline_rescaling
#                 )
#                 avg_precision = result['precision'][0]
#                 avg_recall = result['recall'][0]
#                 avg_f1 = result['f1'][0]
            
#             all_precision.append(avg_precision)
#             all_recall.append(avg_recall)
#             all_f1.append(avg_f1)
        
#         # Calculate statistics
#         precision_scores = np.array(all_precision)
#         recall_scores = np.array(all_recall)
#         f1_scores = np.array(all_f1)
        
#         return {
#             'precision_mean': np.mean(precision_scores),
#             'precision_std': np.std(precision_scores),
#             'recall_mean': np.mean(recall_scores),
#             'recall_std': np.std(recall_scores),
#             'f1_mean': np.mean(f1_scores),
#             'f1_std': np.std(f1_scores),
#             'precision_list': all_precision,
#             'recall_list': all_recall,
#             'f1_list': all_f1,
#             'using_idf': self.use_idf
#         }
    
#     def compute_sentence_matching(
#         self,
#         predictions: List[str],
#         references: List[str],
#         similarity_threshold: float = 0.5
#     ) -> Dict:
#         """
#         Compute sentence-level matching between predictions and references.
        
#         Reveals:
#         - Coverage: What % of reference sentences are well-represented?
#         - Coherence: Are sentences in similar order?
#         - Per-sentence quality: Distribution of matches
        
#         Args:
#             predictions: List of generated texts
#             references: List of reference texts
#             similarity_threshold: Threshold for considering a sentence "matched"
            
#         Returns:
#             Dictionary with sentence-level metrics
#         """
#         all_coverages = []
#         all_avg_similarities = []
#         all_order_correlations = []
#         all_best_matches = []
        
#         for pred, ref in zip(predictions, references):
#             # Split into sentences
#             pred_sentences = nltk.sent_tokenize(pred)
#             ref_sentences = nltk.sent_tokenize(ref)
            
#             if len(pred_sentences) == 0 or len(ref_sentences) == 0:
#                 continue
            
#             # Encode all sentences
#             pred_embeddings = self.sentence_model.encode(
#                 pred_sentences, 
#                 convert_to_tensor=True,
#                 show_progress_bar=False
#             )
#             ref_embeddings = self.sentence_model.encode(
#                 ref_sentences,
#                 convert_to_tensor=True,
#                 show_progress_bar=False
#             )
            
#             # Compute similarity matrix (ref_sentences x pred_sentences)
#             similarity_matrix = util.cos_sim(ref_embeddings, pred_embeddings).cpu().numpy()
            
#             # For each reference sentence, find best match in prediction
#             best_matches = similarity_matrix.max(axis=1)
#             best_match_indices = similarity_matrix.argmax(axis=1)
            
#             # Coverage: % of reference sentences with similarity > threshold
#             coverage = (best_matches >= similarity_threshold).mean()
            
#             # Average similarity of best matches
#             avg_similarity = best_matches.mean()
            
#             # Order correlation: Do matched sentences appear in similar order?
#             if len(best_matches) > 1:
#                 ref_indices = np.arange(len(ref_sentences))
#                 try:
#                     from scipy.stats import spearmanr
#                     order_corr = spearmanr(ref_indices, best_match_indices)[0]
#                     if np.isnan(order_corr):
#                         order_corr = 0.0
#                 except:
#                     # Fallback if scipy not available
#                     order_corr = 0.0
#             else:
#                 order_corr = 1.0
            
#             all_coverages.append(coverage)
#             all_avg_similarities.append(avg_similarity)
#             all_order_correlations.append(order_corr)
#             all_best_matches.extend(best_matches.tolist())
        
#         return {
#             'coverage_mean': np.mean(all_coverages),
#             'coverage_std': np.std(all_coverages),
#             'avg_similarity_mean': np.mean(all_avg_similarities),
#             'avg_similarity_std': np.std(all_avg_similarities),
#             'order_correlation_mean': np.mean(all_order_correlations),
#             'order_correlation_std': np.std(all_order_correlations),
#             'threshold_used': similarity_threshold,
#             'all_sentence_similarities': all_best_matches,
#             'coverage_list': all_coverages,
#             'order_correlation_list': all_order_correlations
#         }
    
#     def compute_embedding_similarity(
#         self, 
#         predictions: List[str], 
#         references: List[str]
#     ) -> Dict:
#         """
#         Compute document-level cosine similarity using sentence embeddings.
        
#         Args:
#             predictions: List of generated texts
#             references: List of reference texts
            
#         Returns:
#             Dictionary with similarity statistics
#         """
#         pred_embeddings = self.sentence_model.encode(
#             predictions, 
#             convert_to_tensor=True,
#             show_progress_bar=False
#         )
#         ref_embeddings = self.sentence_model.encode(
#             references, 
#             convert_to_tensor=True,
#             show_progress_bar=False
#         )
        
#         similarity_matrix = util.cos_sim(pred_embeddings, ref_embeddings)
#         similarities = similarity_matrix.diagonal().cpu().numpy()
        
#         return {
#             'mean': float(np.mean(similarities)),
#             'std': float(np.std(similarities)),
#             'min': float(np.min(similarities)),
#             'max': float(np.max(similarities)),
#             'median': float(np.median(similarities)),
#             'similarity_list': similarities.tolist()
#         }
    
#     def evaluate_all(
#         self, 
#         predictions: List[str], 
#         references: List[str],
#         lang: str = "en",
#         compute_sentence_level: bool = True
#     ) -> Dict:
#         """
#         Compute all metrics at once.
        
#         Args:
#             predictions: List of generated texts (pruned model)
#             references: List of reference texts (unpruned model)
#             lang: Language code for BERTScore
#             compute_sentence_level: Whether to compute sentence-level matching
            
#         Returns:
#             Dictionary containing all metric results
#         """
#         if len(predictions) != len(references):
#             raise ValueError(
#                 f"Number of predictions ({len(predictions)}) must match "
#                 f"number of references ({len(references)})"
#             )
        
#         if self.verbose:
#             print(f"Evaluating {len(predictions)} text pairs...\n")
        
#         if self.verbose:
#             print("Computing BLEU score (lexical drift diagnostic)...")
#         bleu_result = self.compute_bleu(predictions, references)
        
#         if self.verbose:
#             print(f"Computing BERTScore (IDF: {self.use_idf})...")
#         bertscore_result = self.compute_bertscore(predictions, references, lang)
        
#         if self.verbose:
#             print("Computing document-level embedding similarity...")
#         embedding_result = self.compute_embedding_similarity(predictions, references)
        
#         results = {
#             'bleu': bleu_result,
#             'bertscore': bertscore_result,
#             'embedding_similarity': embedding_result
#         }
        
#         if compute_sentence_level:
#             if self.verbose:
#                 print("Computing sentence-level matching...")
#             sentence_result = self.compute_sentence_matching(predictions, references)
#             results['sentence_matching'] = sentence_result
        
#         return results
    
#     def compare_multiple_models(
#         self,
#         references: List[str],
#         model_outputs: Dict[str, List[str]],
#         lang: str = "en",
#         compute_sentence_level: bool = True
#     ) -> Dict[str, Dict]:
#         """
#         Compare multiple model outputs against the same references.
        
#         Args:
#             references: List of reference texts (unpruned baseline)
#             model_outputs: Dictionary mapping model names to their outputs
#             lang: Language code for BERTScore
#             compute_sentence_level: Whether to compute sentence-level metrics
            
#         Returns:
#             Dictionary mapping model names to evaluation results
#         """
#         results = {}
        
#         for model_name, predictions in model_outputs.items():
#             if len(predictions) != len(references):
#                 raise ValueError(
#                     f"Model '{model_name}' has {len(predictions)} outputs but "
#                     f"there are {len(references)} references"
#                 )
            
#             if self.verbose:
#                 print(f"\n{'='*60}")
#                 print(f"Evaluating: {model_name}")
#                 print(f"{'='*60}")
            
#             results[model_name] = self.evaluate_all(
#                 predictions, 
#                 references, 
#                 lang,
#                 compute_sentence_level=compute_sentence_level
#             )
        
#         return results
    
#     def print_results(self, results: Dict, model_name: str = None):
#         """Print evaluation results with statistics."""
#         header = f"EVALUATION RESULTS - {model_name}" if model_name else "EVALUATION RESULTS"
#         print("\n" + "="*80)
#         print(header)
#         print("="*80)
        
#         print("\n[BLEU Score - Lexical Drift Diagnostic]")
#         print(f"  Score: {results['bleu']['bleu']:.4f}")
        
#         print("\n[BERTScore - Semantic Similarity]")
#         print(f"  Using IDF: {results['bertscore'].get('using_idf', False)}")
#         print(f"  Precision: {results['bertscore']['precision_mean']:.4f} "
#               f"(±{results['bertscore']['precision_std']:.4f})")
#         print(f"  Recall:    {results['bertscore']['recall_mean']:.4f} "
#               f"(±{results['bertscore']['recall_std']:.4f})")
#         print(f"  F1:        {results['bertscore']['f1_mean']:.4f} "
#               f"(±{results['bertscore']['f1_std']:.4f})")
        
#         print("\n[Document-Level Embedding Similarity]")
#         print(f"  Mean:      {results['embedding_similarity']['mean']:.4f} "
#               f"(±{results['embedding_similarity']['std']:.4f})")
#         print(f"  Median:    {results['embedding_similarity']['median']:.4f}")
#         print(f"  Range:     [{results['embedding_similarity']['min']:.4f}, "
#               f"{results['embedding_similarity']['max']:.4f}]")
        
#         if 'sentence_matching' in results:
#             sm = results['sentence_matching']
#             print("\n[Sentence-Level Matching - Coverage & Coherence]")
#             print(f"  Coverage:          {sm['coverage_mean']:.4f} "
#                   f"(±{sm['coverage_std']:.4f})")
#             print(f"  Avg Similarity:    {sm['avg_similarity_mean']:.4f} "
#                   f"(±{sm['avg_similarity_std']:.4f})")
#             print(f"  Order Correlation: {sm['order_correlation_mean']:.4f} "
#                   f"(±{sm['order_correlation_std']:.4f})")
#             print(f"  Threshold:         {sm['threshold_used']:.2f}")
#             print(f"  → Coverage: % of reference sentences with good matches")
#             print(f"  → Order Correlation: 1.0=perfect order, 0.0=random, -1.0=reversed")
        
#         print("="*80 + "\n")
    
#     def print_comparison(self, comparison_results: Dict[str, Dict]):
#         """Print comparison table across models with statistics."""
#         print("\n" + "="*110)
#         print("COMPARISON SUMMARY")
#         print("="*110)
        
#         model_names = list(comparison_results.keys())
#         display_names = [name[-60:] if len(name) > 60 else name for name in model_names]
        
#         has_sentence = any('sentence_matching' in r for r in comparison_results.values())
        
#         if has_sentence:
#             print(f"\n{'Model':<62} {'BLEU':<10} {'BERTScore F1':<18} {'Embed Sim':<18} {'Coverage':<12}")
#             print("-" * 110)
#         else:
#             print(f"\n{'Model':<62} {'BLEU':<10} {'BERTScore F1':<18} {'Embed Sim':<18}")
#             print("-" * 98)
        
#         for model_name, display_name in zip(model_names, display_names):
#             res = comparison_results[model_name]
#             bleu = res['bleu']['bleu']
#             bert_f1_mean = res['bertscore']['f1_mean']
#             bert_f1_std = res['bertscore']['f1_std']
#             emb_mean = res['embedding_similarity']['mean']
#             emb_std = res['embedding_similarity']['std']
            
#             line = f"{display_name:<62} {bleu:<10.4f} {bert_f1_mean:.4f}(±{bert_f1_std:.3f}){'':<6} {emb_mean:.4f}(±{emb_std:.3f}){'':<6}"
            
#             if 'sentence_matching' in res:
#                 cov_mean = res['sentence_matching']['coverage_mean']
#                 cov_std = res['sentence_matching']['coverage_std']
#                 line += f" {cov_mean:.4f}(±{cov_std:.3f})"
            
#             print(line)
        
#         print("="*110 + "\n")
    
#     def export_results(self, results: Dict, filepath: str):
#         """Export results to JSON for further analysis."""
        
#         # Create directory if it doesn't exist
#         os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
#         def convert_to_serializable(obj):
#             """Convert numpy types to Python native types for JSON serialization."""
#             if isinstance(obj, dict):
#                 return {k: convert_to_serializable(v) for k, v in obj.items()}
#             elif isinstance(obj, list):
#                 return [convert_to_serializable(item) for item in obj]
#             elif isinstance(obj, np.ndarray):
#                 return obj.tolist()
#             elif isinstance(obj, (np.float32, np.float64)):
#                 return float(obj)
#             elif isinstance(obj, (np.int32, np.int64)):
#                 return int(obj)
#             elif isinstance(obj, np.bool_):
#                 return bool(obj)
#             else:
#                 return obj
        
#         serializable_results = convert_to_serializable(results)
        
#         with open(filepath, 'w') as f:
#             json.dump(serializable_results, f, indent=2)
#         print(f"Results exported to {filepath}")


# # ============================================================================
# # Helper Functions for Loading Data
# # ============================================================================


# def extract_generated_text_from_log(log_filepath: str) -> str:
#     """
#     Extract the 'Newly generated text' section from a log file.
    
#     Args:
#         log_filepath: Path to the log file
        
#     Returns:
#         The generated text content
#     """
#     with open(log_filepath, 'r', encoding='utf-8') as f:
#         content = f.read()
    
#     start_marker = "Newly generated text:"
#     end_marker = "=" * 80
    
#     start_idx = content.find(start_marker)
#     if start_idx == -1:
#         raise ValueError(f"Could not find '{start_marker}' in log file: {log_filepath}")
    
#     start_idx += len(start_marker)
#     end_idx = content.find(end_marker, start_idx)
    
#     if end_idx == -1:
#         extracted_text = content[start_idx:].strip()
#     else:
#         extracted_text = content[start_idx:end_idx].strip()
    
#     return extracted_text


# def extract_all_samples_from_log(log_filepath: str) -> List[Dict[str, str]]:
#     """
#     Extract all samples from a log file where each sample has:
#     - SAMPLE number
#     - Input Document
#     - Generated Summary
#     - Reference Summary
    
#     Args:
#         log_filepath: Path to the log file
        
#     Returns:
#         List of dictionaries, each containing:
#         - 'sample_id': Sample number
#         - 'generated': Generated summary text
#         - 'reference': Reference summary text
#     """
#     with open(log_filepath, 'r', encoding='utf-8') as f:
#         content = f.read()
    
#     samples = []
    
#     # Split by sample markers
#     sample_sections = content.split('---[[ SAMPLE ')
    
#     for section in sample_sections[1:]:  # Skip first empty section
#         try:
#             # Extract sample ID
#             sample_id_end = section.find(']]---')
#             sample_id = section[:sample_id_end].strip()
            
#             # Extract Generated Summary
#             # Try to find any "Generated Summary (XXX tokens):" format
#             gen_marker_base = 'Generated Summary ('
#             gen_start = section.find(gen_marker_base)
            
#             if gen_start != -1:
#                 # Find the end of the marker (the colon after the token count)
#                 colon_pos = section.find(':', gen_start)
#                 if colon_pos != -1:
#                     gen_start = colon_pos + 1
#                 else:
#                     gen_start = -1
            
#             # Fall back to simple "Generated Summary:" if parenthetical format not found
#             if gen_start == -1:
#                 gen_marker = 'Generated Summary:'
#                 gen_start = section.find(gen_marker)
#                 if gen_start != -1:
#                     gen_start += len(gen_marker)
            
#             if gen_start == -1:
#                 print(f"Warning: Could not find generated summary in sample {sample_id}")
#                 continue
            
#             # Find Reference Summary marker
#             ref_marker = 'Reference Summary:'
#             ref_start = section.find(ref_marker, gen_start)
            
#             if ref_start == -1:
#                 print(f"Warning: Could not find reference summary in sample {sample_id}")
#                 continue
            
#             # Extract generated text
#             generated_text = section[gen_start:ref_start].strip()
            
#             # Extract reference text
#             ref_start += len(ref_marker)
#             next_section = section.find('---END OF TEXT---', ref_start)
            
#             if next_section != -1:
#                 reference_text = section[ref_start:next_section].strip()
#             else:
#                 reference_text = section[ref_start:].strip()
            
#             samples.append({
#                 'sample_id': sample_id,
#                 'generated': generated_text,
#                 'reference': reference_text
#             })
            
#         except Exception as e:
#             print(f"Error processing sample: {e}")
#             continue
    
#     return samples

# def load_outputs_from_files(file_paths: List[str]) -> List[str]:
#     """
#     Load model outputs from multiple files.
    
#     Args:
#         file_paths: List of file paths to load
#         from_logs: If True, extract from log files; if False, load plain text files
        
#     Returns:
#         List of text outputs
#     """
#     outputs = []
#     for path in file_paths:
#         text = extract_generated_text_from_log(path)
#         outputs.append(text)
#     return outputs


# # ============================================================================
# # Main Comparison Function
# # ============================================================================

# def custom_prompt_analysis(path_unpruned: str = None, path_pruned: str = None, path_pruned_kd: str = None, output_file: str = None):
#     import glob

#     # Use provided paths or defaults
#     # if path_unpruned is None:
#     #     path_unpruned = "/users/grad/abhishektyagi/wanda/wanda/results/knowledge_drift/custom_prompts/default/meta-llama_Llama-3.2-3B-Instruct/"
#     # if path_pruned is None:
#     #     path_pruned = "/users/grad/abhishektyagi/wanda/wanda/results/knowledge_drift/custom_prompts/drift/meta-llama_Llama-3.2-3B-Instruct/"
#     # if path_pruned_kd is None:
#     #     path_pruned_kd = "/users/grad/abhishektyagi/wanda/wanda/results/knowledge_drift/custom_prompts/driftless_win5_winth5_1.5sigma0.1Max/meta-llama_Llama-3.2-3B-Instruct/"

#     if path_unpruned is None:
#         path_unpruned = "/users/grad/abhishektyagi/wanda/wanda/results/knowledge_drift/custom_prompts_TOT/dense_v2/meta-llama_Llama-3.2-3B-Instruct"
#     if path_pruned is None:
#         path_pruned = "/users/grad/abhishektyagi/wanda/wanda/results/knowledge_drift/custom_prompts_TOT/sparse_v2/meta-llama_Llama-3.2-3B-Instruct"
#     if path_pruned_kd is None:
#         path_pruned_kd = "/users/grad/abhishektyagi/wanda/wanda/results/knowledge_drift/custom_prompts_TOT/sparse_kd/meta-llama_Llama-3.2-3B-Instruct"


#     # Get all directories and build mappings by prompt ID
#     def get_prompt_id(dirname):
#         """Extract prompt ID (e.g., 'prompt0' from 'prompt0_something')"""
#         if dirname.startswith('prompt'):
#             # Find the first underscore after 'prompt'
#             underscore_pos = dirname.find('_')
#             if underscore_pos != -1:
#                 return dirname[:underscore_pos]
#             else:
#                 return dirname  # No underscore, use full name
#         return None
    
#     def build_dir_mapping(base_path):
#         """Build a mapping from prompt ID to full directory path"""
#         mapping = {}
#         if not os.path.exists(base_path):
#             return mapping
#         for d in os.listdir(base_path):
#             full_path = os.path.join(base_path, d)
#             if os.path.isdir(full_path):
#                 prompt_id = get_prompt_id(d)
#                 if prompt_id:
#                     mapping[prompt_id] = full_path
#         return mapping
    
#     unpruned_dirs = build_dir_mapping(path_unpruned)
#     pruned_dirs = build_dir_mapping(path_pruned)
#     pruned_kd_dirs = build_dir_mapping(path_pruned_kd)
    
#     # Find common prompt IDs across all three paths
#     common_prompt_ids = set(unpruned_dirs.keys()) & set(pruned_dirs.keys()) & set(pruned_kd_dirs.keys())
    
#     if not common_prompt_ids:
#         print("Error: No common prompt IDs found across all three paths!")
#         print(f"Unpruned has: {sorted(unpruned_dirs.keys())}")
#         print(f"Pruned has: {sorted(pruned_dirs.keys())}")
#         print(f"Pruned_kd has: {sorted(pruned_kd_dirs.keys())}")
#         return None
    
#     logs_unpruned = []
#     logs_pruned = []
#     logs_pruned_kd = []
    
#     for prompt_id in sorted(common_prompt_ids):
#         unpruned_dir = unpruned_dirs[prompt_id]
#         pruned_dir = pruned_dirs[prompt_id]
#         pruned_kd_dir = pruned_kd_dirs[prompt_id]
        
#         # Find output*.log files in each directory
#         unpruned_logs = glob.glob(os.path.join(unpruned_dir, "output*.log"))
#         pruned_logs = glob.glob(os.path.join(pruned_dir, "output*.log"))
#         pruned_kd_logs = glob.glob(os.path.join(pruned_kd_dir, "output*.log"))
        
#         # Check if all three directories have log files
#         if unpruned_logs and pruned_logs and pruned_kd_logs:
#             logs_unpruned.append(unpruned_logs[0])  # Take first log file
#             logs_pruned.append(pruned_logs[0])
#             logs_pruned_kd.append(pruned_kd_logs[0])
#         else:
#             print(f"Warning: Missing log files for prompt ID '{prompt_id}'")
#             if not unpruned_logs:
#                 print(f"  - Missing in unpruned: {unpruned_dir}")
#             if not pruned_logs:
#                 print(f"  - Missing in pruned: {pruned_dir}")
#             if not pruned_kd_logs:
#                 print(f"  - Missing in pruned_kd: {pruned_kd_dir}")

#     if not logs_unpruned:
#         print("Error: No matching log files found across all three paths!")
#         return None

#     print(f"Found {len(logs_unpruned)} matching prompt directories with log files")

#     unpruned_text = load_outputs_from_files(logs_unpruned)
#     pruned_text = load_outputs_from_files(logs_pruned)
#     pruned_kd_text = load_outputs_from_files(logs_pruned_kd)

#     evaluator = TextSimilarityEvaluator(
#         embedding_model='all-mpnet-base-v2',
#         bertscore_model='roberta-large',
#         use_baseline_rescaling=False,  # Avoid negative scores
#         use_idf=True,
#         max_chunk_tokens=512,
#         verbose=True
#     )

#     results = evaluator.compare_multiple_models(
#         references = unpruned_text,
#         model_outputs = {
#             "unpruned": unpruned_text,
#             "pruned": pruned_text,
#             "pruned_kd": pruned_kd_text
#         },
#         compute_sentence_level = True
#     )

#     evaluator.export_results(results, filepath=output_file)

#     evaluator.print_results(results["unpruned"], model_name="unpruned")
#     evaluator.print_results(results["pruned"], model_name="pruned")
#     evaluator.print_results(results["pruned_kd"], model_name="pruned_kd")

#     evaluator.print_comparison(results)


# def compare_log_files_with_samples(
#     log_file_paths: List[str], 
#     output_file: str = None,
#     use_idf: bool = True,
#     compute_sentence_level: bool = True
# ) -> Dict:
#     """
#     Compare generated summaries vs reference summaries from log files.
    
#     Args:
#         log_file_paths: List of paths to log files to compare
#         output_file: Optional path to save results JSON
#         use_idf: Whether to use IDF weighting in BERTScore
#         compute_sentence_level: Whether to compute sentence-level matching
        
#     Returns:
#         Dictionary containing evaluation results for each log file
#     """
#     evaluator = TextSimilarityEvaluator(
#         embedding_model='all-mpnet-base-v2',
#         bertscore_model='roberta-large',
#         use_baseline_rescaling=False,  # Avoid negative scores
#         use_idf=use_idf,
#         max_chunk_tokens=512,
#         verbose=True
#     )
    
#     all_results = {}
    
#     for log_path in log_file_paths:
#         print(f"\n{'='*80}")
#         print(f"Processing: {log_path}")
#         print('='*80)
        
#         try:
#             # Extract all samples from this log file
#             samples = extract_all_samples_from_log(log_path)
            
#             if not samples:
#                 print(f"Warning: No samples found in {log_path}")
#                 continue
            
#             print(f"Found {len(samples)} samples")
            
#             # Separate generated and reference texts
#             generated_texts = [s['generated'] for s in samples]
#             reference_texts = [s['reference'] for s in samples]
#             sample_ids = [s['sample_id'] for s in samples]
            
#             # Evaluate all samples
#             results = evaluator.evaluate_all(
#                 predictions=generated_texts,
#                 references=reference_texts,
#                 compute_sentence_level=compute_sentence_level
#             )
            
#             # Add sample-level details
#             results['samples'] = []
#             for i, sample_id in enumerate(sample_ids):
#                 sample_result = {
#                     'sample_id': sample_id,
#                     'bertscore_precision': results['bertscore']['precision_list'][i],
#                     'bertscore_recall': results['bertscore']['recall_list'][i],
#                     'bertscore_f1': results['bertscore']['f1_list'][i],
#                     'embedding_similarity': results['embedding_similarity']['similarity_list'][i],
#                 }
                
#                 if compute_sentence_level and 'sentence_matching' in results:
#                     sample_result['sentence_coverage'] = results['sentence_matching']['coverage_list'][i]
#                     sample_result['sentence_order_corr'] = results['sentence_matching']['order_correlation_list'][i]
                
#                 results['samples'].append(sample_result)
            
#             # Store results
#             all_results[log_path] = results
            
#             # Print results for this log file
#             evaluator.print_results(results, model_name=log_path)
            
#         except Exception as e:
#             print(f"Error processing {log_path}: {e}")
#             import traceback
#             traceback.print_exc()
#             continue
    
#     # Print comparison across all log files
#     if len(all_results) > 1:
#         evaluator.print_comparison(all_results)
    
#     # Save results if output file specified
#     if output_file:
#         os.makedirs(os.path.dirname(output_file), exist_ok=True)
#         evaluator.export_results(all_results, output_file)
#         print(f"\nResults saved to: {output_file}")
    
#     return all_results

# def standard_datasets_analysis(path_unpruned: str = None, path_pruned: str = None, path_pruned_kd: str = None, output_file: str = None):

#     sample_log_files = [
#         path_unpruned,
#         path_pruned,
#         path_pruned_kd,
#     ]
    
#     results = compare_log_files_with_samples(
#         log_file_paths=sample_log_files,
#         output_file=output_file
#     )

# # Example usage
# def main_standard_datasets_analysis():
#     #Saving gov report results
#     standard_datasets_analysis(
#         path_unpruned="/users/grad/abhishektyagi/wanda/wanda/results/knowledge_drift/benchmark/meta-llama_Llama-3.2-3B-Instruct/gov_report_0.0prune_kdfalse_threshold/output_20260121_144447.log",
#         path_pruned="/users/grad/abhishektyagi/wanda/wanda/results/knowledge_drift/benchmark/meta-llama_Llama-3.2-3B-Instruct/gov_report_50.0prune_kdfalse_threshold/output_20260121_144410.log",
#         path_pruned_kd="/users/grad/abhishektyagi/wanda/wanda/results/knowledge_drift/benchmark/meta-llama_Llama-3.2-3B-Instruct/gov_report_50.0prune_kdtrue_threshold/output_20260121_144336.log",
#         output_file="results/knowledge_drift/benchmark/compiled/gov_report_del/summary_comparison_results.json"
#     )

#     standard_datasets_analysis(
#         path_unpruned="/users/grad/abhishektyagi/wanda/wanda/results/knowledge_drift/benchmark/meta-llama_Llama-3.2-3B-Instruct/multi_news_0.0prune_kdfalse_threshold/output_20260120_205832.log",
#         path_pruned="/users/grad/abhishektyagi/wanda/wanda/results/knowledge_drift/benchmark/meta-llama_Llama-3.2-3B-Instruct/multi_news_50.0prune_kdfalse_threshold/output_20260120_205931.log",
#         path_pruned_kd="/users/grad/abhishektyagi/wanda/wanda/results/knowledge_drift/benchmark/meta-llama_Llama-3.2-3B-Instruct/multi_news_50.0prune_kdtrue_threshold5_1sigma/output_20260121_165511.log",
#         output_file="results/knowledge_drift/benchmark/compiled/multi_news_del/summary_comparison_results.json"
#     )

#     standard_datasets_analysis(
#         path_unpruned="/users/grad/abhishektyagi/wanda/wanda/results/knowledge_drift/benchmark/meta-llama_Llama-3.2-3B-Instruct/cnn_dailymail_0.0prune_kdtrue_threshold/output_20260120_233950.log",
#         path_pruned="/users/grad/abhishektyagi/wanda/wanda/results/knowledge_drift/benchmark/meta-llama_Llama-3.2-3B-Instruct/cnn_dailymail_50.0prune_kdfalse_threshold/output_20260120_234051.log",
#         path_pruned_kd="/users/grad/abhishektyagi/wanda/wanda/results/knowledge_drift/benchmark/meta-llama_Llama-3.2-3B-Instruct/cnn_dailymail_50.0prune_kdtrue_threshold5_1sigma/output_20260121_165551.log",
#         output_file="results/knowledge_drift/benchmark/compiled/cnn_del/summary_comparison_results.json"
#     )

# def main_custom_prompt_analysis():
#     custom_prompt_analysis(
#         output_file="results/knowledge_drift/custom_prompts/compiled_2/summary_comparison_results.json"
#     )

# if __name__ == "__main__":
#     #main_standard_datasets_analysis()
#     main_custom_prompt_analysis()


"""
Text Similarity Evaluation Script - Enhanced Version
Evaluates semantic and lexical drift in pruned language models by comparing
outputs against unpruned baseline using BLEU, ROUGE, BERTScore, and sentence embeddings.

Includes:
- IDF weighting for BERTScore
- Sentence-level matching for coverage and coherence analysis
- Chunking for very long texts
- ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L, ROUGE-Lsum)
"""

import evaluate
from sentence_transformers import SentenceTransformer, util
import torch
from typing import List, Dict, Tuple
import numpy as np
import warnings
from collections import Counter
import math
import nltk
import json
import os

# Download sentence tokenizer if needed
try:
    nltk.data.find('tokenizers/punkt_tab')
except LookupError:
    print("Downloading NLTK punkt_tab tokenizer...")
    nltk.download('punkt_tab')


class TextSimilarityEvaluator:
    """
    Enhanced evaluator with:
    - ROUGE scores (ROUGE-1, ROUGE-2, ROUGE-L, ROUGE-Lsum)
    - BERTScore with optional IDF weighting
    - Sentence-level matching for coverage/coherence
    - Document-level embedding similarity
    - Chunking support for long texts
    """
    
    def __init__(
        self, 
        embedding_model: str = 'all-mpnet-base-v2',
        bertscore_model: str = 'microsoft/deberta-xlarge-mnli',
        use_baseline_rescaling: bool = False,
        use_idf: bool = True,
        max_chunk_tokens: int = 512,
        verbose: bool = True
    ):
        """
        Initialize evaluator with controlled model specifications.
        
        Args:
            embedding_model: Sentence-transformers model name
            bertscore_model: Specific BERT model for BERTScore
            use_baseline_rescaling: Whether to use baseline rescaling in BERTScore
            use_idf: Whether to use IDF weighting in BERTScore
            max_chunk_tokens: Maximum tokens per chunk for long texts
            verbose: Print loading messages
        """
        self.verbose = verbose
        self.bertscore_model = bertscore_model
        self.use_baseline_rescaling = use_baseline_rescaling
        self.use_idf = use_idf
        self.max_chunk_tokens = max_chunk_tokens
        
        if self.verbose:
            print("Loading evaluation metrics...")
            print(f"  - ROUGE (ROUGE-1, ROUGE-2, ROUGE-L, ROUGE-Lsum)")
            print(f"  - BLEU (diagnostic metric for lexical drift)")
            print(f"  - BERTScore (model: {bertscore_model}, IDF: {use_idf})")
            print(f"  - Sentence embeddings (model: {embedding_model})")
            print(f"  - Max chunk size: {max_chunk_tokens} tokens")
        
        self.rouge_metric = evaluate.load("rouge")
        self.bleu_metric = evaluate.load("bleu")
        self.bertscore_metric = evaluate.load("bertscore")
        self.sentence_model = SentenceTransformer(embedding_model)
        
        # For IDF computation
        self.idf_dict = None
        
        if self.verbose:
            print("All models loaded successfully!\n")
    
    def compute_idf(self, texts: List[str]) -> Dict[str, float]:
        """
        Compute IDF weights from a corpus of texts.
        
        Args:
            texts: List of texts to compute IDF from
            
        Returns:
            Dictionary mapping tokens to IDF weights
        """
        from collections import defaultdict
        doc_freq = defaultdict(int)
        total_docs = len(texts)
        
        for text in texts:
            tokens = set(text.lower().split())
            for token in tokens:
                doc_freq[token] += 1
        
        # Compute IDF: log(N / df)
        idf_dict = {}
        for token, freq in doc_freq.items():
            idf_dict[token] = math.log(total_docs / freq)
        
        return idf_dict
    
    def compute_rouge(
        self, 
        predictions: List[str], 
        references: List[str]
    ) -> Dict:
        """
        Compute ROUGE scores between predictions and references.
        
        Args:
            predictions: List of generated texts
            references: List of reference texts
            
        Returns:
            Dictionary with ROUGE-1, ROUGE-2, ROUGE-L, and ROUGE-Lsum scores
        """
        scores = self.rouge_metric.compute(
            predictions=predictions,
            references=references,
            use_stemmer=True
        )
        
        return {
            'rouge1': scores['rouge1'],
            'rouge2': scores['rouge2'],
            'rougeL': scores['rougeL'],
            'rougeLsum': scores['rougeLsum']
        }
    
    def compute_bleu(
        self, 
        predictions: List[str], 
        references: List[str]
    ) -> Dict:
        """
        Compute BLEU score between predictions and references.
        
        NOTE: BLEU is designed for corpus-level MT evaluation and is used here
        only as a diagnostic signal for catastrophic lexical drift.
        
        Args:
            predictions: List of generated texts
            references: List of reference texts
            
        Returns:
            Dictionary with BLEU scores and statistics
        """
        references_formatted = [[ref] for ref in references]
        result = self.bleu_metric.compute(
            predictions=predictions,
            references=references_formatted
        )
        
        if result['bleu'] == 0.0:
            warnings.warn(
                "BLEU score is 0.0, which may indicate severe lexical drift "
                "or very short/paraphrased outputs."
            )
        
        return result
    
    def chunk_text(self, text: str, max_tokens: int = 512) -> List[str]:
        """
        Split text into chunks if it's very long.
        
        Args:
            text: Input text
            max_tokens: Maximum tokens per chunk (approximate)
            
        Returns:
            List of text chunks
        """
        # Rough approximation: 1 token ≈ 4 characters
        max_chars = max_tokens * 4
        
        if len(text) <= max_chars:
            return [text]
        
        # Split into sentences first
        sentences = nltk.sent_tokenize(text)
        
        chunks = []
        current_chunk = []
        current_length = 0
        
        for sentence in sentences:
            sentence_length = len(sentence)
            
            if current_length + sentence_length > max_chars and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length
            else:
                current_chunk.append(sentence)
                current_length += sentence_length
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks
    
    def compute_bertscore(
        self, 
        predictions: List[str], 
        references: List[str],
        lang: str = "en"
    ) -> Dict:
        """
        Compute BERTScore with optional IDF weighting and chunking.
        
        Args:
            predictions: List of generated texts
            references: List of reference texts
            lang: Language code (default: "en")
            
        Returns:
            Dictionary with precision, recall, F1 scores (mean and std)
        """
        all_precision = []
        all_recall = []
        all_f1 = []
        
        # Compute IDF if enabled and not already computed
        if self.use_idf and self.idf_dict is None:
            if self.verbose:
                print("  Computing IDF weights from corpus...")
            all_texts = predictions + references
            self.idf_dict = self.compute_idf(all_texts)
        
        # Process each prediction-reference pair
        for pred, ref in zip(predictions, references):
            # Check if texts need chunking
            pred_chunks = self.chunk_text(pred, self.max_chunk_tokens)
            ref_chunks = self.chunk_text(ref, self.max_chunk_tokens)
            
            if len(pred_chunks) > 1 or len(ref_chunks) > 1:
                if self.verbose and len(all_precision) == 0:
                    print(f"  Text requires chunking: {len(pred_chunks)} pred chunks, {len(ref_chunks)} ref chunks")
                
                # Compute BERTScore for each chunk pair and average
                chunk_scores = []
                for p_chunk in pred_chunks:
                    for r_chunk in ref_chunks:
                        result = self.bertscore_metric.compute(
                            predictions=[p_chunk],
                            references=[r_chunk],
                            lang=lang,
                            model_type=self.bertscore_model,
                            idf=self.use_idf,
                            rescale_with_baseline=self.use_baseline_rescaling
                        )
                        chunk_scores.append({
                            'precision': result['precision'][0],
                            'recall': result['recall'][0],
                            'f1': result['f1'][0]
                        })
                
                # Average across chunks
                avg_precision = np.mean([s['precision'] for s in chunk_scores])
                avg_recall = np.mean([s['recall'] for s in chunk_scores])
                avg_f1 = np.mean([s['f1'] for s in chunk_scores])
            else:
                # No chunking needed
                result = self.bertscore_metric.compute(
                    predictions=[pred],
                    references=[ref],
                    lang=lang,
                    model_type=self.bertscore_model,
                    idf=self.use_idf,
                    rescale_with_baseline=self.use_baseline_rescaling
                )
                avg_precision = result['precision'][0]
                avg_recall = result['recall'][0]
                avg_f1 = result['f1'][0]
            
            all_precision.append(avg_precision)
            all_recall.append(avg_recall)
            all_f1.append(avg_f1)
        
        # Calculate statistics
        precision_scores = np.array(all_precision)
        recall_scores = np.array(all_recall)
        f1_scores = np.array(all_f1)
        
        return {
            'precision_mean': np.mean(precision_scores),
            'precision_std': np.std(precision_scores),
            'recall_mean': np.mean(recall_scores),
            'recall_std': np.std(recall_scores),
            'f1_mean': np.mean(f1_scores),
            'f1_std': np.std(f1_scores),
            'precision_list': all_precision,
            'recall_list': all_recall,
            'f1_list': all_f1,
            'using_idf': self.use_idf
        }
    
    def compute_sentence_matching(
        self,
        predictions: List[str],
        references: List[str],
        similarity_threshold: float = 0.5
    ) -> Dict:
        """
        Compute sentence-level matching between predictions and references.
        
        Reveals:
        - Coverage: What % of reference sentences are well-represented?
        - Coherence: Are sentences in similar order?
        - Per-sentence quality: Distribution of matches
        
        Args:
            predictions: List of generated texts
            references: List of reference texts
            similarity_threshold: Threshold for considering a sentence "matched"
            
        Returns:
            Dictionary with sentence-level metrics
        """
        all_coverages = []
        all_avg_similarities = []
        all_order_correlations = []
        all_best_matches = []
        
        for pred, ref in zip(predictions, references):
            # Split into sentences
            pred_sentences = nltk.sent_tokenize(pred)
            ref_sentences = nltk.sent_tokenize(ref)
            
            if len(pred_sentences) == 0 or len(ref_sentences) == 0:
                continue
            
            # Encode all sentences
            pred_embeddings = self.sentence_model.encode(
                pred_sentences, 
                convert_to_tensor=True,
                show_progress_bar=False
            )
            ref_embeddings = self.sentence_model.encode(
                ref_sentences,
                convert_to_tensor=True,
                show_progress_bar=False
            )
            
            # Compute similarity matrix (ref_sentences x pred_sentences)
            similarity_matrix = util.cos_sim(ref_embeddings, pred_embeddings).cpu().numpy()
            
            # For each reference sentence, find best match in prediction
            best_matches = similarity_matrix.max(axis=1)
            best_match_indices = similarity_matrix.argmax(axis=1)
            
            # Coverage: % of reference sentences with similarity > threshold
            coverage = (best_matches >= similarity_threshold).mean()
            
            # Average similarity of best matches
            avg_similarity = best_matches.mean()
            
            # Order correlation: Do matched sentences appear in similar order?
            if len(best_matches) > 1:
                ref_indices = np.arange(len(ref_sentences))
                try:
                    from scipy.stats import spearmanr
                    order_corr = spearmanr(ref_indices, best_match_indices)[0]
                    if np.isnan(order_corr):
                        order_corr = 0.0
                except:
                    # Fallback if scipy not available
                    order_corr = 0.0
            else:
                order_corr = 1.0
            
            all_coverages.append(coverage)
            all_avg_similarities.append(avg_similarity)
            all_order_correlations.append(order_corr)
            all_best_matches.extend(best_matches.tolist())
        
        return {
            'coverage_mean': np.mean(all_coverages),
            'coverage_std': np.std(all_coverages),
            'avg_similarity_mean': np.mean(all_avg_similarities),
            'avg_similarity_std': np.std(all_avg_similarities),
            'order_correlation_mean': np.mean(all_order_correlations),
            'order_correlation_std': np.std(all_order_correlations),
            'threshold_used': similarity_threshold,
            'all_sentence_similarities': all_best_matches,
            'coverage_list': all_coverages,
            'order_correlation_list': all_order_correlations
        }
    
    def compute_embedding_similarity(
        self, 
        predictions: List[str], 
        references: List[str]
    ) -> Dict:
        """
        Compute document-level cosine similarity using sentence embeddings.
        
        Args:
            predictions: List of generated texts
            references: List of reference texts
            
        Returns:
            Dictionary with similarity statistics
        """
        pred_embeddings = self.sentence_model.encode(
            predictions, 
            convert_to_tensor=True,
            show_progress_bar=False
        )
        ref_embeddings = self.sentence_model.encode(
            references, 
            convert_to_tensor=True,
            show_progress_bar=False
        )
        
        similarity_matrix = util.cos_sim(pred_embeddings, ref_embeddings)
        similarities = similarity_matrix.diagonal().cpu().numpy()
        
        return {
            'mean': float(np.mean(similarities)),
            'std': float(np.std(similarities)),
            'min': float(np.min(similarities)),
            'max': float(np.max(similarities)),
            'median': float(np.median(similarities)),
            'similarity_list': similarities.tolist()
        }
    
    def evaluate_all(
        self, 
        predictions: List[str], 
        references: List[str],
        lang: str = "en",
        compute_sentence_level: bool = True
    ) -> Dict:
        """
        Compute all metrics at once.
        
        Args:
            predictions: List of generated texts (pruned model)
            references: List of reference texts (unpruned model)
            lang: Language code for BERTScore
            compute_sentence_level: Whether to compute sentence-level matching
            
        Returns:
            Dictionary containing all metric results
        """
        if len(predictions) != len(references):
            raise ValueError(
                f"Number of predictions ({len(predictions)}) must match "
                f"number of references ({len(references)})"
            )
        
        if self.verbose:
            print(f"Evaluating {len(predictions)} text pairs...\n")
        
        if self.verbose:
            print("Computing ROUGE scores...")
        rouge_result = self.compute_rouge(predictions, references)
        
        if self.verbose:
            print("Computing BLEU score (lexical drift diagnostic)...")
        bleu_result = self.compute_bleu(predictions, references)
        
        if self.verbose:
            print(f"Computing BERTScore (IDF: {self.use_idf})...")
        bertscore_result = self.compute_bertscore(predictions, references, lang)
        
        if self.verbose:
            print("Computing document-level embedding similarity...")
        embedding_result = self.compute_embedding_similarity(predictions, references)
        
        results = {
            'rouge': rouge_result,
            'bleu': bleu_result,
            'bertscore': bertscore_result,
            'embedding_similarity': embedding_result
        }
        
        if compute_sentence_level:
            if self.verbose:
                print("Computing sentence-level matching...")
            sentence_result = self.compute_sentence_matching(predictions, references)
            results['sentence_matching'] = sentence_result
        
        return results
    
    def compare_multiple_models(
        self,
        references: List[str],
        model_outputs: Dict[str, List[str]],
        lang: str = "en",
        compute_sentence_level: bool = True
    ) -> Dict[str, Dict]:
        """
        Compare multiple model outputs against the same references.
        
        Args:
            references: List of reference texts (unpruned baseline)
            model_outputs: Dictionary mapping model names to their outputs
            lang: Language code for BERTScore
            compute_sentence_level: Whether to compute sentence-level metrics
            
        Returns:
            Dictionary mapping model names to evaluation results
        """
        results = {}
        
        for model_name, predictions in model_outputs.items():
            if len(predictions) != len(references):
                raise ValueError(
                    f"Model '{model_name}' has {len(predictions)} outputs but "
                    f"there are {len(references)} references"
                )
            
            if self.verbose:
                print(f"\n{'='*60}")
                print(f"Evaluating: {model_name}")
                print(f"{'='*60}")
            
            results[model_name] = self.evaluate_all(
                predictions, 
                references, 
                lang,
                compute_sentence_level=compute_sentence_level
            )
        
        return results
    
    def print_results(self, results: Dict, model_name: str = None):
        """Print evaluation results with statistics."""
        header = f"EVALUATION RESULTS - {model_name}" if model_name else "EVALUATION RESULTS"
        print("\n" + "="*80)
        print(header)
        print("="*80)
        
        print("\n[ROUGE Scores]")
        print(f"  ROUGE-1:    {results['rouge']['rouge1']:.4f}")
        print(f"  ROUGE-2:    {results['rouge']['rouge2']:.4f}")
        print(f"  ROUGE-L:    {results['rouge']['rougeL']:.4f}")
        print(f"  ROUGE-Lsum: {results['rouge']['rougeLsum']:.4f}")
        
        print("\n[BLEU Score - Lexical Drift Diagnostic]")
        print(f"  Score: {results['bleu']['bleu']:.4f}")
        
        print("\n[BERTScore - Semantic Similarity]")
        print(f"  Using IDF: {results['bertscore'].get('using_idf', False)}")
        print(f"  Precision: {results['bertscore']['precision_mean']:.4f} "
              f"(±{results['bertscore']['precision_std']:.4f})")
        print(f"  Recall:    {results['bertscore']['recall_mean']:.4f} "
              f"(±{results['bertscore']['recall_std']:.4f})")
        print(f"  F1:        {results['bertscore']['f1_mean']:.4f} "
              f"(±{results['bertscore']['f1_std']:.4f})")
        
        print("\n[Document-Level Embedding Similarity]")
        print(f"  Mean:      {results['embedding_similarity']['mean']:.4f} "
              f"(±{results['embedding_similarity']['std']:.4f})")
        print(f"  Median:    {results['embedding_similarity']['median']:.4f}")
        print(f"  Range:     [{results['embedding_similarity']['min']:.4f}, "
              f"{results['embedding_similarity']['max']:.4f}]")
        
        if 'sentence_matching' in results:
            sm = results['sentence_matching']
            print("\n[Sentence-Level Matching - Coverage & Coherence]")
            print(f"  Coverage:          {sm['coverage_mean']:.4f} "
                  f"(±{sm['coverage_std']:.4f})")
            print(f"  Avg Similarity:    {sm['avg_similarity_mean']:.4f} "
                  f"(±{sm['avg_similarity_std']:.4f})")
            print(f"  Order Correlation: {sm['order_correlation_mean']:.4f} "
                  f"(±{sm['order_correlation_std']:.4f})")
            print(f"  Threshold:         {sm['threshold_used']:.2f}")
            print(f"  → Coverage: % of reference sentences with good matches")
            print(f"  → Order Correlation: 1.0=perfect order, 0.0=random, -1.0=reversed")
        
        print("="*80 + "\n")
    
    def print_comparison(self, comparison_results: Dict[str, Dict]):
        """Print comparison table across models with statistics."""
        print("\n" + "="*125)
        print("COMPARISON SUMMARY")
        print("="*125)
        
        model_names = list(comparison_results.keys())
        display_names = [name[-55:] if len(name) > 55 else name for name in model_names]
        
        has_sentence = any('sentence_matching' in r for r in comparison_results.values())
        
        if has_sentence:
            print(f"\n{'Model':<57} {'ROUGE-L':<10} {'BLEU':<10} {'BERTScore F1':<18} {'Embed Sim':<18} {'Coverage':<12}")
            print("-" * 125)
        else:
            print(f"\n{'Model':<57} {'ROUGE-L':<10} {'BLEU':<10} {'BERTScore F1':<18} {'Embed Sim':<18}")
            print("-" * 113)
        
        for model_name, display_name in zip(model_names, display_names):
            res = comparison_results[model_name]
            rougeL = res['rouge']['rougeL']
            bleu = res['bleu']['bleu']
            bert_f1_mean = res['bertscore']['f1_mean']
            bert_f1_std = res['bertscore']['f1_std']
            emb_mean = res['embedding_similarity']['mean']
            emb_std = res['embedding_similarity']['std']
            
            line = f"{display_name:<57} {rougeL:<10.4f} {bleu:<10.4f} {bert_f1_mean:.4f}(±{bert_f1_std:.3f}){'':<6} {emb_mean:.4f}(±{emb_std:.3f}){'':<6}"
            
            if 'sentence_matching' in res:
                cov_mean = res['sentence_matching']['coverage_mean']
                cov_std = res['sentence_matching']['coverage_std']
                line += f" {cov_mean:.4f}(±{cov_std:.3f})"
            
            print(line)
        
        print("="*125 + "\n")
    
    def export_results(self, results: Dict, filepath: str):
        """Export results to JSON for further analysis."""
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        def convert_to_serializable(obj):
            """Convert numpy types to Python native types for JSON serialization."""
            if isinstance(obj, dict):
                return {k: convert_to_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_to_serializable(item) for item in obj]
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            elif isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            elif isinstance(obj, np.bool_):
                return bool(obj)
            else:
                return obj
        
        serializable_results = convert_to_serializable(results)
        
        with open(filepath, 'w') as f:
            json.dump(serializable_results, f, indent=2)
        print(f"Results exported to {filepath}")


# ============================================================================
# Helper Functions for Loading Data
# ============================================================================


def extract_generated_text_from_log(log_filepath: str) -> str:
    """
    Extract the 'Newly generated text' section from a log file.
    
    Args:
        log_filepath: Path to the log file
        
    Returns:
        The generated text content
    """
    with open(log_filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    start_marker = "Newly generated text:"
    end_marker = "=" * 80
    
    start_idx = content.find(start_marker)
    if start_idx == -1:
        raise ValueError(f"Could not find '{start_marker}' in log file: {log_filepath}")
    
    start_idx += len(start_marker)
    end_idx = content.find(end_marker, start_idx)
    
    if end_idx == -1:
        extracted_text = content[start_idx:].strip()
    else:
        extracted_text = content[start_idx:end_idx].strip()
    
    return extracted_text


def extract_all_samples_from_log(log_filepath: str) -> List[Dict[str, str]]:
    """
    Extract all samples from a log file where each sample has:
    - SAMPLE number
    - Input Document
    - Generated Summary
    - Reference Summary
    
    Args:
        log_filepath: Path to the log file
        
    Returns:
        List of dictionaries, each containing:
        - 'sample_id': Sample number
        - 'generated': Generated summary text
        - 'reference': Reference summary text
    """
    with open(log_filepath, 'r', encoding='utf-8') as f:
        content = f.read()
    
    samples = []
    
    # Split by sample markers
    sample_sections = content.split('---[[ SAMPLE ')
    
    for section in sample_sections[1:]:  # Skip first empty section
        try:
            # Extract sample ID
            sample_id_end = section.find(']]---')
            sample_id = section[:sample_id_end].strip()
            
            # Extract Generated Summary
            # Try to find any "Generated Summary (XXX tokens):" format
            gen_marker_base = 'Generated Summary ('
            gen_start = section.find(gen_marker_base)
            
            if gen_start != -1:
                # Find the end of the marker (the colon after the token count)
                colon_pos = section.find(':', gen_start)
                if colon_pos != -1:
                    gen_start = colon_pos + 1
                else:
                    gen_start = -1
            
            # Fall back to simple "Generated Summary:" if parenthetical format not found
            if gen_start == -1:
                gen_marker = 'Generated Summary:'
                gen_start = section.find(gen_marker)
                if gen_start != -1:
                    gen_start += len(gen_marker)
            
            if gen_start == -1:
                print(f"Warning: Could not find generated summary in sample {sample_id}")
                continue
            
            # Find Reference Summary marker
            ref_marker = 'Reference Summary:'
            ref_start = section.find(ref_marker, gen_start)
            
            if ref_start == -1:
                print(f"Warning: Could not find reference summary in sample {sample_id}")
                continue
            
            # Extract generated text
            generated_text = section[gen_start:ref_start].strip()
            
            # Extract reference text
            ref_start += len(ref_marker)
            next_section = section.find('---END OF TEXT---', ref_start)
            
            if next_section != -1:
                reference_text = section[ref_start:next_section].strip()
            else:
                reference_text = section[ref_start:].strip()
            
            samples.append({
                'sample_id': sample_id,
                'generated': generated_text,
                'reference': reference_text
            })
            
        except Exception as e:
            print(f"Error processing sample: {e}")
            continue
    
    return samples

def load_outputs_from_files(file_paths: List[str]) -> List[str]:
    """
    Load model outputs from multiple files.
    
    Args:
        file_paths: List of file paths to load
        
    Returns:
        List of text outputs
    """
    outputs = []
    for path in file_paths:
        text = extract_generated_text_from_log(path)
        outputs.append(text)
    return outputs


# ============================================================================
# Main Comparison Function
# ============================================================================

def custom_prompt_analysis(path_unpruned: str = None, path_pruned: str = None, path_pruned_kd: str = None, output_file: str = None):
    import glob

    # Use provided paths or defaults
    if path_unpruned is None:
        path_unpruned = "/users/grad/abhishektyagi/wanda/wanda/results/knowledge_drift/custom_prompts_TOT/dense_v2/meta-llama_Llama-3.2-3B-Instruct"
    if path_pruned is None:
        path_pruned = "/users/grad/abhishektyagi/wanda/wanda/results/knowledge_drift/custom_prompts_TOT/sparse_v2/meta-llama_Llama-3.2-3B-Instruct"
    if path_pruned_kd is None:
        path_pruned_kd = "/users/grad/abhishektyagi/wanda/wanda/results/knowledge_drift/custom_prompts_TOT/sparse_kd/meta-llama_Llama-3.2-3B-Instruct"

    # Get all directories and build mappings by prompt ID
    def get_prompt_id(dirname):
        """Extract prompt ID (e.g., 'prompt0' from 'prompt0_something')"""
        if dirname.startswith('prompt'):
            # Find the first underscore after 'prompt'
            underscore_pos = dirname.find('_')
            if underscore_pos != -1:
                return dirname[:underscore_pos]
            else:
                return dirname  # No underscore, use full name
        return None
    
    def build_dir_mapping(base_path):
        """Build a mapping from prompt ID to full directory path"""
        mapping = {}
        if not os.path.exists(base_path):
            return mapping
        for d in os.listdir(base_path):
            full_path = os.path.join(base_path, d)
            if os.path.isdir(full_path):
                prompt_id = get_prompt_id(d)
                if prompt_id:
                    mapping[prompt_id] = full_path
        return mapping
    
    unpruned_dirs = build_dir_mapping(path_unpruned)
    pruned_dirs = build_dir_mapping(path_pruned)
    pruned_kd_dirs = build_dir_mapping(path_pruned_kd)
    
    # Find common prompt IDs across all three paths
    common_prompt_ids = set(unpruned_dirs.keys()) & set(pruned_dirs.keys()) & set(pruned_kd_dirs.keys())
    
    if not common_prompt_ids:
        print("Error: No common prompt IDs found across all three paths!")
        print(f"Unpruned has: {sorted(unpruned_dirs.keys())}")
        print(f"Pruned has: {sorted(pruned_dirs.keys())}")
        print(f"Pruned_kd has: {sorted(pruned_kd_dirs.keys())}")
        return None
    
    logs_unpruned = []
    logs_pruned = []
    logs_pruned_kd = []
    
    for prompt_id in sorted(common_prompt_ids):
        unpruned_dir = unpruned_dirs[prompt_id]
        pruned_dir = pruned_dirs[prompt_id]
        pruned_kd_dir = pruned_kd_dirs[prompt_id]
        
        # Find output*.log files in each directory
        unpruned_logs = glob.glob(os.path.join(unpruned_dir, "output*.log"))
        pruned_logs = glob.glob(os.path.join(pruned_dir, "output*.log"))
        pruned_kd_logs = glob.glob(os.path.join(pruned_kd_dir, "output*.log"))
        
        # Check if all three directories have log files
        if unpruned_logs and pruned_logs and pruned_kd_logs:
            logs_unpruned.append(unpruned_logs[0])  # Take first log file
            logs_pruned.append(pruned_logs[0])
            logs_pruned_kd.append(pruned_kd_logs[0])
        else:
            print(f"Warning: Missing log files for prompt ID '{prompt_id}'")
            if not unpruned_logs:
                print(f"  - Missing in unpruned: {unpruned_dir}")
            if not pruned_logs:
                print(f"  - Missing in pruned: {pruned_dir}")
            if not pruned_kd_logs:
                print(f"  - Missing in pruned_kd: {pruned_kd_dir}")

    if not logs_unpruned:
        print("Error: No matching log files found across all three paths!")
        return None

    print(f"Found {len(logs_unpruned)} matching prompt directories with log files")

    unpruned_text = load_outputs_from_files(logs_unpruned)
    pruned_text = load_outputs_from_files(logs_pruned)
    pruned_kd_text = load_outputs_from_files(logs_pruned_kd)

    evaluator = TextSimilarityEvaluator(
        embedding_model='all-mpnet-base-v2',
        bertscore_model='roberta-large',
        use_baseline_rescaling=False,  # Avoid negative scores
        use_idf=True,
        max_chunk_tokens=512,
        verbose=True
    )

    results = evaluator.compare_multiple_models(
        references = unpruned_text,
        model_outputs = {
            "unpruned": unpruned_text,
            "pruned": pruned_text,
            "pruned_kd": pruned_kd_text
        },
        compute_sentence_level = True
    )

    evaluator.export_results(results, filepath=output_file)

    evaluator.print_results(results["unpruned"], model_name="unpruned")
    evaluator.print_results(results["pruned"], model_name="pruned")
    evaluator.print_results(results["pruned_kd"], model_name="pruned_kd")

    evaluator.print_comparison(results)


def compare_log_files_with_samples(
    log_file_paths: List[str], 
    output_file: str = None,
    use_idf: bool = True,
    compute_sentence_level: bool = True
) -> Dict:
    """
    Compare generated summaries vs reference summaries from log files.
    
    Args:
        log_file_paths: List of paths to log files to compare
        output_file: Optional path to save results JSON
        use_idf: Whether to use IDF weighting in BERTScore
        compute_sentence_level: Whether to compute sentence-level matching
        
    Returns:
        Dictionary containing evaluation results for each log file
    """
    evaluator = TextSimilarityEvaluator(
        embedding_model='all-mpnet-base-v2',
        bertscore_model='roberta-large',
        use_baseline_rescaling=False,  # Avoid negative scores
        use_idf=use_idf,
        max_chunk_tokens=512,
        verbose=True
    )
    
    all_results = {}
    
    for log_path in log_file_paths:
        print(f"\n{'='*80}")
        print(f"Processing: {log_path}")
        print('='*80)
        
        try:
            # Extract all samples from this log file
            samples = extract_all_samples_from_log(log_path)
            
            if not samples:
                print(f"Warning: No samples found in {log_path}")
                continue
            
            print(f"Found {len(samples)} samples")
            
            # Separate generated and reference texts
            generated_texts = [s['generated'] for s in samples]
            reference_texts = [s['reference'] for s in samples]
            sample_ids = [s['sample_id'] for s in samples]
            
            # Evaluate all samples
            results = evaluator.evaluate_all(
                predictions=generated_texts,
                references=reference_texts,
                compute_sentence_level=compute_sentence_level
            )
            
            # Add sample-level details
            results['samples'] = []
            for i, sample_id in enumerate(sample_ids):
                sample_result = {
                    'sample_id': sample_id,
                    'bertscore_precision': results['bertscore']['precision_list'][i],
                    'bertscore_recall': results['bertscore']['recall_list'][i],
                    'bertscore_f1': results['bertscore']['f1_list'][i],
                    'embedding_similarity': results['embedding_similarity']['similarity_list'][i],
                }
                
                if compute_sentence_level and 'sentence_matching' in results:
                    sample_result['sentence_coverage'] = results['sentence_matching']['coverage_list'][i]
                    sample_result['sentence_order_corr'] = results['sentence_matching']['order_correlation_list'][i]
                
                results['samples'].append(sample_result)
            
            # Store results
            all_results[log_path] = results
            
            # Print results for this log file
            evaluator.print_results(results, model_name=log_path)
            
        except Exception as e:
            print(f"Error processing {log_path}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Print comparison across all log files
    if len(all_results) > 1:
        evaluator.print_comparison(all_results)
    
    # Save results if output file specified
    if output_file:
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        evaluator.export_results(all_results, output_file)
        print(f"\nResults saved to: {output_file}")
    
    return all_results

def standard_datasets_analysis(path_unpruned: str = None, path_pruned: str = None, path_pruned_kd: str = None, output_file: str = None):

    sample_log_files = [
        path_unpruned,
        path_pruned,
        path_pruned_kd,
    ]
    
    results = compare_log_files_with_samples(
        log_file_paths=sample_log_files,
        output_file=output_file
    )

# Example usage
def main_standard_datasets_analysis():
    #Saving gov report results
    standard_datasets_analysis(
        path_unpruned="/users/grad/abhishektyagi/wanda/wanda/results/knowledge_drift/benchmark/meta-llama_Llama-3.2-3B-Instruct/gov_report_0.0prune_kdfalse_threshold/output_20260121_144447.log",
        path_pruned="/users/grad/abhishektyagi/wanda/wanda/results/knowledge_drift/benchmark/meta-llama_Llama-3.2-3B-Instruct/gov_report_50.0prune_kdfalse_threshold/output_20260121_144410.log",
        path_pruned_kd="/users/grad/abhishektyagi/wanda/wanda/results/knowledge_drift/benchmark/meta-llama_Llama-3.2-3B-Instruct/gov_report_50.0prune_kdtrue_threshold/output_20260121_144336.log",
        output_file="results/knowledge_drift/benchmark/compiled/gov_report_del/summary_comparison_results.json"
    )

    standard_datasets_analysis(
        path_unpruned="/users/grad/abhishektyagi/wanda/wanda/results/knowledge_drift/benchmark/meta-llama_Llama-3.2-3B-Instruct/multi_news_0.0prune_kdfalse_threshold/output_20260120_205832.log",
        path_pruned="/users/grad/abhishektyagi/wanda/wanda/results/knowledge_drift/benchmark/meta-llama_Llama-3.2-3B-Instruct/multi_news_50.0prune_kdfalse_threshold/output_20260120_205931.log",
        path_pruned_kd="/users/grad/abhishektyagi/wanda/wanda/results/knowledge_drift/benchmark/meta-llama_Llama-3.2-3B-Instruct/multi_news_50.0prune_kdtrue_threshold5_1sigma/output_20260121_165511.log",
        output_file="results/knowledge_drift/benchmark/compiled/multi_news_del/summary_comparison_results.json"
    )

    standard_datasets_analysis(
        path_unpruned="/users/grad/abhishektyagi/wanda/wanda/results/knowledge_drift/benchmark/meta-llama_Llama-3.2-3B-Instruct/cnn_dailymail_0.0prune_kdtrue_threshold/output_20260120_233950.log",
        path_pruned="/users/grad/abhishektyagi/wanda/wanda/results/knowledge_drift/benchmark/meta-llama_Llama-3.2-3B-Instruct/cnn_dailymail_50.0prune_kdfalse_threshold/output_20260120_234051.log",
        path_pruned_kd="/users/grad/abhishektyagi/wanda/wanda/results/knowledge_drift/benchmark/meta-llama_Llama-3.2-3B-Instruct/cnn_dailymail_50.0prune_kdtrue_threshold5_1sigma/output_20260121_165551.log",
        output_file="results/knowledge_drift/benchmark/compiled/cnn_del/summary_comparison_results.json"
    )

def main_custom_prompt_analysis():
    custom_prompt_analysis(
        output_file="results/knowledge_drift/custom_prompts/compiled_2/summary_comparison_results.json"
    )

if __name__ == "__main__":
    #main_standard_datasets_analysis()
    main_custom_prompt_analysis()