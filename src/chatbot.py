#!/usr/bin/env python3
"""
Simple Profiled LLM Chatbot
Uses existing LLMMetricsCollector with minimal additional code
"""

import torch
import os
from transformers import AutoTokenizer
# Import your existing functions/classes
from perf_analysis import LLMMetricsCollector, get_llm

def simple_chatbot(model_name="gpt2", max_new_tokens=50, temperature=0.7):
    # 1) Load model and tokenizer once
    print(f"Loading {model_name}...")
    collector = LLMMetricsCollector(monitoring_interval=0.1)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    model = get_llm(model_name)
    model.eval()
    print("Model loaded! ✅\n")
    
    # 2) Chat loop
    chat_history = ""
    turn = 0
    
    print("Welcome to your profiled chatbot! Type 'exit' to quit.\n")
    
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        
        turn += 1
        
        # 3) Build prompt with conversation history
        if chat_history:
            prompt = f"{chat_history}User: {user_input}\nBot: "
        else:
            prompt = f"User: {user_input}\nBot: "
        
        # 4) Run your benchmarking-based generation
        print("Thinking...", end="", flush=True)
        metrics = collector.benchmark(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            model_name=model_name,
            max_new_tokens=max_new_tokens,
            temperature=temperature
        )
        
        # 5) Use simple generation to avoid the CUDA indexing bug
        with torch.no_grad():
            inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=tokenizer.eos_token_id
            )
            full_response = tokenizer.decode(outputs[0], skip_special_tokens=True)
            bot_reply = full_response.replace(prompt, "").strip()
        
        print(f"\rBot: {bot_reply}")
        print(f"⚡ {metrics.total_time_seconds:.2f}s | {metrics.tokens_per_second:.1f} tok/s | GPU: {metrics.peak_gpu_memory_mb:.0f}MB\n")
        
        # 6) Update conversation history
        chat_history = f"{prompt}{bot_reply}\n"
        
        # 7) Optionally save metrics
        os.makedirs("logs", exist_ok=True)
        collector.save_metrics(metrics, f"logs/turn_{turn:03d}.json")

if __name__ == "__main__":
    # Simple usage
    simple_chatbot(
        model_name="meta-llama/Llama-2-7b-hf",  # Trained for conversation
        max_new_tokens=30,
        temperature=0.0
    )