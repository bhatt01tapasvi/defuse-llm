## Installation

### Prerequisites
- Python 2.2.0 or higher
- CUDA-capable GPU (recommended for model inference)
- 16GB+ RAM recommended

### Setup

1. **Clone the repository**
```bash
git clone https://github.com/ATygah/defuse-llm.git
cd defuse-llm
```

2. **Create a virtual environment** (recommended)
```bash
conda create -n prune_llm python=3.9
conda activate prune_llm
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up HuggingFace token** (for accessing gated models like LLaMA - OPTIONAL)
```bash
# Option 1: Environment variable
export HF_TOKEN="your_token_here"

# Option 2: Use huggingface-cli
huggingface-cli login
```

5. **Verify installation**
```bash
python -c "import torch; print(f'PyTorch version: {torch.__version__}')"
python -c "import transformers; print(f'Transformers version: {transformers.__version__}')"
```