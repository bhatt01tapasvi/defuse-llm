# Project Context: PyTorch to Triton FFN Conversion

## Objective
Convert a custom PyTorch-based Feed-Forward Network (FFN) with a dynamic, structured pruning framework into highly optimized OpenAI Triton kernels. The final output must seamlessly integrate into a Hugging Face `transformers` model architecture.

## Architecture Stack
* **Framework:** PyTorch
* **Target GPU Backend:** OpenAI Triton (`@triton.jit`)
* **Integration Target:** Hugging Face `nn.Module` (specifically replacing gated MLP/FFN blocks like LlamaMLP or MistralMLP).
* **Pruning Strategy:** Dynamic, structural whole-neuron pruning using exact Top-K selection.

## Triton Coding Guidelines & Best Practices
When writing or refactoring Triton code in this repository, strictly adhere to the following:

1. **Memory Coalescing:** Always structure pointer arithmetic to ensure contiguous memory access. FFNs are memory-bound; prioritize coalesced reads/writes over compute optimizations.
2. **Block Sizes as Meta-Parameters:** Never hardcode block sizes. Use `tl.constexpr` for `BLOCK_SIZE_M`, `BLOCK_SIZE_N`, and `BLOCK_SIZE_K`. These must be tunable via `@triton.autotune`.
3. **SRAM Limits:** Be mindful of shared memory limits. If computing large tiles, ensure the block sizes don't exceed the GPU's SRAM capacity per SM.
4. **Masking:** Always use boundary masks when loading and storing data (`tl.load(..., mask=...)`) to prevent out-of-bounds memory faults on sequence lengths or hidden dimensions that are not perfect multiples of the block size.
5. **Pruning Logic:** The kernel must handle the sparsity indices natively. Never load pruned zero-weights into SRAM. 

---

## Step-by-Step Conversion Roadmap

Please follow these steps sequentially when asked to convert a new piece of Python code:

### Step 1: Isolate & Profile the PyTorch Logic
* **Task:** Extract the exact mathematical operations of the Python FFN layer.
* **Requirement:** Ensure there is a pure PyTorch baseline implementation to compare against for both correctness (using `torch.testing.assert_close`) and performance.

### Step 2: Define the Memory Layout and Grid (Structured Neuron Pruning)
* **Task:** Map the input tensors `[tokens, hidden_dim]` and the surviving neuron weight matrices to Triton's block grids.
* **Requirement:** Explicitly define how the kernel handles the dynamically pruned dimensions. Implement a 1D index pointer (e.g., `surviving_neuron_indices`) so the kernel knows exactly which columns/rows to load into SRAM, skipping the pruned neurons entirely. Always use 2D block grids that align with the surviving dimensions to maximize Tensor Core utilization.

### Step 3: Draft the Triton Kernels (Two-Stage Gated Pipeline)
* **Task:** Write two separate `@triton.jit` kernels to handle the compute-select-compute pipeline for a gated FFN (e.g., SwiGLU). Avoid global synchronization blocks within a single kernel.
* **Requirement:**
    * **Kernel 1 (Gated Compute & Score):** Compute the Gate projection and the Up projection. Apply the activation function to the Gate, and multiply them together. Write this dense intermediate tensor to High Bandwidth Memory (HBM). This serves as both the activation scores and the input for the next layer.
    * **Kernel 2 (Sparse Gather Matmul):** Accept the intermediate tensor, the Down projection weights, and the `top_k_indices`. Use `tl.load` with pointer arithmetic to dynamically gather *only* the Top-K active features and their corresponding rows in the Down projection. Compute the final output, strictly skipping all pruned neurons.
    * **Accumulator Precision:** Use `tl.zeros` with `tl.float32` for the accumulators to maintain mathematical precision, casting back to `tl.float16` or `tl.bfloat16` before storing.

### Step 4: Create the PyTorch Wrapper (Top-K Orchestrator)
* **Task:** Write the PyTorch forward pass that orchestrates the kernels and performs the strict sorting.
* **Requirement:**
    * **Phase 1 Execution:** Launch Kernel 1 to calculate the dense gated activations.
    * **Strict Selection:** Calculate the $L_1$ norm or absolute magnitude of the intermediate activations. Use PyTorch's `torch.topk(..., k=user_defined_k, dim=-1)` to extract the exact indices of the surviving neurons. 
    * **Phase 2 Execution:** Calculate the new grid launch dimensions using `triton.cdiv` based on the user-defined $k$ dimension, and launch Kernel 2 using the extracted indices.

### Step 5: Hugging Face Integration
* **Task:** Wrap the PyTorch orchestrator inside a standard `torch.nn.Module` designed to hot-swap into a Hugging Face model.
* **Requirement:**
    * **Signature Matching:** The module's `__init__` must accept the standard Hugging Face `PretrainedConfig` object. The `forward` pass must accept the exact same arguments as the target architecture's MLP layer.
    * **Weight Initialization:** Provide a helper script or method to correctly map and copy the pre-trained weights from the standard HF model into your custom layer's tensors.
    * **Autocast Compatibility:** Ensure the module correctly handles `torch.autocast` so it can run seamlessly in mixed-precision FP16/BF16 training or inference pipelines.

## Testing Protocol
Before finalizing any kernel, generate a random tensor test script that checks:
1. Max difference (`max_diff`) against the PyTorch baseline.
2. Execution time using `triton.testing.do_bench`.
