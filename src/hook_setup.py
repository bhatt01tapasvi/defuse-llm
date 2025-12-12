import torch
import numpy as np
import gc

def setup_hooks_gpt2(model, neuronDefuser, pre_ln1_activations, pre_attn_activations, 
                     post_attn_activations, pre_ln2_activations, pre_mlp1_activations,
                     pre_mlp2_activations, post_mlp2_activations, post_layer_activations,
                     mlp2_forward_proxy, embedding_weights, save_activations=False):
    """Setup hooks for GPT2 architecture."""
    hooks = []
    
    def create_hook_post(layer_name, sublayer_name):
        def hook_fn(module, inp, outp):
            if "layer" in sublayer_name and isinstance(outp, tuple):
                activation_tensor = outp[0]
            elif "attn" in sublayer_name and isinstance(outp, tuple):
                activation_tensor = outp[0]
            elif isinstance(outp, torch.Tensor):
                activation_tensor = outp
            else:
                activation_tensor = outp
            
            if not save_activations:
                return

            # TODO: Fix this .mean operation because we are sending just one batch, it's used for crushing the batch dimension.
            # the promptInEmbedSpace.py function uses the 2d activation_magnitude tensor so it has to be edited to accomodate 3d tensors.
            
            if len(activation_tensor.shape) == 3:
                activation_magnitude = activation_tensor.mean(dim=0)
            else:
                activation_magnitude = activation_tensor

            if "attn" in sublayer_name:
                post_attn_activations[layer_name].append(activation_magnitude.detach().cpu().numpy())
            elif "mlp2" in sublayer_name:
                post_mlp2_activations[layer_name].append(activation_magnitude.detach().cpu().numpy())
            elif "layer" in sublayer_name:
                post_layer_activations[layer_name].append(activation_magnitude.detach().cpu().numpy())
        return hook_fn
    
    def create_hook_pre(layer_name, sublayer_name):
        def hook_fn(module, inp):
            if isinstance(inp, tuple):
                inp_tensor = inp[0]
                inp_rest = inp[1:] if len(inp) > 1 else ()
            else:
                inp_tensor = inp
                inp_rest = ()

            # For mlp2, we MUST defuse neurons (this is critical path)
            if "mlp2" in sublayer_name:
                if save_activations:
                    pre_mlp2_activations[layer_name].append(inp_tensor.detach().cpu().numpy())
                modified_tensor = neuronDefuser.defuse_neurons(layer_name, inp_tensor)
                return (modified_tensor,) + inp_rest
            
            # Only save other activations if flag is enabled
            if not save_activations:
                return
            # TODO: Fix this .mean operation because we are sending just one batch, it's used for crushing the batch dimension.
            # the promptInEmbedSpace.py function uses the 2d activation_magnitude tensor so it has to be edited to accomodate 3d tensors.
            
            activation_magnitude = inp_tensor
            if len(inp_tensor.shape) == 3:
                activation_magnitude = inp_tensor.mean(dim=0)
            else:
                activation_magnitude = inp_tensor

            if "ln1" in sublayer_name:
                pre_ln1_activations[layer_name].append(activation_magnitude.detach().cpu().numpy())
            elif "ln2" in sublayer_name:
                pre_ln2_activations[layer_name].append(activation_magnitude.detach().cpu().numpy())
            elif "attn" in sublayer_name:
                pre_attn_activations[layer_name].append(activation_magnitude.detach().cpu().numpy())
            elif "mlp1" in sublayer_name:
                pre_mlp1_activations[layer_name].append(activation_magnitude.detach().cpu().numpy())
        return hook_fn

    for i, block in enumerate(model.transformer.h):
        ln_1 = block.ln_1
        attn = block.attn
        ln_2 = block.ln_2
        mlp = block.mlp
        
        hook_ln1 = ln_1.register_forward_pre_hook(create_hook_pre(f"layer_{i}", "ln1")) 
        hook_attn = attn.register_forward_pre_hook(create_hook_pre(f"layer_{i}", "attn"))
        hook_attn_post = attn.register_forward_hook(create_hook_post(f"layer_{i}", "attn"))
        hook_ln2 = ln_2.register_forward_pre_hook(create_hook_pre(f"layer_{i}", "ln2"))
        hooks.extend([hook_ln1, hook_attn, hook_attn_post, hook_ln2])

        if hasattr(mlp, 'c_fc'):
            hook_fc = mlp.c_fc.register_forward_pre_hook(create_hook_pre(f"layer_{i}", "mlp1"))
            hooks.append(hook_fc)

        if hasattr(mlp, 'c_proj'):
            hook_proj = mlp.c_proj.register_forward_pre_hook(create_hook_pre(f"layer_{i}", "mlp2"))
            hook_proj_post = mlp.c_proj.register_forward_hook(create_hook_post(f"layer_{i}", "mlp2"))
            hooks.extend([hook_proj, hook_proj_post])

        weight = block.mlp.c_proj.weight.data
        if save_activations:
            mlp2_forward_proxy[f'layer_{i}'] = (weight @ embedding_weights.T).detach().cpu().numpy()
        neuronDefuser.populate_forward_proxy(f'layer_{i}', weight, embedding_weights)
        
        del weight
        torch.cuda.empty_cache()

        hook_layer_post = block.register_forward_hook(create_hook_post(f"layer_{i}", "layer"))
        hooks.append(hook_layer_post)
    
    gc.collect()
    return hooks

def setup_hooks_llama(model, neuronDefuser, pre_ln1_activations, pre_attn_activations, 
                      post_attn_activations, pre_ln2_activations, pre_mlp1_activations,
                      pre_mlp2_activations, post_mlp2_activations, post_layer_activations,
                      mlp2_forward_proxy, embedding_weights, mlp2_weights, save_activations=False):
    """Setup hooks for LLaMA architecture."""
    hooks = []
    
    def create_hook_post(layer_name, sublayer_name):
        def hook_fn(module, inp, outp):
            if not save_activations:
                return
                
            if isinstance(outp, tuple):
                activation_tensor = outp[0]
            elif isinstance(outp, torch.Tensor):
                activation_tensor = outp
            else:
                activation_tensor = outp
            
            # TODO: Fix this .mean operation because we are sending just one batch, it's used for crushing the batch dimension.
            # the promptInEmbedSpace.py function uses the 2d activation_magnitude tensor so it has to be edited to accomodate 3d tensors.
            
            # Average over batch dimension, keep token dimension
            if len(activation_tensor.shape) == 3:
                activation_magnitude = activation_tensor.mean(dim=0)
            else:
                activation_magnitude = activation_tensor

            if "attn" in sublayer_name or "self_attn" in sublayer_name:
                post_attn_activations[layer_name].append(activation_magnitude.detach().cpu().numpy())
            elif "mlp_down" in sublayer_name:
                post_mlp2_activations[layer_name].append(activation_magnitude.detach().cpu().numpy())
            elif "layer" in sublayer_name:
                post_layer_activations[layer_name].append(activation_magnitude.detach().cpu().numpy())
        return hook_fn
    
    def create_hook_pre_with_kwargs(layer_name, sublayer_name):
        def hook_fn(module, args, kwargs):
            # Extract hidden_states from either args or kwargs
            if args and len(args) > 0:
                inp_tensor = args[0]
                args_rest = args[1:] if len(args) > 1 else ()
            elif kwargs and 'hidden_states' in kwargs:
                inp_tensor = kwargs['hidden_states']
                args_rest = ()
            else:
                # No hidden_states found, skip
                return
            
            if inp_tensor is None:
                return

            # For mlp_down, we MUST defuse neurons (critical path)
            if "mlp_down" in sublayer_name:
                if save_activations:
                    pre_mlp2_activations[layer_name].append(inp_tensor.detach().cpu().numpy())
                    
                modified_tensor = neuronDefuser.defuse_neurons(layer_name, inp_tensor)
                
                if args and len(args) > 0:
                    return ((modified_tensor,) + args_rest, kwargs)
                else:
                    kwargs['hidden_states'] = modified_tensor
                    return (args, kwargs)
            
            # TODO: Fix this .mean operation because we are sending just one batch, it's used for crushing the batch dimension.
            # the promptInEmbedSpace.py function uses the 2d activation_magnitude tensor so it has to be edited to accomodate 3d tensors.
            
            if not save_activations:
                return
                
            activation_magnitude = inp_tensor
            if len(inp_tensor.shape) == 3:
                activation_magnitude = inp_tensor.mean(dim=0)
            else:
                activation_magnitude = inp_tensor

            if "input_layernorm" in sublayer_name:
                pre_ln1_activations[layer_name].append(activation_magnitude.detach().cpu().numpy())
            elif "post_attention_layernorm" in sublayer_name:
                pre_ln2_activations[layer_name].append(activation_magnitude.detach().cpu().numpy())
            elif "self_attn" in sublayer_name:
                pre_attn_activations[layer_name].append(activation_magnitude.detach().cpu().numpy())
            elif "mlp_gate" in sublayer_name:
                pre_mlp1_activations[layer_name].append(activation_magnitude.detach().cpu().numpy())
        return hook_fn

    # LLaMA uses model.model.layers instead of model.transformer.h
    for i, layer in enumerate(model.model.layers):
        input_layernorm = layer.input_layernorm
        self_attn = layer.self_attn
        post_attention_layernorm = layer.post_attention_layernorm
        mlp = layer.mlp
        
        # Hook for input_layernorm (equivalent to ln_1)
        hook_ln1 = input_layernorm.register_forward_pre_hook(
            create_hook_pre_with_kwargs(f"layer_{i}", "input_layernorm"), 
            with_kwargs=True
        )
        
        # Hook for self_attn - IMPORTANT: use with_kwargs=True!
        hook_attn = self_attn.register_forward_pre_hook(
            create_hook_pre_with_kwargs(f"layer_{i}", "self_attn"), 
            with_kwargs=True
        )
        hook_attn_post = self_attn.register_forward_hook(create_hook_post(f"layer_{i}", "self_attn"))
        
        # Hook for post_attention_layernorm (equivalent to ln_2)
        hook_ln2 = post_attention_layernorm.register_forward_pre_hook(
            create_hook_pre_with_kwargs(f"layer_{i}", "post_attention_layernorm"), 
            with_kwargs=True
        )
        
        hooks.extend([hook_ln1, hook_attn, hook_attn_post, hook_ln2])

        # LLaMA MLP has gate_proj, up_proj, and down_proj
        # gate_proj corresponds to the first transformation (like c_fc)
        if hasattr(mlp, 'gate_proj'):
            hook_gate = mlp.gate_proj.register_forward_pre_hook(
                create_hook_pre_with_kwargs(f"layer_{i}", "mlp_gate"), 
                with_kwargs=True
            )
            hooks.append(hook_gate)

        # down_proj is the final projection (like c_proj in GPT2)
        if hasattr(mlp, 'down_proj'):
            hook_down = mlp.down_proj.register_forward_pre_hook(
                create_hook_pre_with_kwargs(f"layer_{i}", "mlp_down"), 
                with_kwargs=True
            )
            hook_down_post = mlp.down_proj.register_forward_hook(create_hook_post(f"layer_{i}", "mlp_down"))
            hooks.extend([hook_down, hook_down_post])

        # LLaMA uses nn.Linear which has shape (hidden_size, intermediate_size)
        # Forward proxy shape should be: (intermediate_size, vocab_size)
        weight = layer.mlp.down_proj.weight.data  # (hidden_size, intermediate_size)
        # embedding_weights: (vocab_size, hidden_size)
        # Transpose weight to get (intermediate_size, hidden_size), then multiply
        if save_activations:
            #mlp1_weights[f'layer_{i}'] = layer.mlp.gate_proj.weight.data.detach().cpu().numpy()
            mlp2_weights[f'layer_{i}'] = weight.detach().cpu().numpy()
            mlp2_forward_proxy[f'layer_{i}'] = (weight.T @ embedding_weights.T).detach().cpu().numpy()  # (intermediate_size, vocab_size)
        neuronDefuser.populate_forward_proxy(f'layer_{i}', weight.T, embedding_weights)
        
        del weight
        torch.cuda.empty_cache()

        hook_layer_post = layer.register_forward_hook(create_hook_post(f"layer_{i}", "layer"))
        hooks.append(hook_layer_post)
    
    gc.collect()
    return hooks