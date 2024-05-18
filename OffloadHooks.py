import torch
import torch.nn as nn
import time
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

#define some hooks to support parameter prefetch and compute overlap


compute_stream=torch.cuda.current_stream()
transfer_stream=torch.cuda.Stream()
non_blocking=True #note: change it to False to disable overlap!
offload_done = torch.cuda.Event()
prefetch_done = torch.cuda.Event()

def LlamaDecoder2GPU(decoder:LlamaDecoderLayer):
    decoder.self_attn.to('cuda',non_blocking=non_blocking)
    decoder.mlp.to('cuda',non_blocking=non_blocking)
    decoder.input_layernorm.to('cuda',non_blocking=non_blocking)
    decoder.post_attention_layernorm.to('cuda',non_blocking=non_blocking)
    
def LlamaDecoder2CPU(decoder:LlamaDecoderLayer):
    decoder.self_attn.to('cpu',non_blocking=non_blocking)
    decoder.mlp.to('cpu',non_blocking=non_blocking)
    decoder.input_layernorm.to('cpu',non_blocking=non_blocking)
    decoder.post_attention_layernorm.to('cpu',non_blocking=non_blocking)


#给Decoder_layer注册,预取下一层
def offload_forward_pre_hook(module:LlamaDecoderLayer,input):
    is_first_layer = not hasattr(module,'prev_module')
    if hasattr(module,'next_module'):
        with torch.cuda.stream(transfer_stream):
            if not is_first_layer and not hasattr(module.prev_module,'prev_module'):
                offload_done.wait() #前两层不需要wait
            LlamaDecoder2GPU(module.next_module)
            if is_first_layer:
                LlamaDecoder2GPU(module) 
            prefetch_done.record()
    if hasattr(module,'next_module'):
        with torch.cuda.stream(compute_stream):
            prefetch_done.wait()  #最后一层不需要wait
        
def offload_forward_post_hook(module:LlamaDecoderLayer,input,output):
    with torch.cuda.stream(compute_stream):
        LlamaDecoder2CPU(module)
        if hasattr(module,'prev_module') and hasattr(module,'next_module'):
            offload_done.record() #第一层和最后一层不需要record
        
#prefetch previous layer
def offload_backward_pre_hook(module:LlamaDecoderLayer,grad_output):
    is_first_layer=not hasattr(module,'next_module')
    if hasattr(module,'prev_module'):
        with torch.cuda.stream(transfer_stream):
            if not is_first_layer and not hasattr(module.next_module,'next_module'):
                offload_done.wait() #倒数前两层不需要wait
            LlamaDecoder2GPU(module.prev_module)
            if is_first_layer:
                LlamaDecoder2GPU(module)   
            prefetch_done.record()
    if hasattr(module,'prev_module'):
        with torch.cuda.stream(compute_stream):
            prefetch_done.wait()         #最后一层不需要wait
                
def offload_backward_post_hook(module,grad_input,grad_output):
    with torch.cuda.stream(compute_stream):
        LlamaDecoder2CPU(module)
        if hasattr(module,'next_module') and hasattr(module,'prev_module'):
            offload_done.record()  #第一层和最后一层不需要record