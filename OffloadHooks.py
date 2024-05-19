import torch
import torch.nn as nn
import time
from transformers.models.llama.modeling_llama import LlamaDecoderLayer

#define some hooks to support parameter prefetch and compute overlap


compute_stream=torch.cuda.current_stream()
transfer_stream=torch.cuda.Stream()
enable_overlap = True #change it to False to disable offloading overlap


def LlamaDecoder2GPU(decoder:LlamaDecoderLayer,non_blocking=True):
    decoder.self_attn.to('cuda',non_blocking=non_blocking)
    decoder.mlp.to('cuda',non_blocking=non_blocking)
    decoder.input_layernorm.to('cuda',non_blocking=non_blocking)
    decoder.post_attention_layernorm.to('cuda',non_blocking=non_blocking)
    
def LlamaDecoder2CPU(decoder:LlamaDecoderLayer,non_blocking=True):
    decoder.self_attn.to('cpu',non_blocking=non_blocking)
    decoder.mlp.to('cpu',non_blocking=non_blocking)
    decoder.input_layernorm.to('cpu',non_blocking=non_blocking)
    decoder.post_attention_layernorm.to('cpu',non_blocking=non_blocking)


#给Decoder_layer注册,预取下一层
def offload_forward_pre_hook(module:LlamaDecoderLayer,input):
    is_first_layer = not hasattr(module,'prev_module')
    if is_first_layer:
        with torch.cuda.stream(transfer_stream):
            LlamaDecoder2GPU(module,non_blocking=False) #第一层采用阻塞式
    torch.cuda.synchronize()
    if hasattr(module,'next_module'):
        with torch.cuda.stream(transfer_stream):
            LlamaDecoder2GPU(module.next_module,non_blocking=True and enable_overlap)
        
def offload_forward_post_hook(module:LlamaDecoderLayer,input,output):
    with torch.cuda.stream(compute_stream):
        LlamaDecoder2CPU(module,non_blocking=True and enable_overlap)
        
#prefetch previous layer
def offload_backward_pre_hook(module:LlamaDecoderLayer,grad_output):
    is_first_layer = not hasattr(module,'next_module')
    if is_first_layer:
        with torch.cuda.stream(transfer_stream):
            LlamaDecoder2GPU(module,non_blocking=False) #第一层采用阻塞式
    torch.cuda.synchronize()
    if hasattr(module,'prev_module'):
        with torch.cuda.stream(transfer_stream):
            LlamaDecoder2GPU(module.prev_module,non_blocking=True and enable_overlap)
                
def offload_backward_post_hook(module,grad_input,grad_output):
    with torch.cuda.stream(compute_stream):
        LlamaDecoder2CPU(module,non_blocking=True and enable_overlap)