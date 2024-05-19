import torch
from transformers import LlamaForCausalLM,LlamaConfig,LlamaModel
from OffloadHooks import *
import torch.nn as nn
#slightly fix the code from transformers.LlamaModel to support my offloading
class MyLlamaModel(LlamaModel):
    def __init__(self,config :LlamaConfig):
        super().__init__(config)
        self.__offloading = False
        
    #set config for offloading
    def enable_offloading(self):
        self.to('cpu')
        self.norm.to(torch.device('cuda'))
        self.embed_tokens.to(torch.device('cuda')) #为了方便，这俩模块就放GPU上了
        for decoder_layer in self.layers:
            decoder_layer.register_forward_pre_hook(offload_forward_pre_hook)
            decoder_layer.register_forward_hook(offload_forward_post_hook)
            decoder_layer.register_full_backward_pre_hook(offload_backward_pre_hook)
            decoder_layer.register_full_backward_hook(offload_backward_post_hook)
        for i in range(len(self.layers)):
            if i+1<len(self.layers):
                self.layers[i].next_module=self.layers[i+1]
            if i>0:
                self.layers[i].prev_module=self.layers[i-1]
        self.__offloading = True
    
    def state_dict(self, *args, **kwargs):
        if self.__offloading:
            for i in range(len(self.layers)):
                if hasattr(self.layers[i], "next_module"):
                    del self.layers[i].next_module
                if hasattr(self.layers[i], "prev_module"):
                    del self.layers[i].prev_module
        sd = super().state_dict(*args, **kwargs)
        if self.__offloading:
            for i in range(len(self.layers)):
                if i+1<len(self.layers):
                    self.layers[i].next_module=self.layers[i+1]
                if i>0:
                    self.layers[i].prev_module=self.layers[i-1]
        return sd
    
    def load_state_dict(self, sd, *args, **kwargs):
        if self.__offloading:
            for i in range(len(self.layers)):
                if hasattr(self.layers[i], "next_module"):
                    del self.layers[i].next_module
                if hasattr(self.layers[i], "prev_module"):
                    del self.layers[i].prev_module
        sd = super().load_state_dict(sd, *args, **kwargs)
        if self.__offloading:
            for i in range(len(self.layers)):
                if i+1<len(self.layers):
                    self.layers[i].next_module=self.layers[i+1]
                if i>0:
                    self.layers[i].prev_module=self.layers[i-1]
        return sd
        

class MyLlamaForCausalLM(LlamaForCausalLM):
    def __init__(self,config):
        super().__init__(config)
        self.model = MyLlamaModel(config) #change the original model to my customized model
        
    def enable_offloading(self):
        self.model.enable_offloading()
        if not next(self.lm_head.parameters()).is_cuda:
            self.lm_head.to('cuda')


    