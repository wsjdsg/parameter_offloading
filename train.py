import torch
from torch.utils.data import DataLoader, Dataset
from transformers import  LlamaTokenizer
from torch.optim import Adam
from argument import get_args
import os
from MyLlama import MyLlamaForCausalLM
from transformers import LlamaConfig
import json
import datasets
from torch.nn.utils.rnn import pad_sequence
import time
from torch.profiler import profile,ProfilerActivity

#make labels
def random_mask(input_ids, mask_prob=0.05, mask_token_id=-100):
    mask_tensor = torch.rand(input_ids.shape) < mask_prob
    mask_tensor = mask_tensor.to(torch.int)
    masked_input_ids = input_ids * (1 - mask_tensor)
    masked_input_ids[mask_tensor == 1] = mask_token_id
    return masked_input_ids

def collate_fn(batch):
    input_ids = pad_sequence([torch.tensor(item['input_ids']) for item in batch], batch_first=True)
    attention_mask = pad_sequence([torch.tensor(item['attention_mask']) for item in batch], batch_first=True)
    labels = random_mask(input_ids)
    return {'input_ids': input_ids, 'attention_mask': attention_mask, 'labels': labels}

# 训练函数
def train(model:MyLlamaForCausalLM, train_dataloader, optimizer:Adam, epochs, clip_grad):
    for epoch in range(epochs):
        total_loss = 0
        for iter, batch in enumerate(train_dataloader):
            input_ids = batch['input_ids'].cuda()
            attention_mask = batch['attention_mask'].cuda()
            labels = batch['labels'].cuda()
            start_time=time.time()
            optimizer.zero_grad()
            if use_fp16: #use autocast to avoid precision-caused gradient underflow
                with torch.cuda.amp.autocast():
                    outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss    
            else:
                outputs = model(input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss       
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip_grad) 
            #update on cpu
            optimizer.step()         
                    
            end_time=time.time()
            delta_time = end_time-start_time
            mem = torch.cuda.max_memory_reserved() / (1024 ** 3)
            print(f"Epoch {epoch}, Step {iter}, Loss: {loss.item()}, time: {'%.3f'%(delta_time*1000)} ms, mem: {mem} GB")
        model_save_path = f"model_epoch_{epoch + 1}.pt"
        # torch.save(model.state_dict(), model_save_path)
        print(f"Model saved at {model_save_path}")
        print(f"Average training loss for epoch {epoch+1}: {total_loss / len(train_dataloader)}")


torch.cuda.reset_peak_memory_stats()
torch.cuda.empty_cache()
args=get_args()
current_dir = os.path.dirname(os.path.abspath(__file__))
#prepare config
config_path = os.path.join(f'{current_dir}/llama/llama_{args.model_config}_hf.json')
model_config = {}
with open(config_path) as file:
    model_config = json.load(file)
config = LlamaConfig(**model_config)

#preapare dataset
tokenizer = LlamaTokenizer.from_pretrained(f'{current_dir}/llama/tokenizer.model')
tokenizer.pad_token = tokenizer.eos_token
def preprocess_function(examples):
    return tokenizer(examples["text"], padding=True, truncation=True)
dataset_path=os.path.join(current_dir,'dataset/')
dataset = datasets.load_from_disk(dataset_path)

train_dataset = dataset["train"].map(preprocess_function, batched=True)
test_dataset = dataset["test"].map(preprocess_function, batched=True)
train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, 
                              shuffle=True,num_workers=16, collate_fn=collate_fn)
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size,num_workers=16,collate_fn=collate_fn)



device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
precision = torch.float32 if not torch.cuda.is_available() else \
            torch.float32 if torch.cuda.get_device_capability(device)[0] < 7 else \
            torch.float16
use_fp16 = args.fp16
#first initialize model on cpu
model = MyLlamaForCausalLM(config).to(device=torch.device('cpu'), dtype=torch.float32)
model.train()
#enable_offloading
if args.offloading:
    model.enable_offloading() 
    print('offload enabled')
else:
    model.cuda()
    print('train soly on GPU')
optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
train(model, train_dataloader, optimizer, args.epochs,args.clip_grad)