import torch
from torch.utils.data import DataLoader
from transformers import LlamaTokenizer
from MyLlama import MyLlamaForCausalLM
from transformers import LlamaConfig
import json
import datasets
from torch.nn.utils.rnn import pad_sequence
import time
import os
from argument import get_args
from torch.profiler import profile,ProfilerActivity
# 定义随机mask函数
def random_mask(input_ids, mask_prob=0.05, mask_token_id=-100):
    mask_tensor = torch.rand(input_ids.shape) < mask_prob
    mask_tensor = mask_tensor.to(torch.int)
    masked_input_ids = input_ids * (1 - mask_tensor)
    masked_input_ids[mask_tensor == 1] = mask_token_id
    return masked_input_ids

# 定义collate函数
def collate_fn(batch):
    input_ids = pad_sequence([torch.tensor(item['input_ids']) for item in batch], batch_first=True)
    attention_mask = pad_sequence([torch.tensor(item['attention_mask']) for item in batch], batch_first=True)
    return {'input_ids': input_ids, 'attention_mask': attention_mask}

# 推理函数
def inference(model, dataloader):
    for iter, batch in enumerate(dataloader):
        input_ids = batch['input_ids'].cuda()  # 将输入数据移动到GPU上
        attention_mask = batch['attention_mask'].cuda()
        start_time = time.time()
        # 执行模型推理
        outputs = model(input_ids)
        end_time = time.time()
        delta_time = end_time - start_time
        mem = torch.cuda.max_memory_reserved() / (1024 ** 3)
        print(f"Step {iter}, Inference time: {'%.3f'%(delta_time*1000)} ms, memory :{mem} GB")
# 重置GPU内存统计信息并清除缓存
torch.cuda.reset_peak_memory_stats()
torch.cuda.empty_cache()

# 获取参数
args = get_args()

current_dir = os.path.dirname(os.path.abspath(__file__))

# 准备模型配置
config_path = os.path.join(f'{current_dir}/llama/llama_{args.model_config}_hf.json')
model_config = {}
with open(config_path) as file:
    model_config = json.load(file)
config = LlamaConfig(**model_config)

# 准备数据集
tokenizer = LlamaTokenizer.from_pretrained(f'{current_dir}/llama/tokenizer.model',padding_side='left')
tokenizer.pad_token = tokenizer.eos_token
def preprocess_function(examples):
    return tokenizer(examples["text"], padding=True, truncation=True)
dataset_path = os.path.join(current_dir,'dataset/')
dataset = datasets.load_from_disk(dataset_path)
test_dataset = dataset["test"].map(preprocess_function, batched=True)
test_dataloader = DataLoader(test_dataset, batch_size=args.batch_size, num_workers=16, collate_fn=collate_fn)

# 初始化模型并移到GPU上
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
precision = torch.float32 if not torch.cuda.is_available() else \
            torch.float32 if torch.cuda.get_device_capability(device)[0] < 7 else \
            torch.float16
model = MyLlamaForCausalLM(config).to(device='cpu', dtype=precision)
model.eval()  # 将模型设为评估模式
if args.offloading:
    model.enable_offloading() 
    print('offload enabled')
else:
    model.cuda()
    print('inference soly on GPU')

inference(model, test_dataloader)
