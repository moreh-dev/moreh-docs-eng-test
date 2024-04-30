---
icon: terminal
tags: [guide]
order: 40
---
# 2. Moreh’s training code in-depth

Once you have prepared all the training data, let’s look at the contents of the `train_gpt.py` script that will execute the actual fine-tuning process. At this stage, you will realize that the MoAI Platform is fully compatible with PyTorch, so the training code is 100% identical to the PyTorch code that is used on Nvidia GPUs. **Plus, you will also see how efficiently complex paralellization techniques can be implemented on MoAI Platform.** 

First, we recommend that you follow the tutorial to the end using the script we provide.  Afterward, you can modify the script based on your need to fine-tune the Cerebras-GPT-13B model or other published models in many other ways. Moreh also prepared [LLM Fine-tuning 파라미터 가이드](https://www.notion.so/LLM-Fine-tuning-a169bf8a667c4a0689ec2d4ff464775b?pvs=21) for you to experience the best performance on MoAI Platform.

If you just want to skip reading the script and jump right into the fine-tuning, please go to “3. Execute Training”.

# Training Code

**All codes are completely identical to your normal PyTorch experience.**

First, import the required modules from the `transformers` library.

```python
from transformers import AutoModelForCausalLM, AdamW, AutoTokenizer
```

Import the model config and checkpoint released in Hugging Face.

```python
model = AutoModelForCausalLM.from_pretrained("cerebras/Cerebras-GPT-13B")
tokenizer = AutoTokenizer.from_pretrained("cerebras/Cerebras-GPT-13B") 
```

Load the preprocessed dataset saved during [1.Prepare Fine-tuning](https://www.notion.so/1-Prepare-Fine-tuning-94e2e2a98e6b4bf59476213ed204de31?pvs=21) step and define the data loader.

```python
  dataset = torch.load("gpt_dataset.pt")

  # Create a DataLoader for the training set
  train_dataloader = torch.utils.data.DataLoader(
      dataset,
      batch_size=args.batch_size,
      shuffle=True,
      drop_last=True,
  )
```

The rest of the training proceeds in the same way as model training using general PyTorch.

```python
    # Mask pad tokens for training
    def mask_pads(input_ids, attention_mask, ignore_index = -100):
        idx_mask = attention_mask
        labels = copy.deepcopy(input_ids)
        labels[~idx_mask.bool()] = ignore_index
        return labels

    # Define AdamW optimizer
    optim = AdamW(model.parameters(), lr=args.lr)

    # Start training
    for epoch in range(args.epoch):
        for i, batch in enumerate(train_dataloader, 0):
            input_ids = batch["input_ids"]
            attn_mask = batch["attention_mask"]
            labels = mask_pads(input_ids, attn_mask)
            outputs = model(
                input_ids.cuda(),
                attention_mask=attn_mask.cuda(),
                labels=labels.cuda(),
                use_cache=False,
            )

            loss = outputs[0]
            loss.backward()

            optim.step()
            model.zero_grad(set_to_none=True)
```

**As shown above, the code for MoAI Platform can be written in the same way as general PyTorch code.**

# Advanced Parallelism

The training script used in this tutorial, there is one additional line of code as shown below. This is the code that performs the best parallelization available exclusively on MoAI Platform.

```bash
torch.moreh.option.enable_advanced_parallelization()
```

The large language models such as  [Cerebras-GPT-13B](https://huggingface.co/cerebras/Cerebras-GPT-13B) used in this tutorial, require multiple GPUs to train. When using a framework other than MoAI Platform, parallelization techniques such as Data Parallel, Pipeline Parallel and Tensor Parellel must be adapted during model training.

For example, if a user wants to apply DDP in regular PyTorch code, the following code snippet should be added. (https://pytorch.org/tutorials/intermediate/ddp_tutorial.html)

```python
...
def setup(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
...

def main(rank, world_size, args):
	setup(rank, world_size)
...
	sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
	loader = DataLoader(dataset, batch_size=64, sampler=sampler)
...

...
world_size = torch.cuda.device_count()  # Change this if you want a different number of GPUs
rank = int(os.environ['LOCAL_RANK'])
main(rank, world_size, args)
...
```

```bash
# single node 실행
torchrun --standalone --nnodes=1 --nproc_per_node=8 train.py
# multi node 실행
torchrun --nnodes=2 --nproc_per_node=8 --rdzv_id=100 --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:29400 train.py
```

In addition to these basic settings, you must understand how the Python code works under a multi-processing environment while writing the training script. Especially, when training in a multi-node environment, you must do additional setup on the nodes used for training. Moreover, it takes a lot of time and effort to find the optimal parallelization method considering the type, size, and dataset of the model.

**On the other hand, MoAI Platform’s exclusive Advanced Parallelism allows you to conduct optimized parallel training by simply adding the following line of code to the training script without any need to apply additional parallelization techniques by yourself.** 

```python
import torch
...
torch.moreh.option.enable_advanced_parallelization()

model = AutoModelForCausalLM.from_pretrained("cerebras/Cerebras-GPT-13B")
tokenizer = AutoTokenizer.from_pretrained("cerebras/Cerebras-GPT-13B") 
...
```

Please experience **optimal distributed parallel processing** through MoAI Platform’s Advanced Parallelism (AP), a parallelization optimization and automation function that cannot be experienced in other frameworks.

 Using the AP function, you can obtain the optimal combination of parameters and environment variables for Pipeline Parallelism and Tensor Parallelism, which are generally required when training large-scale models, **with a single line of very simple code**.