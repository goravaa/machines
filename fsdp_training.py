# fsdp_training.py
import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.utils.data import DataLoader, DistributedSampler
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from datasets import load_dataset

# Define the initialization function
def init_distributed(rank, world_size):
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)

# Define the model creation function
def create_model(rank):
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(rank)
    model = FSDP(model)
    return model

# Define data loading with DistributedSampler
def get_data_loader(rank, world_size, batch_size=2):
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    def tokenize_function(example):
        return tokenizer(example["text"], padding="max_length", truncation=True, max_length=128)

    tokenized_dataset = dataset.map(tokenize_function, batched=True)
    tokenized_dataset.set_format(type="torch", columns=["input_ids", "attention_mask"])

    sampler = DistributedSampler(tokenized_dataset, num_replicas=world_size, rank=rank)
    return DataLoader(tokenized_dataset, batch_size=batch_size, sampler=sampler)

# Define the training function
def train(rank, world_size, epochs=1):
    init_distributed(rank, world_size)
    model = create_model(rank)
    data_loader = get_data_loader(rank, world_size)

    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)

    model.train()
    for epoch in range(epochs):
        for batch in data_loader:
            inputs = batch["input_ids"].to(rank)
            attention_mask = batch["attention_mask"].to(rank)

            optimizer.zero_grad()
            outputs = model(input_ids=inputs, attention_mask=attention_mask, labels=inputs)
            loss = outputs.loss
            loss.backward()
            optimizer.step()

            # Print loss from each GPU
            print(f"GPU {rank} - Epoch {epoch+1}, Loss: {loss.item()}")

    dist.destroy_process_group()

# Define the main function for spawning processes
def main():
    world_size = 2  # Number of GPUs
    epochs = 3      # Number of training epochs
    mp.spawn(train, args=(world_size, epochs), nprocs=world_size, join=True)

if __name__ == "__main__":
    main()
