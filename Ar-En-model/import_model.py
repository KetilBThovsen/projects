import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

import math
import time
import torch.nn as nn
import torch
import torch.optim as optim
import random
from typing import Tuple

import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch import Tensor


path = r"C:\Users\KBT007\Desktop\Ketil\Ar-En-model\opus-mt-ar-en"

tokenizer = AutoTokenizer.from_pretrained(path, local_files_only=True)

model = AutoModelForSeq2SeqLM.from_pretrained(path, local_files_only=True)

model.eval()


def translate(input_str):

    input_ids = tokenizer(input_str, return_tensors="pt").input_ids


    res = model.generate(input_ids=input_ids)

    print([tokenizer.decode(el, skip_special_tokens=True) for el in res])
    
    
def count_parameters(model: nn.Module):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


print(f'The model has {count_parameters(model):,} trainable parameters')





N_EPOCHS = 10
CLIP = 1

best_valid_loss = float('inf')

for epoch in range(N_EPOCHS):

    start_time = time.time()

    train_loss = train(model, train_iter, optimizer, criterion, CLIP)
    valid_loss = evaluate(model, valid_iter, criterion)

    end_time = time.time()

    epoch_mins, epoch_secs = epoch_time(start_time, end_time)

    print(f'Epoch: {epoch+1:02} | Time: {epoch_mins}m {epoch_secs}s')
    print(f'\tTrain Loss: {train_loss:.3f} | Train PPL: {math.exp(train_loss):7.3f}')
    print(f'\t Val. Loss: {valid_loss:.3f} |  Val. PPL: {math.exp(valid_loss):7.3f}')

test_loss = evaluate(model, test_iter, criterion)

print(f'| Test Loss: {test_loss:.3f} | Test PPL: {math.exp(test_loss):7.3f} |')

