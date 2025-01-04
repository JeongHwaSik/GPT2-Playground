import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import torch
import torch.optim as optim

from .dataset import NewsDataLoader
from .classifier import LSTMClassifier

# Train: python -m document_classification.train.py
learning_rate = 2e-4
iters = 167621 * 2 # epochs = 2
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"

ds = NewsDataLoader(device=device)
model = LSTMClassifier()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate)

model = model.to(device)
for i, iter in enumerate(range(iters)):
  x, y = ds.get_next_batch()
  x = x.to(device)
  y = y.to(device)
  optimizer.zero_grad()

  loss, _ = model(x, y)
  if i % 1000 == 0:
    print(f"{iter}/{iters} || Loss: {loss}")

  loss.backward()
  optimizer.step()


# Validation
x_val, y_val = ds.x_val, ds.y_val
total_n = len(y_val)
count = 0

for iter in range(1310): # validation_size(41906) / batch_size(32) = 1309.xx
  x, y = ds.get_next_val_batch()
  x = x.to(device)
  y = y.to(device)

  _, logits = model(x)

  pred = torch.argmax(logits, dim=-1)

  for predict, gt in zip(pred, y):
    if predict == gt:
      count += 1

print(f"RNN Classifier with GPT Embedding Accuracy: {(count * 100/total_n):.2f}%")