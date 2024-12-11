# 1. Training Optimization Analysis with GPT-2

## 🌴🌳🌲 Branch Overview
Navigate to the appropriate branch and run `python3 train_gpt2.py` to see what happens❗️❗️❗️
<p align="center">
<img width="538" alt="Screenshot 2024-10-14 at 11 52 38 PM" src="https://github.com/user-attachments/assets/a27be35e-01a8-4338-b93a-d8d0d7b1dbf3">
</p>

## 🏃🏻💨 SpeedUp Results
The results in the table below compare the training speed improvements achieved by sequentially applying the methodologies for speed-up training shown in the graph above. 
The 'Time' column represents the time taken per epoch, measured in microseconds, while 'TPS' stands for tokens per second, indicating the number of tokens the model processes per second. 
Through seven stages of incremental optimization, a 386x improvement in training speed was achieved compared to the initial training speed.

![Screenshot 2024-11-23 at 4 01 05 PM](https://github.com/user-attachments/assets/fea489a0-6fc0-4280-a2e8-9999b6b3db5b)

## Details
Detailed information about default settings and all 12 training optimizations can be found [here](https://github.com/JeongHwaSik/nano-GPT2/blob/main/DETAILS.md)

</br>

# 2. Tokenizer
 Byte Pair Encoding tokenizer was first introduced in the [GPT-2 paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) as a practical compromise between character-level and word-level language modeling, and it operates independently of the language models (see the figrue below). The BPE tokenizer works by grouping relevant Unicode byte sequences using regular expressions, then progressively merging the most frequent repeated sequences into single tokens, step-by-step.
<p align="center">
  <img width="500" alt="Screenshot 2024-12-11 at 11 21 21 PM" src="https://github.com/user-attachments/assets/603567b7-f558-4f53-a2f1-5d185b4345f3" />
</p>

## 🏄‍♂️ Training Tokenizer
 I implemented Byte Pair Encoding (BPE) tokenizer based on [GPT-2 github page](https://github.com/openai/gpt-2/blob/master/src/encoder.py) and trained it with ‘BTS’ wikipedia page. See the results with the following command `python3 tokenizer.py`.
