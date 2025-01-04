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
<p align="center">
<img width="700" alt="Screenshot 2024-11-23 at 4 01 05 PM" src="https://github.com/user-attachments/assets/fea489a0-6fc0-4280-a2e8-9999b6b3db5b">
</p>

## Details
Detailed information about default settings and all 12 training optimizations can be found [🔥here🔥](https://github.com/JeongHwaSik/GPT2-Playground/blob/main/optimization_analysis/README.md)

<br>
</br>

# 2. Tokenizer
 Byte Pair Encoding tokenizer was first introduced in the [GPT-2 paper](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) as a practical compromise between character-level and word-level language modeling, and it operates independently of the language models (see the figrue below). The BPE tokenizer works by grouping relevant Unicode byte sequences using regular expressions, then progressively merging the most frequent repeated sequences into single tokens, step-by-step.
<p align="center">
  <img width="500" alt="Screenshot 2024-12-11 at 11 21 21 PM" src="https://github.com/user-attachments/assets/603567b7-f558-4f53-a2f1-5d185b4345f3" />
</p>

## 🏄‍♂️ Training Tokenizer
 I implemented Byte Pair Encoding (BPE) tokenizer based on [GPT-2 github page](https://github.com/openai/gpt-2/blob/master/src/encoder.py) and trained it with BTS wikipedia page. See the results with the following command `python3 tokenizer.py`.
<p align="center">
<img width="500" alt="Screen Recording 2024-12-11 at 11 50 13 PM" src="https://github.com/user-attachments/assets/8e4da457-cbc6-4417-b104-344e0f97427b">
</p>

<br>
</br>

# 3. Document Classification 

I developed and compared several document classification models, including:

- Naive Bayes Classifier
- Linear Classifier with TF-IDF Embeddings
- Naive RNN Classifier
- RNN Classifier with GPT-2

The performance of these models was evaluated using top-1 accuracy as the primary metric. The comparative results are presented below. For a more detailed analysis, please refer to this [link](https://github.com/JeongHwaSik/GPT2-Playground/tree/main/document_classification).

<br>
</br>

# 4. Lyrics Generating AI

I split the `train_gpt2.py` file into several modular components: `config.py`, `dataset.py`, `model.py`, `train.py`, and `generate.py`. 
Afterward, I trained a GPT-2 decoder using the Spotify Million Song Dataset to create a lyrics generator.

- Train with single GPU
```
python train.py
```

- Generate lyrics given 
```
python generate.py
```
