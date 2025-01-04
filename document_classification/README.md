# Document Classification with GPT-2 Embedding

I used a pre-trained GPT-2 model as a word embedding generator. 
Due to the varying lengths of the input texts, the embeddings were processed individually, with each input consisting of a single text vector. 
After generating the embeddings, I averaged them along the length dimension to ensure all vectors have the same size. 
Finally, these processed text vectors were grouped into batches to serve as input for the LSTM classifier.
Figure below shows a brief model architecture.

<p align="center">
<img width="500" alt="Screenshot 2024-12-24 at 10 47 01 PM" src="https://github.com/user-attachments/assets/393f1548-c469-428d-944c-f0ecd065f2f9" />
</p>

## News Category Dataset Visualization

<p align="center">
<img width="1000" alt="Screenshot 2024-12-24 at 10 43 08 PM" src="https://github.com/user-attachments/assets/bcfede8e-40f2-480f-bbff-c06500eefa1c" />
</p>

## See with Other Models
<p align="center">
<img width="485" alt="Screenshot 2025-01-04 at 11 21 30 PM" src="https://github.com/user-attachments/assets/c28f2e45-45a7-4c24-847a-6b817737f331" />
</p>