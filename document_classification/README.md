# Document Classification with GPT-2 Embedding

I used a pre-trained GPT-2 model as a word embedding generator. 
Due to the varying lengths of the input texts, the embeddings were processed individually, with each input consisting of a single text vector. 
After generating the embeddings, I averaged them along the length dimension to ensure all vectors have the same size. 
Finally, these processed text vectors were grouped into batches to serve as input for the LSTM classifier.
Figure below shows a brief model architecture.

<img width="394" alt="Screenshot 2024-12-24 at 10 47 01 PM" src="https://github.com/user-attachments/assets/393f1548-c469-428d-944c-f0ecd065f2f9" />

## News Category Dataset Visualization

<img width="772" alt="Screenshot 2024-12-24 at 10 43 08 PM" src="https://github.com/user-attachments/assets/bcfede8e-40f2-480f-bbff-c06500eefa1c" />

## See with Other Models