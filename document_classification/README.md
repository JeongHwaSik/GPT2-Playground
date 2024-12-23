# Document Classification with GPT-2 Embedding

I will use a pre-trained GPT-2 model as a word embedding generator. 
Due to the varying lengths of the input texts, the embeddings will be processed individually, with each input consisting of a single text vector. 
After generating the embeddings, I will average them along the length dimension to ensure all vectors have the same size. 
Finally, these processed text vectors will be grouped into batches to serve as input for the RNN classifier.

## News Category Dataset Visualization
- Train Dataset Pie Chart
<img width="555" alt="Screenshot 2024-12-23 at 11 41 00 PM" src="https://github.com/user-attachments/assets/500bd230-f59d-4ec5-82b3-52790d2452ea" />

- Validation Dataset Pie Chart
<img width="555" alt="Screenshot 2024-12-23 at 11 42 46 PM" src="https://github.com/user-attachments/assets/0e758870-8165-4e06-bd99-ecd396ff6adb" />