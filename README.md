# Seq2Seq Neural Machine Translation | WPI December 2022

This project implements a seq2seq neural machine translation model to translate English sentences to French using PyTorch. The project is structured into three model versions:

1. Supervised training
2. Pretraining as an autoencoder
3. Employing pre-trained GloVe embeddings for the encoder

These strategies showcase innovative approaches to enhance the performance of the machine translation model.

## Tech Stack

The project is implemented using the following technologies:

- Python: The primary programming language used for implementing the machine translation model.
- PyTorch: An open-source machine learning library used for implementing the seq2seq model.
- Jupyter Notebook: An open-source web application that allows the creation and sharing of documents that contain live code, equations, visualizations, and narrative text.

## Prerequisites

Before you begin, ensure you have met the following requirements:

- You have installed the latest version of Python.
- You have a Windows/Linux/Mac machine.
- You have read the [PyTorch documentation](https://pytorch.org/docs/stable/index.html).
- You have a basic understanding of Machine Learning and Neural Networks.

## Project Structure

The project is contained in a Jupyter notebook file named `homework6_version2_MCDZWILSSHARMA4.ipynb`. The notebook includes the implementation of the seq2seq model, training and testing procedures, and visualization of results.

## Testing and Validation

The data is partitioned into 80% for training and 20% for testing. The model's effectiveness and accuracy are ensured by systematically reporting the testing loss at critical epochs during the training process.

## Visualization

The notebook includes code to visualize the attention mechanism of the seq2seq model. For example, the following code snippet visualizes the attention weights when translating the French sentence "je suis trop froid .":

```python
attentions = evaluate(encoder1, attn_decoder1, "je suis trop froid .")
plt.matshow(attentions.numpy())