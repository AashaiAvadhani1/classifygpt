import numpy as np
import torch
from transformers import BertTokenizer, BertModel
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
class BiasHandler:

    def __init__(self, model_name="bert-base-uncased"):
        self.tokenizer = BertTokenizer.from_pretrained(model_name)
        self.model = BertModel.from_pretrained(model_name)
        self.model.eval()  # set model to eval mode for inference

    def get_embedding(self, word):
        tokens = self.tokenizer(word, return_tensors="pt")
        with torch.no_grad():
            outputs = self.model(**tokens)
        return outputs["last_hidden_state"][0][0].numpy()

    def compute_bias_direction(self, word_pairs):
        differences = []
        for word1, word2 in word_pairs:
            vec1 = self.get_embedding(word1)
            vec2 = self.get_embedding(word2)
            differences.append(vec1 - vec2)
        return np.mean(differences, axis=0)

    def neutralizeWord(self, word, biased_space):
        word_embedding = self.get_embedding(word)
        return self.neutralizeVector(word_embedding, biased_space)

    def neutralizeVector(self, vector, biased_space):
        num = (np.dot(vector, biased_space))
        denom = (np.linalg.norm(biased_space) ** 2)
        biased_component = (num / denom) * biased_space
        return vector - biased_component
    
    def tsne_reduce(self, embeddings, perplexity=5):
        tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=300)
        reduced_embeddings = tsne.fit_transform(embeddings)
        return reduced_embeddings
    
    def plot_embeddings(self, reduced_embeddings, labels):
        plt.figure(figsize=(10, 6))
        
        for i, label in enumerate(labels):
            x, y = reduced_embeddings[i, :]
            plt.scatter(x, y, marker='o', color='red' if 'female' in label else 'blue')
            plt.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points', ha='right', va='bottom')
        
        plt.xlabel("t-SNE 1")
        plt.ylabel("t-SNE 2")
        plt.title("2D t-SNE of BERT embeddings")
        plt.show()