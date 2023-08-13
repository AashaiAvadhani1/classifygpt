import os
from io import open
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import gensim as gs
from sklearn.manifold import TSNE
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix
from sklearn import metrics

#The package that determines whether text is written by chatgpt or not 
#Uses T-SNEs to visualize word embeddings 
class ClassifyGPT(object):
    
    def __init__(self, entire_df):
        self.orig_data = entire_df
        self.final_table = self.clean_data()
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.final_table["text"], self.final_table["label"],    test_size=0.3)
        self.model_lg = RandomForestClassifier(max_depth = 50)
        self.vectorizer = TfidfVectorizer()
        #rf = RandomForestClassifier(max_depth = 50)

    """
    Logistic Regression for now, if theres time add more models
    """
    def train_tfidf(self):
        #vectorizer = TfidfVectorizer()
        X_train_tfidf = self.vectorizer.fit_transform(self.X_train)
        X_test_tfidf = self.vectorizer.transform(self.X_test)
        self.model_lg.fit(X_train_tfidf, self.y_train)
        pr = self.model_lg.predict(X_test_tfidf)
        acc_score = metrics.accuracy_score(self.y_test,pr)
        
        # Compute the confusion matrix
        cm = confusion_matrix(self.y_test, pr)
        print("Confusion Matrix:\n", cm)
        return acc_score

    
    def predict_sentence(self, sentence):
        df = pd.DataFrame(columns=['text'])
        df.loc[0] = sentence
        meow = self.clean_prediction(df)
        vectored_text = self.vectorizer.transform(meow)
        prediction = self.model_lg.predict(vectored_text)
        pred_val = prediction.item()
        if(pred_val == 0):
            return "I think this was written by a human!"
        return "I think this was written by ChatGPT!"
        
        
    def remove_stop_words(self,sentence):
        stop_words = set(stopwords.words('english'))
        words = sentence.split() 
        filtered_sentence = [word for word in words if word.lower() not in stop_words]
        return ' '.join(filtered_sentence)
    
    def clean_prediction(self, df_prediction):
        return df_prediction['text'].apply(self.remove_stop_words)
    
    "Removes stop words and joins the data in a more formattable process"
    def clean_data(self):
        self.orig_data['human_answers'] = self.orig_data['human_answers'].apply(' '.join)
        self.orig_data['chatgpt_answers'] = self.orig_data['chatgpt_answers'].apply(' '.join)
        self.orig_data['human_written'] = 0
        self.orig_data['chat_gpt_written'] = 1
        self.orig_data['human_cleaned_sentences'] = self.orig_data['human_answers'].apply(self.remove_stop_words)
        self.orig_data['chat_gpt_cleaned_sentences'] = self.orig_data['chatgpt_answers'].apply(self.remove_stop_words)

        chat_gpt = self.orig_data[['chat_gpt_written', 'chatgpt_answers']]
        chat_gpt['text'] = chat_gpt['chatgpt_answers']
        chat_gpt['label'] = chat_gpt['chat_gpt_written']

        human_answers = self.orig_data[['human_written', 'human_answers']]
        human_answers['text'] = human_answers['human_answers']
        human_answers['label'] = human_answers['human_written']

        result = pd.concat([chat_gpt, human_answers], axis=0, ignore_index=True)
        result.drop(columns = ["chat_gpt_written", "chatgpt_answers", "human_written", "human_answers"])

        return result

    def visualize_top_words_embeddings(X_train, tokenizer, model, num_words=100):
        """
        Visualizes the embeddings of the top words in X_train using t-SNE.
        
        Parameters:
        - X_train: A list of sentences.
        - tokenizer: A tokenizer compatible with the model.
        - model: The BERT model to generate embeddings.
        - num_words: Number of top words to visualize. Default is 100.
        """
        
        # Tokenization and word frequency calculation
        all_words = [word_tokenize(sentence) for sentence in X_train]
        all_words = [word for sublist in all_words for word in sublist]
        word_freq = Counter(all_words)
        top_words = [word for word, freq in word_freq.most_common(num_words)]
        
        # Get embeddings for top words
        def get_bert_embeddings(words):
            embeddings = []
            for word in words:
                inputs = tokenizer(word, return_tensors='pt', truncation=True, padding=True, max_length=512)
                with torch.no_grad():
                    output = model(**inputs)
                embeddings.append(output.last_hidden_state.mean(dim=1).squeeze().numpy())
            return embeddings

        embeddings = get_bert_embeddings(top_words)

        # Transform embeddings to 2D using t-SNE
        tsne = TSNE(n_components=2, random_state=42)
        embeddings_array = np.array(embeddings)
        embeddings_2d = tsne.fit_transform(embeddings_array)

        # Plotting
        x = embeddings_2d[:, 0]
        y = embeddings_2d[:, 1]

        plt.figure(figsize=(12, 12))
        plt.scatter(x, y)

        # Annotate each point with its word
        for i, word in enumerate(top_words):
            plt.annotate(word, (x[i], y[i]), fontsize=8, alpha=0.7)

        plt.show()
    



    def word_embedding_fit():
        pass

    
    def predict(self, sentence):
        pass
        #prediction of the model uses the internall sklearn .predict function
        
        #if label == 1:
        #    return "I think it was Written by a human!"
        #return "I think it was written by chatgpt"
    
    def remove_stop_words(self,sentence):
        stop_words = set(stopwords.words('english'))
        words = sentence.split() 
        filtered_sentence = [word for word in words if word.lower() not in stop_words]
        return ' '.join(filtered_sentence)

    
    def read_data(dataframe):
        pass
    
    
    """
    Uses a T-SNE to visualize the relationship of word embeddings
    """
    def visualize_embedding():
        pass
    
    
    """
    Word embedding neturalizing for 
    """
    def neturalize_bias():
        pass
    
    def decision_tree_train():
        pass
        
    
    
    def fit():
        pass
    
    def predict():
        pass