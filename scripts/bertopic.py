import numpy as np
import pandas as pd 
import re 
import emoji 
from sklearn.feature_extraction.text import CountVectorizer
from sentence_transformers import SentenceTransformer
import umap 
import hdbscan
from bertopic import BERTopic

class Bertopic:
    def __init__(self, text):
        self.em_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.text = text
        self.text_embeddings = self.get_embeddings(self.text)
        self.um_model = self.umap_model()
        self.hs_model = self.hdbscan_model()
        self.vectorizer_model = self.vectorizer()

    def get_embeddings(self, texts):
        embeddings = self.em_model.encode(texts, show_progress_bar=True)
        return embeddings

    def umap_model(self):
        um_model = umap.UMAP(
        n_neighbors=15,
        n_components=5,
        min_dist=0.0,
        metric="cosine",
        random_state=42
        )
        return um_model

    def hdbscan_model(self):
        hs_model = hdbscan.HDBSCAN(
        min_cluster_size=15,
        metric="euclidean",
        cluster_selection_method="eom",
        prediction_data=True)
        return hs_model

    def vectorizer(self):
        vectorizer_model = CountVectorizer(stop_words="english")
        return vectorizer_model

    def main(self):
        self.tc_model = BERTopic(
        embedding_model=self.em_model,
        umap_model=self.um_model,
        hdbscan_model=self.hs_model,
        vectorizer_model=self.vectorizer_model,
        calculate_probabilities=True,
        verbose=True)
        topics, probabs = self.tc_model.fit_transform(self.text, self.text_embeddings)
        return topics, probabs 