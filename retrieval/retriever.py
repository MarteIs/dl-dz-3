from sklearn.feature_extraction.text import TfidfVectorizer
import torch

class TfIdfSearch:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
    
    def train(self, docs: list[str]):
        self.doc_vectors = self.vectorizer.fit_transform(docs)
    
    def search(self, queries: list[str], docs: list[str]):
        query_vecs = self.vectorizer.transform(queries)
        doc_vecs = self.vectorizer.transform(docs)
        scores = torch.tensor((query_vecs @ doc_vecs.T).toarray())
        return scores.argsort(descending=True)