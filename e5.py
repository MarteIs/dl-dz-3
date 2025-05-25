from metrics import recall_at_k, mrr
from retrieval import load_data, TransformerRetreiver
import torch

if __name__ == "__main__":
    test = load_data(test=True)
    retriever = TransformerRetreiver()

    predict = retriever.retrieve(test)

    target = torch.arange(len(test))

    print(f"Recall@1: {recall_at_k(target, predict, k=1):.4f}",
          f"Recall@3: {recall_at_k(target, predict, k=3):.4f}",
          f"Recall@10: {recall_at_k(target, predict, k=10):.4f}",
          f"MRR: {mrr(target, predict):.4f}", sep="\n")
