from typing import Iterable, List
from bert_score import BERTScorer
import numpy as np


def load_bertscore():
    scorer = BERTScorer("microsoft/deberta-xlarge-mnli", use_fast_tokenizer=True)

    def compute_bertscore(
        candidates: Iterable[str], references: Iterable[str]
    ) -> np.ndarray:
        precision, recall, f1 = scorer.score(candidates, references)
        return precision.numpy(), recall.numpy(), f1.numpy()

    return compute_bertscore
