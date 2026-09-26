"""Turning a question into the vector the semantic layer compares.

Two sentence-embedding models run locally through ONNX Runtime, and a TF-IDF baseline
learned from the training questions shows what a model buys over word overlap.
"""
import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import make_union

from . import data


def unit(v):
    return (v / np.maximum(np.linalg.norm(v, axis=1, keepdims=True), 1e-12)).astype(np.float32)


class OnnxEmbedder:
    def __init__(self, name="bge-small", max_tokens=128):
        import onnxruntime as ort
        from tokenizers import Tokenizer
        folder = data.model_dir(name)
        _, _, _, onnx_file, self.pooling = data.MODELS[name]
        self.name = name
        self.tokenizer = Tokenizer.from_file(str(folder / "tokenizer.json"))
        self.tokenizer.enable_truncation(max_length=max_tokens)
        self.tokenizer.enable_padding(pad_id=0, pad_token="[PAD]")
        options = ort.SessionOptions()
        options.intra_op_num_threads = 4
        self.session = ort.InferenceSession(str(folder / onnx_file), options, providers=["CPUExecutionProvider"])
        self.inputs = {i.name for i in self.session.get_inputs()}

    def encode(self, texts, batch=64):
        out = []
        for i in range(0, len(texts), batch):
            enc = self.tokenizer.encode_batch(list(texts[i:i + batch]))
            ids = np.array([e.ids for e in enc], dtype=np.int64)
            mask = np.array([e.attention_mask for e in enc], dtype=np.int64)
            feeds = {"input_ids": ids, "attention_mask": mask}
            if "token_type_ids" in self.inputs:
                feeds["token_type_ids"] = np.zeros_like(ids)
            hidden = self.session.run(None, feeds)[0]
            if self.pooling == "cls":
                vectors = hidden[:, 0]
            else:
                vectors = (hidden * mask[..., None]).sum(1) / np.maximum(mask.sum(1, keepdims=True), 1)
            out.append(unit(vectors))
        return np.vstack(out) if out else np.zeros((0, 1), np.float32)


class TfidfEmbedder:
    """Word and character n-grams, reduced to 256 dimensions: similarity by shared wording."""

    name = "tfidf"

    def __init__(self, corpus, dims=256, seed=0):
        self.vec = make_union(TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=2),
                              TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True, min_df=2))
        x = self.vec.fit_transform(corpus)
        self.svd = TruncatedSVD(dims, random_state=seed).fit(x)

    def encode(self, texts, batch=None):
        return unit(self.svd.transform(self.vec.transform(list(texts))))


def embedder(name):
    if name == "tfidf":
        return TfidfEmbedder([t for t, _ in data.questions("train")])
    return OnnxEmbedder(name)
