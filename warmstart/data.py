"""The public data and models Warmstart runs on, downloaded on first use from pinned
sources, checked against SHA-256 and cached in WARMSTART_DATA (default ~/.cache/warmstart).

- Banking77 (Casanueva et al., 2020, CC BY 4.0): 13,083 customer-support questions to a
  digital bank, each labelled with one of 77 intents. Two questions with the same intent
  want the same answer; that label is how a wrong cache hit is caught.
- bge-small-en-v1.5 (BAAI, MIT) and all-MiniLM-L6-v2 (Apache 2.0) as ONNX files, from the
  bucket the fastembed library downloads from.
"""
import csv
import hashlib
import io
import os
import tarfile
import urllib.request
from functools import lru_cache
from pathlib import Path

BANKING77 = "https://raw.githubusercontent.com/PolyAI-LDN/task-specific-datasets/57ec275d8078af65b7731c2a98be812d844a6d6b/banking_data"
SOURCES = {
    "train": (f"{BANKING77}/train.csv", "b06e26ac675513959a63135f11b94ea7786ed02da65db93a5650d8838cbc664b"),
    "test": (f"{BANKING77}/test.csv", "d12d6e3bc4c3103966ae786dc435913c0c563dfa328f5a3646d0e62cfeeb474d"),
}
MODELS = {   # name: (url, sha256, folder in the archive, onnx file, pooling)
    "bge-small": ("https://storage.googleapis.com/qdrant-fastembed/fast-bge-small-en-v1.5.tar.gz",
                  "3858004b3822f64f940280874b8f2d2dc25b34a4f3eb3cdf617bdceeb21ed9ed", "fast-bge-small-en-v1.5", "model_optimized.onnx", "cls"),
    "minilm": ("https://storage.googleapis.com/qdrant-fastembed/sentence-transformers-all-MiniLM-L6-v2.tar.gz",
               "2735afe656e156af64ed603dbb1c96f3cae7f937286a8feb27fff7fa979f6a77", "fast-all-MiniLM-L6-v2", "model.onnx", "mean"),
}

# Intents whose answer depends on the customer's own account: their card's delivery, their
# declined payment, their missing transfer. The assistant looks these up, so the answer
# carries that customer's data and must never be served to anyone else. Everything else
# (fees, limits, how to verify, which countries) has one answer for everybody.
PERSONAL = {
    "Refund_not_showing_up", "balance_not_updated_after_bank_transfer", "balance_not_updated_after_cheque_or_cash_deposit",
    "beneficiary_not_allowed", "card_arrival", "card_payment_fee_charged", "card_payment_not_recognised",
    "card_payment_wrong_exchange_rate", "cash_withdrawal_not_recognised", "declined_card_payment", "declined_cash_withdrawal",
    "declined_transfer", "direct_debit_payment_not_recognised", "extra_charge_on_statement", "failed_transfer",
    "pending_card_payment", "pending_cash_withdrawal", "pending_top_up", "pending_transfer", "reverted_card_payment?",
    "top_up_failed", "top_up_reverted", "transaction_charged_twice", "transfer_fee_charged",
    "transfer_not_received_by_recipient", "wrong_amount_of_cash_received", "wrong_exchange_rate_for_cash_withdrawal",
}
# Intents whose answer quotes a fee or a rate; a pricing change has to invalidate them.
FEES = {
    "atm_support", "card_payment_fee_charged", "cash_withdrawal_charge", "exchange_charge", "exchange_rate",
    "getting_spare_card", "order_physical_card", "top_up_by_bank_transfer_charge", "top_up_by_card_charge",
    "top_up_limits", "transfer_fee_charged", "disposable_card_limits",
}


def cache_dir() -> Path:
    path = Path(os.environ.get("WARMSTART_DATA", Path.home() / ".cache" / "warmstart"))
    path.mkdir(parents=True, exist_ok=True)
    return path


def download(url, sha, name):
    path = cache_dir() / name
    if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest() == sha:
        return path
    with urllib.request.urlopen(url, timeout=300) as r:
        body = r.read()
    got = hashlib.sha256(body).hexdigest()
    if got != sha:
        raise RuntimeError(f"{name}: expected sha256 {sha}, downloaded {got}")
    path.write_bytes(body)
    return path


@lru_cache(maxsize=2)
def questions(split):
    """[(text, intent)] for the train or test split."""
    url, sha = SOURCES[split]
    with open(download(url, sha, f"banking77-{split}.csv"), encoding="utf-8") as f:
        return [(row["text"].strip(), row["category"]) for row in csv.DictReader(f)]


def model_dir(name) -> Path:
    url, sha, folder, _, _ = MODELS[name]
    target = cache_dir() / folder
    if not (target / "tokenizer.json").exists():
        archive = download(url, sha, f"{folder}.tar.gz")
        with tarfile.open(archive) as tar:
            members = [m for m in tar.getmembers() if not Path(m.name).name.startswith("._")]
            tar.extractall(cache_dir(), members=members, filter="data")
    return target
