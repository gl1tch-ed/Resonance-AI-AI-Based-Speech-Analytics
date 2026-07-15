"""Fine-tune a transformer (BERT / RoBERTa) for sentiment classification.

Demonstrates transfer learning from a pretrained checkpoint plus a small
grid hyperparameter search (learning rate x batch size), selecting the run
with the best validation macro-F1. Run:

    python training/train_sentiment.py --model roberta-base --dataset tweet_eval

Requires: transformers, datasets, scikit-learn, accelerate.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np


def compute_metrics(eval_pred):
    from sklearn.metrics import accuracy_score, f1_score

    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }


def build_datasets(dataset: str, subset: str, tokenizer, max_len: int):
    from datasets import load_dataset

    ds = load_dataset(dataset, subset)
    def tok(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_len)
    ds = ds.map(tok, batched=True)
    return ds


def run_trial(args, lr: float, batch_size: int, num_labels: int, tokenized):
    from transformers import (
        AutoModelForSequenceClassification,
        DataCollatorWithPadding,
        Trainer,
        TrainingArguments,
    )
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForSequenceClassification.from_pretrained(args.model, num_labels=num_labels)

    targs = TrainingArguments(
        output_dir=f"checkpoints/lr{lr}_bs{batch_size}",
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size * 2,
        num_train_epochs=args.epochs,
        eval_strategy="epoch",
        save_strategy="no",
        weight_decay=0.01,
        warmup_ratio=0.1,
        fp16=args.fp16,
        logging_steps=50,
        report_to=[],
    )
    trainer = Trainer(
        model=model,
        args=targs,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    metrics = trainer.evaluate()
    return trainer, metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="roberta-base")
    ap.add_argument("--dataset", default="tweet_eval")
    ap.add_argument("--subset", default="sentiment")
    ap.add_argument("--epochs", type=float, default=3)
    ap.add_argument("--max-len", type=int, default=128)
    ap.add_argument("--fp16", action="store_true")
    ap.add_argument("--lrs", default="2e-5,3e-5")
    ap.add_argument("--batch-sizes", default="16,32")
    ap.add_argument("--out", default="checkpoints/best")
    args = ap.parse_args()

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    ds = build_datasets(args.dataset, args.subset, tokenizer, args.max_len)
    num_labels = len(set(ds["train"]["label"]))

    grid = list(itertools.product(
        [float(x) for x in args.lrs.split(",")],
        [int(x) for x in args.batch_sizes.split(",")],
    ))
    print(f"Hyperparameter search over {len(grid)} configs...")

    best = {"f1_macro": -1.0}
    best_trainer = None
    results = []
    for lr, bs in grid:
        print(f"\n=== trial lr={lr} batch_size={bs} ===")
        trainer, metrics = run_trial(args, lr, bs, num_labels, ds)
        row = {"lr": lr, "batch_size": bs, **metrics}
        results.append(row)
        if metrics["eval_f1_macro"] > best["f1_macro"]:
            best = {"lr": lr, "batch_size": bs, "f1_macro": metrics["eval_f1_macro"]}
            best_trainer = trainer

    Path(args.out).mkdir(parents=True, exist_ok=True)
    if best_trainer is not None:
        best_trainer.save_model(args.out)
        tokenizer.save_pretrained(args.out)
    Path("checkpoints/search_results.json").write_text(json.dumps(results, indent=2))
    print(f"\nBest config: {best}\nSaved best model to {args.out}")


if __name__ == "__main__":
    main()
