import argparse
import pandas as pd
import re
from tqdm import tqdm
from datasets import Dataset
from transformers import (
    BertTokenizerFast,
    AutoModelForTokenClassification,
    DataCollatorForTokenClassification,
    TrainingArguments,
    Trainer,
)
import evaluate


label2id = {"O": 0, "B-MOUNTAIN": 1, "I-MOUNTAIN": 2}
id2label = {v: k for k, v in label2id.items()}


def clean_text(text):
    """Remove extra spaces and line breaks."""
    return re.sub(r"\s+", " ", text).strip()


def tag_sentence(sentence, mountain):
    """Assign B/I/O tags to tokens in a sentence based on the mountain name."""
    tokens = sentence.replace(",", " ,").replace(".", " .").split()
    tags = ["O"] * len(tokens)

    ent_tokens = mountain.split()
    for i in range(len(tokens) - len(ent_tokens) + 1):
        if tokens[i:i + len(ent_tokens)] == ent_tokens:
            tags[i] = "B-MOUNTAIN"
            for j in range(1, len(ent_tokens)):
                tags[i + j] = "I-MOUNTAIN"
    return tokens, tags


def build_dataset_from_csv(csv_file):
    df = pd.read_csv(csv_file)
    df.dropna(subset=["sentence"], inplace=True)

    dataset = []
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Tagging sentences"):
        sentence = clean_text(row["sentence"])
        mountain = row["word"]
        tokens, tags = tag_sentence(sentence, mountain)
        tag_ids = [label2id[t] for t in tags]
        dataset.append({"tokens": tokens, "ner_tags": tag_ids})

    print(f"Built dataset with {len(dataset)} examples")
    return Dataset.from_list(dataset)


def tokenize_and_align_labels(example, tokenizer):
    tokenized = tokenizer(example["tokens"], is_split_into_words=True, truncation=True)
    word_ids = tokenized.word_ids()
    label_ids = []
    for word_idx in word_ids:
        label_ids.append(-100 if word_idx is None else example["ner_tags"][word_idx])
    tokenized["labels"] = label_ids
    return tokenized


def compute_metrics(eval_preds):
    metric = evaluate.load("seqeval")
    logits, labels = eval_preds
    predictions = logits.argmax(-1)

    true_preds, true_labels = [], []
    for pred, lab in zip(predictions, labels):
        tmp_pred, tmp_lab = [], []
        for p, l in zip(pred, lab):
            if l != -100:
                tmp_pred.append(p)
                tmp_lab.append(l)
        true_preds.append(tmp_pred)
        true_labels.append(tmp_lab)

    preds_str = [[id2label[p] for p in seq] for seq in true_preds]
    labels_str = [[id2label[l] for l in seq] for seq in true_labels]

    results = metric.compute(predictions=preds_str, references=labels_str)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


def train_model(csv_file, output_dir, epochs=3, batch_size=8):
    dataset = build_dataset_from_csv(csv_file)
    dataset = dataset.shuffle(seed=42)

    tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")
    tokenized_dataset = dataset.map(
        lambda ex: tokenize_and_align_labels(ex, tokenizer)
    )

    train_test = tokenized_dataset.train_test_split(test_size=0.2)
    train_dataset = train_test["train"]
    val_dataset = train_test["test"]

    model = AutoModelForTokenClassification.from_pretrained(
        "bert-base-cased",
        num_labels=len(label2id),
        id2label=id2label,
        label2id=label2id,
    )

    args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=batch_size,
        num_train_epochs=epochs,
        weight_decay=0.01,
        logging_dir="./logs",
        logging_steps=10,
        save_total_limit=1,
    )

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    trainer.train()
    model.save_pretrained(f"{output_dir}/ner")
    tokenizer.save_pretrained(f"{output_dir}/tokenizer")
    print("Model training complete.")


if __name__ == "__main__":
    train_path = "../data/mountain_sentences_final.csv"
    output = "../models/"
    parser = argparse.ArgumentParser(description="Train a BERT NER model.")
    parser.add_argument("--csv", type=str, default=train_path, help="Path to training CSV file.")
    parser.add_argument("--out", type=str, default=output, help="Output directory.")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs.")
    parser.add_argument("--batch", type=int, default=8, help="Batch size.")
    args = parser.parse_args()

    train_model(args.csv, args.out, args.epochs, args.batch)
