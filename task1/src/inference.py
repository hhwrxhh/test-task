import argparse
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification


def run_inference(model_dir, text):
    tokenizer = AutoTokenizer.from_pretrained(f"{model_dir}/tokenizer")
    model = AutoModelForTokenClassification.from_pretrained(f"{model_dir}/ner")
    id2label = model.config.id2label

    tokens = tokenizer(text, return_tensors="pt", truncation=True)
    outputs = model(**tokens)
    predictions = torch.argmax(outputs.logits, dim=2)

    print("\nToken predictions:")
    for token_id, pred_id in zip(tokens["input_ids"][0], predictions[0]):
        token = tokenizer.convert_ids_to_tokens([token_id])[0]
        label = id2label[pred_id.item()]
        print(f"{token:<15} -> {label}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run NER inference with a trained model.")
    parser.add_argument("--model_dir", type=str, required=True, help="Path to model directory.")
    parser.add_argument("--text", type=str, required=True, help="Text for NER prediction.")
    args = parser.parse_args()

    run_inference(args.model_dir, args.text)
