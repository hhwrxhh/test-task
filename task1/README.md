# Mountain NER Project

### Overview

Since no suitable dataset was available online, a custom dataset was created for this Named Entity Recognition (NER) task.
Fifteen mountain names were selected, and sentences containing these names were collected from Wikipedia, resulting in around 900 sentences.

Because the data were initially imbalanced, additional sentences containing the word "mountain" were scraped from the web.
For mountain names with fewer than 80 examples, the word *“mountain”* was replaced with the respective mountain name — making the final dataset more balanced and representative.

---

## Setup Instructions

### 1. Create a Virtual Environment

Navigate to the project folder and run:

```bash
python -m venv myenv
```

### 2. Activate the Virtual Environment

**For Linux/macOS:**

```bash
source myenv/bin/activate
```

**For Windows (CMD):**

```bash
myenv\Scripts\activate.bat
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## Training the Model

To train the model, run:

```bash
python src/train.py --csv data/mountain_sentences_final.csv --epochs 5 --batch 16
```

---

## Inference

To make predictions using a trained model:

```bash
python src/inference.py --model_dir models/ --text "Everest is the best"
```

---

## Running the Notebook

1. Install the environment kernel:

   ```bash
   python -m ipykernel install --user --name=.myvenv --display-name "demo-env"
   jupyter notebook
   ```
2. Open the notebook file `demo.ipynb`.
3. Select the kernel **demo-env** and run all cells.

---

## Download Pretrained Models

Вownload the pretrained model and tokenizer from the link and place them in the `models/` folder:

**[Download pretrained model and tokenizer](https://drive.google.com/drive/folders/1LaRpOROiHCd6lKG9F_dDF6qlnrYN5Lk5?usp=sharing)**

---

## Possible Improvements

It would be beneficial to increase the number of sentences for each mountain name to provide the model with more diverse linguistic contexts.
Expanding the list of mountains would also improve the model’s generalization, allowing it to recognize a wider range of named entities related to geographical features.
Additionally, enlarging the dataset and experimenting with other transformer-based models could further enhance accuracy and robustness.
