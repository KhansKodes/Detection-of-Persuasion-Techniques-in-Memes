# Deep Learning - Detection of Persuasion Techniques in Texts and Images

## Project Description
This project aims to detect persuasion techniques in memes, focusing on both textual and visual content. The main goal is to identify if a meme attempts to persuade viewers using propagandist claims, a common tactic in social media campaigns. The project is inspired by the SemEval-2021 Task 6 challenge and leverages state-of-the-art transformer models.

## Tasks/Subtasks
1. **Textual Multilabel Classification:** Identify persuasion techniques in meme text.
2. **Textual Span Detection:** Identify techniques and their corresponding text spans.
3. **Multimodal Classification:** Detect techniques using both text and image content.

## Datasets
The project uses several datasets for training, validation, and testing. The main data files are:
- `training_set_task1.txt`
- `dev_set_task1.txt`
- `test_set_task1.txt`

Each file contains a list of meme samples in JSON format, where each sample has:
- `id`: Unique identifier for the meme
- `labels`: List of persuasion techniques present in the meme (can be empty)
- `text`: The textual content of the meme

### Example Entry
```
{
    "id": "128",
    "labels": [
        "Black-and-white Fallacy/Dictatorship"
    ],
    "text": "THERE ARE ONLY TWO GENDERS\n\nFEMALE \n\nMALE\n"
}
```

### Persuasion Technique Labels
The following persuasion techniques are annotated in the dataset (not exhaustive, but representative):
- Loaded Language
- Name calling/Labeling
- Slogans
- Smears
- Causal Oversimplification
- Exaggeration/Minimisation
- Appeal to fear/prejudice
- Black-and-white Fallacy/Dictatorship
- Misrepresentation of Someone's Position (Straw Man)
- Doubt
- Glittering generalities (Virtue)
- Reductio ad hitlerum
- Flag-waving
- Repetition
- Obfuscation, Intentional vagueness, Confusion
- Whataboutism
- Presenting Irrelevant Data (Red Herring)
- Appeal to authority
- Thought-terminating cliché

> **Note:** Some memes have no persuasion technique labels (i.e., `labels: []`).

### Data Statistics
- **Training set:** 5171 samples
- **Development set:** 488 samples
- **Test set:** 1496 samples

## Techniques & Models
- Utilizes transformer-based models (BERT, XLNet, DeBERTa, ALBERT, RoBERTa, etc.) from Hugging Face.
- Implemented with PyTorch Lightning for modularity and reproducibility.
- Evaluation metrics: F1-micro and F1-macro scores.

## Project Structure
- Jupyter Notebook(s) for data processing, model training, and evaluation.
- Data loading and preprocessing scripts.
- Model definition and training routines.
- Visualization utilities for confusion matrices and class distributions.

## Installation & Requirements
- Python 3.8+
- Main dependencies:
  - `torch`
  - `pytorch-lightning`
  - `transformers`
  - `datasets`
  - `pandas`
  - `numpy`
  - `scikit-learn`
  - `matplotlib`
  - `opencv-python`
  - `Pillow`
- Install requirements with:
  ```bash
  pip install torch pytorch-lightning transformers datasets pandas numpy scikit-learn matplotlib opencv-python Pillow
  ```

## Usage
1. **Clone the repository and install dependencies.**
2. **Prepare the datasets** as described above.
3. **Run the notebook** to train and evaluate models on the tasks.
4. **Modify the `folder_name` variable** in the notebook to point to your dataset location.

## Results
- The models achieve strong performance on multilabel classification and span detection tasks.
- Macro and micro F1 scores are used for evaluation, with detailed results and confusion matrices available in the notebook.

## References
- [SemEval-2021 Task 6](https://propaganda.math.unipd.it/ptc/)
- [Hugging Face Transformers](https://huggingface.co/)
- [PyTorch Lightning](https://pytorch-lightning.readthedocs.io/)

## Acknowledgements
- Inspired by top-ranked teams in SemEval-2021 Task 6.
- Code adheres to PEP8 standards.
