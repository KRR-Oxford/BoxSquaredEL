# Dual Box Embeddings for the Description Logic EL++

This repository is the official implementation of the paper [Dual Box Embeddings for the Description Logic EL++](https://arxiv.org/abs/2301.11118).

## Requirements

This implementation requires a working installation of PyTorch 1.12.0 (optionally with GPU support for faster training and inference). You will additionally need the following Python packages: `joblib==1.1.0`, `numpy==1.22.3`, `tqdm==4.64.0`, `wandb==0.13.9`.

## Data

To obtain the data, unzip `data.zip`:
```sh
unzip data.zip
```

Our benchmark for subsumption prediction is included in the `prediction` subfolder of the folder of the relevant ontology, e.g., the data for GALEN can be found in `data/GALEN/prediction`. The data is split into training, validation and testing sets in the relevant subfolders, and we include `json` files that specify the mapping of classes and relations to integers used in the tensor-based representation.

The data for the PPI prediction task can be found in the `PPI` subfolder. We again include the training/validation/testing splits and the mapping from classes and relations to integers.

The deductive reasoning benchmark data is contained in the `inferences` subfolder. It consists of the training data in form of the full OWL ontology, and validation and testing sets as `json` files.

## Training

In order to train Box<sup>2</sup>EL or one of the baseline methods, edit the file `train.py` (for subsumption prediction and deductive reasoning) or `train_ppi.py` (for PPI prediction) with the desired combination of method and dataset. For example, to run Box<sup>2</sup>EL for subsumption prediction on GALEN, you need to:
1. Open the file `train.py`
2. In the `run` function, set the `task` to `'prediction'` (or `'inferences'` for deductive reasoning)
3. Set the model and desired hyperparameters
4. Run the file

Training should finish within a couple of minutes on a GPU. The best performing model on the validation set will be saved, evaluated on the testing set, and the results will be printed to the console.

We also provide the script `run_many_seeds.py`, which executes the configuration in `train.py` five times and reports the average results.

## Evaluation

To evaluate trained models, we provide the files `evaluate.py` and `evaluate_ppi.py`.

## Results

Our model achieves the following performance (combined across normal forms) on subsumption prediction:

| Dataset | H@1  | H@10 | H@100 | Med | MRR | MR | AUC |
|---------|------|------|-------|-----|-----|----|-----|
| GALEN   | 0.05 | 0.20 | 0.35  | 669 | 0.10 | 4375 | 0.81 |
| GO      | 0.04 | 0.23 | 0.59  | 48  | 0.10 | 3248 | 0.93 |
| Anatomy | 0.16 | 0.47 | 0.70  | 13  | 0.26 | 2675 | 0.97 |

PPI prediction:

| Dataset | H@10 | H@10 (F) | H@100 | H@100 (F) | MR | MR (F) | AUC | AUC (F) |
|-|-|-|-|-|-|-|-|-
| Yeast |  0.11  | 0.33     | 0.64  | 0.87      | 168| 118    | 0.97| 0.98 |
| Human | 0.09   | 0.28     | 0.55  | 0.83      | 343| 269    | 0.98| 0.98 |

Approximating deductive reasoning:

| Dataset | H@1  | H@10 | H@100 | Med | MRR | MR | AUC |
|---------|------|------|-------|-----|-----|----|-----|
| GALEN   | 0.01 | 0.09 | 0.24  | 1003 | 0.03 | 2833 | 0.88 |
| GO      | 0.00 | 0.08 | 0.49  | 107  | 0.04 | 1689 | 0.96 |
| Anatomy | 0.01 | 0.09 | 0.44  | 152  | 0.04 | 1599 | 0.99 |


## Troubleshooting

* **Out of memory**. If you run out of memory, this is most likely due to a too large batch size during rank computation. Try decreasing the `batch_size` default argument in `evaluate.py`  / `evaluate_ppi.py`.

If you encounter any other issues or have general questions about the code, feel free to contact me at `mathias (dot) jackermeier (at) cs (dot) ox (dot) ac (dot) uk`.

## Citing

Please cite this work using the following BibTex entry:
```bibtex
@inproceedings{jackermeier2024dual,
    author = {Jackermeier, Mathias and Chen, Jiaoyan and Horrocks, Ian},
    title = {Dual Box Embeddings for the Description Logic {EL}++},
    year = {2024},
    doi = {10.1145/3589334.3645648},
    booktitle = {Proceedings of the ACM Web Conference 2024},
    series = {WWW '24}
}
```