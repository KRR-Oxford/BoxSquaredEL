# Box<sup>2</sup>EL: Concept and Role Box Embeddings for the Description Logic EL++

This is the implementation of the paper *Box<sup>2</sup>EL: Concept and Role Box Embeddings for the Description Logic EL++*. All results given in the paper can be verified by running the appropiate model with the hyperparameters listed in the appendix on the relevant datasets, which are included in the `data` folder.

## Requirements

This implementation requires a working installation of PyTorch 1.12.0 (optionally with GPU support for faster training and inference). You will additionally need the following Python packages: `numpy==1.22.3`, `tqdm==4.64.0`, `wandb==0.13.9`.

## Data

All data used in the paper is included in the `data` folder (obtained by unzipping `data.zip`). Our novel benchmark for subsumption prediction is included in the `prediction` subfolder of the folder of the relevant ontology, i.e., the data for GALEN can be found in `data/GALEN/prediction`. The data is split into training, validation and testing sets in the relevant subfolders, and we include `json` files that specify the mapping of classes and relations to integers used in the tensor-based representation.

The data for the PPI prediction task can be found in the `PPI` subfolder of the relevant ontology. We again include the training/validation/testing split and the mapping from classes and relations to integers.

The deductive reasoning benchmark data is contained in the `inferences` subfolder of the relevant ontology. In consists of the training data in form of the full OWL ontology, and validation and testing sets as `json` files.

## Reproducing the experiments

In order to reproduce the experiments conducted in the paper, it suffices to edit the relevant training file with the desired combination of method and dataset. For instance, to run Box<sup>2</sup>EL for subsumption prediction on GALEN, you need to:
1. Open the file `train.py`
2. In the `run` function, set the `task` to `'prediction'` (or `'inferences'` for deductive reasoning)
3. Set the model and desired hyperparameters
4. Run the file

The model will be trained, which could take up to 30min–1h (or less if you have a powerful GPU). Once it is trained, the best performing model on the validation set will be evaluated on the testing set, and the results will be printed to the console.

To run the PPI prediction experiments, the same has to be done with the `train_ppi.py` file.

We also provide the script `run_many_seeds.py`, which executes the configuration in `train.py` five times and reports the average results.

## Troubleshooting

* **Out of memory**. If you run out of memory, this is most likely due to a too large batch size during rank computation. Try decreasing the `batch_size` default argument in `evaluate.py` (or `evaluate_ppi_boxsqel.py`).

If you encounter any other issues or have general questions about the code, feel free to send me a message at `firstname.lastname@keble.ox.ac.uk`.
