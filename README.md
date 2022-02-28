# EL2Box_embedding

## Protein-protein Intersection Prediction

For PPI task, run `PPIExpTrain.py` to train, the default path of concept and relation will under the `./data/classPPIEmbed.pkl` and `./data/relationPPIEmbed.pkl`. The parameter can modify in the file.

then run `evaluate_interactions.py` to test, the output will show the result of ELBE. The results of other methods can be found at https://arxiv.org/pdf/1902.10499.pdf .



## Entailment of Equivalence Concepts

For EEC task, run `EECExpTrain.py` to train then run `EECExpTest.py` to get the result of ELBE.

run `BallEECExpTrain.py` to train then run `BallEECExpTest.py` to get the result of ELEm.

When run the train.py, the command line will output the triple, copy them to `ELBE/data/data-train/interGo.txt`.



## Dependency:

torch=1.9.0

absl-py==0.7.0

astor==0.7.1

Click==7.0

cycler==0.10.0

gast==0.2.2

grpcio==1.18.0

h5py==2.9.0

matplotlib==3.0.2

numpy==1.16.0

pandas==0.23.4

protobuf==3.6.1

pyparsing==2.3.1

python-dateutil==2.7.5

pytz==2018.9

scikit-learn==0.20.2

scipy==1.2.0

six==1.12.0