# Ontotype Perturbations Prediction

## Installation

### Requirements

- python 3.6>=

Other dependencies are explained in the setup section. Tested on Windows and Linux-Ubuntu 18.04.

### Setup

1. Clone the GitHub repository (or download as zip and unzip):
```
git clone https://github.com/ShaharKSegal/Ontotype.git
```
        
2. Go to the ontotype folder.

3. Install all dependant packages (on conda install each package separately):
```
pip install -r requirements.txt
```

4. In the data folder, extract `gene2go.gz`, `go-basic.zip` and `goa_human.gaf.gz`

5. (Optional) unzip `imex_data.zip`

6. (Optional) Download Supplementary Table S3 from [Sahni and Vidal et al. paper](https://doi.org/10.1016/j.cell.2015.04.013).
    <br/> Place the file in the data dir under the name `1-s2.0-S0092867415004304-mmc3.xlsx` (This should be the default name)

## Run

Basic run:
<br/>(note that your first run might be slow due to first time computing of objects which are cached later on)
```
python run.py --data vidal --model asym_go
```
This will run the asymmetric model with IMEx data and plot the ROC curse with 5-Fold cross validation on the data.
<br/> To run on imex's data change the `-data` argument to `-data vidal`, alternatively chose `-data imex vidal` for both.

For a test run that will compare different models vs shuffled variants:
```
python run.py --data vidal --model asym_go --test
```

For more information on the different parameters that can be altered:
```
python run.py --help
```

## Using other datasets
Datasets are read by objects inheriting from `PPIData`. Your own implementation must inherit from it as well.
<br/> I recommand looking into `ppi_data.py` to understand what is necessary. `VidalPPIData` is a good example.
<br/> In order to unify your dataset with others add your usecase to `UnifiedPPIData.add_ppi`.
<br/>
<br/> For the `run.py` to recognize your new implementation,
add your option to the 'data' argument in the parser and add your case to lines referring to that argument.

## Creting your own model
All models inherit from `Model`. Most of the ontotype models inherit `GOModel`, which adds useful properties and functions to the basic model that are relevant for ontotype models.

<br/> For the `run.py` to recognize your new implementation,
add your option to the 'model' argument in the parser and add your case to lines referring to that argument.
