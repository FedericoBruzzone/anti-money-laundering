|Decision Tree | Random Forest|
|-|-|
|[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg) ](https://colab.research.google.com/github/federicobruzzone/anti-money-laundering/blob/main/decision_tree.ipynb)| [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg) ](https://colab.research.google.com/github/federicobruzzone/anti-money-laundering/blob/main/random_forest.ipynb)|

# Anti Money Laundering

The project is based on the analysis of the «IBM Transactions for Anti Money Laundering» dataset published on Kaggle. The task is to implement a system which predicts whether or not a transaction is illicit, using the attribute "Is Laundering" as a label to be predicted.

## Starting point working locally

**Kaggle instructions**

1. Create `.env` file using the following template:
```env
KAGGLE_USER=
KAGGLE_KEY=
KAGGLE_DATASET_LINK=ealtman2019/ibm-transactions-for-anti-money-laundering-aml
VERBOSE=0
```

2. If you prefer to use Kaggle-style configuration, you need to create `.kaggle` folder in the root directory and add `kaggle.json` into using the following template:
```json
{
    "username":"",
    "key":""
}
```

**Create and start a new virtual environment**

`source create_venv.sh venv` 

**Start current virtual environment**

`source venv/bin/activate`

**Deactivate the current virtual environment**

`deactivate`

### Run the `main.py` file 

`python3 -m src.main`


