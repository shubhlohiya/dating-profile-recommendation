## Profile recommendation system for Online Dating

This repository contains our code for the course project of the ME781 - Engineering Data Mining and Applications course conducted at IIT Bombay.

### Usage

#### **Install Dependencies**

```bash
pip install numpy pandas scikit-learn tqdm nltk argparse jupyter
```

#### **Download Dataset**

Download the [dataset](https://www.kaggle.com/andrewmvd/okcupid-profiles) from Kaggle and place the extracted csv in a `data/` directory in the root of the repository. 

#### **Preprocess Data**

Run the `EDA_and_Preprocessing.ipynb` notebook to view some EDA results and to pre-process the data. The preprocessed train and test splits of the data are also saved in `data/` directory.

#### **Generate Training Data for Supervised Learning**

```bash
python train_data_gen.py
```

#### **Train Model**

```bash
python train.py --model [logistic/SVM/MLP]
```

#### **Evaluate Recommendation Engine**

```bash
python eval.py --model [logistic/SVM/MLP] --testsize 100
```


**Note** Further details about implementation and evaluation can be found in the [report](./report.pdf) and the [code documentation](https://shubhlohiya.github.io/dating-profile-recommendation/).