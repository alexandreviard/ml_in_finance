# Projet **Machine Learning in Finance : Theoretical Foundations - ENSAE 3A**

> _Réimplémentation, en PyTorch, des principaux modèles du papier  
> « Hierarchical Multi-Scale Gaussian Transformer for Stock Movement Prediction » (2020)._  
> Résultats reproduits sur le **S&P 500 (2010-2019)** : comparaison CNN / LSTM / Transformer ± priors gaussiens.

---

## 🚀 Résultats (S&P 500 – données journalières)

| Modèle | Fenêtre *N* | Accuracy | MCC (×10<sup>-2</sup>) | Paramètres |
|--------|-------------|----------|------------------------|------------|
| CNN              | 40 | 0.529 | 5.8  | **6 977** |
| LSTM             | 40 | 0.534 | 8.6  | 13 473 |
| Transformer      | 40 | 0.554 | 8.5  | 152 619 |
| Transformer-GO   | 40 | **0.562** | **12.8** | 152 619 |

ℹ️ Les Transformers Gaussian + Orthogonalisation (« GO ») dépassent systématiquement les autres familles dès *N ≥ 20*.

---

## 📂 Arborescence

```
ml_in_finance/
├── ntbks/                
│   ├── main_ntbk.ipynb
│   └── checkpoints/metrics.csv
├── src/                  # Code (datasets, modèles, trainer…)
│   ├── data.py
│   ├── models_paper.py
│   ├── trainer.py
│   └── …
├── requirements.txt
├── setup.py
└── README.md             # ← VOUS ÊTES ICI
```

---

## ⚙️ Installation

```bash
git clone https://github.com/alexandreviard/ml_in_finance.git
cd ml_in_finance

pip install -e .
pip install -r requirements.txt
```

---

## 🏃‍♀️ Premier essai

```python
from src.data import MarketDataset, get_sp500_tickers
from src.models_paper import CNN_Daily
from src.trainer import BinaryTrainer
from src.utils import _init_weights, print_num_params

# 1) Dataset
data = MarketDataset(
    tickers=get_sp500_tickers(),
    start="2010-07-01", end="2019-07-01",
    window_N=10
)
Xtr, ytr, Xval, yval, Xte, yte = data.get_splits(tensor=True)

# 2) Modèle
model = CNN_Daily(N=10, F=Xtr.shape[2])
model.apply(_init_weights)
print_num_params(model)

# 3) Entraînement + test
trainer = BinaryTrainer(
    model=model,
    X_train=Xtr, y_train=ytr,
    X_val=Xval, y_val=yval,
    batch_size=128,
    num_epochs=5,
    lr=3e-4,
    name="CNN_Daily_SP500_N10"
)
trainer.fit()
trainer.evaluate(Xte, yte)   # métriques + log CSV
```

---

## 🔬 (Re)-entraîner rapidement

| Fichier | Rôle |
|---------|------|
| `src/data.py`         | Téléchargement & fenêtrage des séries |
| `src/models_paper.py` | Implémentations CNN / LSTM / Transformer |
| `src/trainer.py`      | Boucle d’entraînement + checkpoints |
| `ntbks/main_ntbk.ipynb` | Notebook démo (exécutable en script) |

Lancez :

```bash
python -m ntbks.main_ntbk
```

Tous les checkpoints vont dans `ntbks/checkpoints/`, les métriques s’empilent dans `metrics.csv`.


* Li et al., 2023 — *Hierarchical Multi-Scale Gaussian Transformer for Stock Movement Prediction*  
* Gu et al., 2020 — *Empirical Asset Pricing via Machine Learning*  
* Zhang & Zohren, 2022 — *Applications of Deep Learning in Finance*
