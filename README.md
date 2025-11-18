# # GNN Benchmarking for Multi Asset Return Prediction

This repository contains the implementation and experiments for my **Master’s Thesis** on benchmarking Graph Neural Networks (GNNs) and traditional ML models (e.g. XGBoost) for **stock return prediction**.  
It includes the data preprocessing, graph construction, feature embedding (Node2Vec), model training, and evaluation workflow.

---

## 📂 Project Structure


This project implements the main experiments for the thesis:

- Graph neural networks on rolling correlation and graphical lasso graphs
- Non graph neural baselines (LSTM, MLP)
- Representation baselines (node2vec embeddings plus XGBoost)
- Portfolio level evaluation with long short top k strategy

To run an experiment on the HU GPU server:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

python train.py --config configs/gnn_corr.yaml
