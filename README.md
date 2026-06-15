# Feature-Based Knowledge Distillation on CIFAR-100

![Python](https://img.shields.io/badge/python-3.8%2B-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)
![Course](https://img.shields.io/badge/Course-DL26-green)

Questo repository contiene il codice e la documentazione per il progetto finale del corso **Deep Learning: Advanced Models and Methods**.

Il progetto esplora l'efficacia di diverse strategie di **Knowledge Distillation**, sia basate sui logit (Vanilla KD) che sulle feature intermedie (FitNets, Attention Transfer); il fine è comprimere un modello _Teacher_ ad alta capacità (ResNet-50) in un modello _Student_ più compatto e veloce (ResNet-18) sul dataset CIFAR-100.

**Autore:** Salvatore Iurato (Gruppo G26, Progetto 4)

---

## 🎯 Obiettivo del Progetto

I modelli deep sono molto accurati ma costosi in fase di deployment. L'obiettivo principale è rispondere al seguente quesito:

> _Una ResNet-18 compatta può eguagliare o superare l'accuratezza di una baseline addestrata da zero con sola cross-entropy, distillando la conoscenza da una ResNet-50 fine-tuned su CIFAR-100? Quale strategia di distillazione intermedia funziona meglio?_

## 🗂 Struttura del Repository

L'alberatura del repository è la seguente:

```text
.
├── cluster/                # Script bash per l'esecuzione su cluster SLURM
├── data/                   # Dataset CIFAR-100 (scaricato automaticamente)
├── docs/                   # Documentazione (incluso il REPORT.md finale)
├── experiments/
│   ├── checkpoints/        # Pesi salvati dei modelli (.pth)
│   ├── configs/            # File YAML di configurazione degli esperimenti
│   └── logs/               # File di log e log di TensorBoard
├── figures/                # Immagini, grafici generati e heatmaps di attenzione
├── notebooks/              # Jupyter notebooks per esplorazione dati e test
├── scripts/                # Script di utilità per setup e visualizzazione
├── src/                    # Codice sorgente principale del modulo
│   ├── datasets/           # Script per caricamento e data augmentation
│   ├── evaluation/         # Metriche di valutazione e test
│   ├── models/             # Architetture (ResNet CIFAR-adapted)
│   ├── training/           # Funzioni di costo (KDLoss, ATLoss) e training loop
│   └── utils/              # Seed deterministico, logging e helper
├── LICENSE                 # Licenza del progetto
├── requirements.txt        # Dipendenze Python
└── README.md               # Questo file
```

## Esempio FitNet Deep: lancio degli script per l'addestramento

```Bash
python -m src.training.train_fitnet_stage1 --config experiments/configs/fitnet_deep_s1.yaml
```

```Bash
python -m src.training.train_fitnet_stage2 --config experiments/configs/fitnet_deep_s2.yaml
```

## 📊 Risultati Principali

| Modello   | Metodo               | Temperatura (T) | Top-1 Accuracy | Parametri (MB) | Latenza  |
| --------- | -------------------- | --------------- | -------------- | -------------- | -------- |
| ResNet-50 | Teacher (fine-tuned) | -               | 82.16%         | 90.63          | 0.172 ms |
| ResNet-18 | Baseline (CE-only)   | -               | 76.75%         | 42.84          | 0.038 ms |
| ResNet-18 | Vanilla KD           | 20              | 80.92%         | 42.84          | 0.038 ms |
| ResNet-18 | Attention Transfer   | 4               | 79.52%         | 42.84          | 0.030 ms |
| ResNet-18 | FitNet Deep S2       | 20              | 81.11%         | 42.84          | 0.031 ms |
