# Action Plan: Progetto di Knowledge Distillation & FitNets

## Fase 1: Setup e Costruzione della Baseline

L'obiettivo di questa fase è stabilire l'infrastruttura di base e ottenere i parametri di riferimento (baseline) senza alcuna forma di distillazione.

1. **Scelta del Dataset:** \* Utilizzare **CIFAR-100**. È il dataset ideale: sufficientemente complesso per giustificare l'uso della distillazione, ma abbastanza leggero da permettere iterazioni rapide durante lo sviluppo.
2. **Definizione delle Architetture:**
   - **Teacher:** Una rete "larga" e performante (es. ResNet-50). Deve essere pre-addestrata.
   - **Student (FitNet):** Una rete progettata per essere "più sottile" (meno canali) e, idealmente, "più profonda" o uguale al teacher (es. una variante snella di ResNet-18 o MobileNetV2).
3. **Addestramento Baseline (Student Standalone):**
   - Addestrare lo Student da zero utilizzando unicamente la standard cross-entropy sui target reali (hard labels).
4. **Misurazioni Iniziali:** \* Registrare: Accuratezza sul Test Set (%), Dimensione del modello (MB dei pesi salvati), Latenza di inferenza (ms per singola immagine).

---

## Fase 2: Vanilla Knowledge Distillation (Logit-based KD)

Implementazione della distillazione classica basata sull'output finale, seguendo rigorosamente le formule di Hinton et al. (2015).

1. **Implementazione della Loss di Hinton:**
   La funzione di costo globale da minimizzare è una media pesata tra l'imitazione del Teacher e i target reali:
   `L_totale = (α * T^2 * L_soft) + ((1 - α) * L_hard)`

2. **Dettagli Implementativi della Loss:**
   - **L_soft:** È la divergenza Kullback-Leibler (KL) tra i logit dello Student e i logit del Teacher. **Attenzione:** Prima di applicare la funzione softmax per ottenere le probabilità, i logit di _entrambi_ i modelli devono essere divisi per il parametro di Temperatura `T` (es. T=3 o T=4).
   - **L_hard:** È la standard Cross-Entropy tra i logit dello Student e le etichette reali (Ground Truth). Qui la temperatura deve rimanere a `T=1`.
   - **Fattore T^2:** I gradienti della KL divergence scalano di 1/T^2. È cruciale moltiplicare L_soft per `T^2` per bilanciare il segnale di apprendimento rispetto a L_hard.
   - **Parametro α:** Bilancia le due loss. Usa un valore moderato/alto (es. 0.7 - 0.9) per dare priorità all'imitazione del teacher.
3. **Addestramento e Valutazione:** Addestrare lo Student con questa configurazione e confrontare le metriche con la Baseline.

---

## Fase 3: Feature-Based Distillation (Metodo FitNets)

Implementazione del trasferimento di conoscenza tramite le rappresentazioni intermedie (Hints), basata sul paper di Romero et al. (2015).

1. **Selezione dei Layer (Hint e Guided):**
   - Scegliere un layer intermedio del Teacher che fungerà da **Hint layer** (suggerimento).
   - Scegliere un layer intermedio dello Student che fungerà da **Guided layer** (layer guidato). Tipicamente, si scelgono i layer centrali delle rispettive reti.
2. **Costruzione del Regressore Convoluzionale:**
   - Poiché lo Student è più "sottile" (meno canali) e potrebbe avere dimensioni spaziali diverse, le feature del _Guided layer_ non combaciano con quelle dell'_Hint layer_.
   - Inserire un **layer convoluzionale addizionale (regressore)** attaccato al _Guided layer_ dello Student. Questo regressore proietta le feature dello Student nello stesso spazio dimensionale (stesso numero di canali e dimensione spaziale) dell'_Hint layer_ del Teacher.
3. **Stage-Wise Training (Procedura in 2 Stadi):**
   - **Stadio 1 (Pre-addestramento parziale):**
     - Congelare il Teacher.
     - Addestrare _solo_ la porzione dello Student che va dall'input fino al _Guided layer_, includendo il regressore.
     - La funzione di costo per questo stadio è esclusivamente la **Mean Squared Error (MSE)** tra l'output del regressore dello Student e l'output dell'_Hint layer_ del Teacher: `L_HT = 1/2 * MSE(Hint_Teacher, Regressore_Student)`.
     - Fermare l'addestramento quando la loss di validazione si stabilizza.
   - **Stadio 2 (Distillazione Globale):**
     - Rimuovere (o ignorare) il regressore.
     - Usare i pesi imparati nello Stadio 1 per inizializzare i primi layer dello Student. Inizializzare randomicamente il resto della rete Student.
     - Addestrare _l'intera_ rete Student utilizzando la **Vanilla KD Loss** definita nella Fase 2 (quindi `L_soft` + `L_hard`).

---

## Fase 4: Reportistica e Valutazione Finale

Per validare rigorosamente l'efficacia delle tecniche, compilare una tabella comparativa finale.

| Modello     | Metodo di Addestramento          | Accuratezza Test (%) | Dimensione (MB) | Latenza (ms) |
| :---------- | :------------------------------- | :------------------- | :-------------- | :----------- |
| **Teacher** | Standard Cross-Entropy           | ...                  | ...             | ...          |
| **Student** | Baseline (Nessuna Distillazione) | ...                  | ...             | ...          |
| **Student** | Vanilla KD (Hinton)              | ...                  | ...             | ...          |
| **Student** | FitNet (Stage-wise: Hint + KD)   | ...                  | ...             | ...          |
