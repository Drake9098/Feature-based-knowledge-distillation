# Feature-Based Knowledge Distillation on CIFAR-100

- **Group ID**: [G26]
- **Project ID**: [4]
- **Author**: Salvatore Iurato

---

## 1. Introduction and Objective

Deep neural networks have achieved strong performance on image classification, but their size and inference cost often make them impractical for deployment on resource-constrained hardware. **Knowledge Distillation (KD)** addresses this problem by transferring knowledge from a large, accurate _teacher_ model to a smaller, faster _student_ model.

This project focuses on **feature-based distillation**, where the student learns not only from the teacher's final predictions (logits), but also from its internal representations. The objective is to answer the following question:

> _Can a compact ResNet-18 student match or exceed the accuracy of a standalone baseline by distilling knowledge from a fine-tuned ResNet-50 teacher on CIFAR-100, and which distillation strategy works best?_

**Initial hypothesis.** It is expected that:

1. A student trained with vanilla logit-based KD (Hinton et al., 2015) will outperform a student trained from scratch with cross-entropy alone.
2. Feature-based methods (FitNets, Romero et al., 2015; Attention Transfer, Zagoruyko & Komodakis, 2017) will further improve the student by aligning intermediate representations, at the cost of more complex training pipelines.
3. The choice of hint layer(s) in FitNets and the distillation temperature _T_ will significantly affect the trade-off between accuracy, model size, and inference latency.

All methods are evaluated on the same dataset, with the same student architecture, reporting **top-1 test accuracy**, **model size (MB)**, and **inference latency (ms/image)** to assess both predictive performance and efficiency.

---

## 2. Contribution and Added Value

A complete, reproducible distillation pipeline in PyTorch was implemented, ranging from data loading to cluster-based training, rather than running an off-the-shelf library. The main contributions are:

- **End-to-end training framework.** Modular scripts for teacher fine-tuning, student baseline, vanilla KD, FitNets (two-stage), and Attention Transfer, all driven by YAML configuration files.
- **CIFAR-adapted ResNets.** Both teacher (ResNet-50) and student (ResNet-18) use a CIFAR-specific stem (`conv1` 3×3, stride 1; `maxpool` disabled) so that ImageNet-pretrained backbones can be transferred to 32×32 inputs without architectural mismatch.
- **Multiple feature-based strategies under fair comparison.** Three FitNet hint configurations (middle / deep / full multi-layer) plus Attention Transfer were implemented and compared, using the same frozen teacher checkpoint for all student experiments.
- **Correct distillation losses.** Custom `KDLoss` with temperature scaling and the _T²_ gradient rescaling factor (Hinton et al.); FitNet Stage 1 with 1×1 conv regressors and MSE hint loss; AT loss with *F*²_sum attention maps and L2-normalized spatial alignment (Zagoruyko & Komodakis).
- **Hyperparameter exploration.** Temperature sweeps on all KD-based student runs: _T_ ∈ {4, 8, 20} for Vanilla KD and AT; _T_ ∈ {3, 8, 20} for FitNet Stage 2.
- **Efficiency-aware evaluation.** Beyond accuracy, model footprint and inference latency are systematically measured to quantify the student–teacher compression benefit.

---

## 3. Data Used

### Source and statistics

The experiments use **CIFAR-100** (Krizhevsky, 2009), loaded via `torchvision.datasets.CIFAR100`:

| Split    | Samples | Classes | Image size  |
| :------- | ------: | ------: | :---------- |
| Training |  50,000 |     100 | 32 × 32 RGB |
| Test     |  10,000 |     100 | 32 × 32 RGB |

Each image belongs to one of 100 fine-grained object categories. The **official train/test partition** provided by the dataset is used; no additional validation split is held out during training. In alignment with standard literature for CIFAR-100 benchmarking, a separate validation subset was intentionally not carved out from the training data: with only 500 training images per class, further reducing this quota would significantly degrade the generalization capacity of high-capacity models such as the ResNet-50 teacher.

### Preprocessing and augmentation

Transforms are defined in `src/data/transforms.py` and applied consistently across all experiments:

**Training augmentations:**

- `RandomCrop(32, padding=4)`
- `RandomHorizontalFlip()`
- `RandomRotation(15°)`
- `ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2)`

**Evaluation (test set):**

- No augmentation; only `ToTensor()` and normalization.

**Normalization:** ImageNet statistics — mean `(0.485, 0.456, 0.406)`, std `(0.229, 0.224, 0.225)`. This choice is consistent with the ImageNet-pretrained teacher backbone and is a common practice when fine-tuning ResNets originally trained on ImageNet.

**DataLoader settings:** batch size 128, 4 workers, `pin_memory=True`, shuffle on train, no shuffle on test.

---

## 4. Methodology and Architecture

### 4.1 Overview of the experimental pipeline

All experiments follow a fixed pipeline to ensure comparability:

```
Teacher fine-tuning (ResNet-50, CE)
        ↓
Student baseline (ResNet-18, CE, no distillation)
        ↓
Distillation methods (ResNet-18 student, frozen teacher):
  • Vanilla KD
  • FitNets: Stage 1 (MSE hints) followed by Stage 2 (KD)
  • Attention Transfer + KD
```

The teacher is fine-tuned once and its best checkpoint is reused for every distillation run. FitNets Stage 2 and vanilla KD initialize the student from scratch with random weights or from Stage 1 weights (FitNets only), as described below.

### 4.2 Architectures

| Role        | Architecture | Key adaptation for CIFAR-100                                            |
| :---------- | :----------- | :---------------------------------------------------------------------- |
| **Teacher** | ResNet-50    | Stem modified: `conv1` 3×3 s1, `maxpool` → Identity; `fc` → 100 classes |
| **Student** | ResNet-18    | Same CIFAR stem; `fc` → 100 classes                                     |

The teacher is initialized from **ImageNet-1K pretrained weights** (`resnet50_imagenet1k_v1`), loading only shape-compatible layers. It is then fine-tuned on CIFAR-100 with **differential learning rates**: `lr = 0.01` for newly initialized layers (`conv1`, `fc`), `lr = 0.001` for the pretrained backbone (`backbone_lr_mult = 0.1`), cosine schedule, 100 epochs.

The student baseline is trained from random initialization for 200 epochs with SGD (`lr = 0.1`, momentum 0.9, weight decay 5×10⁻⁴) and cosine annealing.

During distillation, the **teacher is frozen**: only student parameters and FitNet regressors in Stage 1 are updated.

### 4.3 Vanilla Knowledge Distillation

The student loss combines hard labels and soft teacher predictions:

$$
\mathcal{L}_{\text{KD}} = \alpha \cdot \text{CE}(y, z_s) + (1 - \alpha) \cdot T^2 \cdot \text{KL}\!\left(\sigma(z_s / T)\,\|\,\sigma(z_t / T)\right)
$$

where $z_s, z_t$ are student and teacher logits, $y$ are ground-truth labels, $T$ is the temperature, and $\sigma$ is the softmax. The $T^2$ factor rescales gradients so that the soft-target term remains effective at high temperatures.

**Hyperparameters:** $\alpha = 0.1$, 200 epochs, SGD with cosine schedule (same optimizer settings as baseline). Distillation temperature is swept over **$T \in \{4, 8, 20\}$** with identical $\alpha$ and training schedule.

### 4.4 FitNets

FitNets uses a **two-stage, stage-wise** training procedure:

**Stage 1: Hint training.** A 1×1 convolutional regressor (`Conv2d + BatchNorm`) projects the student's guided-layer features into the teacher's hint-layer channel space. The loss is **MSE** between regressor output and teacher hint features. Only the student layers up to (and including) the guided layer, plus the regressor, are trained; deeper student layers and `fc` are frozen. Training: 100 epochs, `lr = 5×10⁻⁴`, SGD.

**Stage 2: Full network KD.** The regressor is discarded. The student is warm-started from Stage 1 weights and the **entire network** is fine-tuned with the vanilla KD loss (§4.3). Training: 200 epochs, $\alpha = 0.1$, cosine schedule. Stage 2 uses the same loss form as §4.3 but a separate temperature grid **$T \in \{3, 8, 20\}$**. The starting value of **3** was chosen to match the original FitNets paper, then extended to **8** and **20** to explore the temperature range.

Three hint configurations were implemented and compared:

| Config     | Hint layer(s)                          | Reasoning                                                  |
| :--------- | :------------------------------------- | :--------------------------------------------------------- |
| **Middle** | `layer2` (512 ch @ 16×16 → 128 ch)     | Original FitNets proposal: intermediate semantic guidance  |
| **Deep**   | `layer4` (2048 ch @ 4×4 → 512 ch)      | Align final backbone representations before the classifier |
| **Full**   | `layer1` to `layer4` (all four stages) | Continuous multi-scale guidance along the entire backbone  |

The chosen hint layers, other than **Middle** configuration, were motivated by the following considerations:

- **Deep:** Aims to see how the student adapts when the intermediate supervision is placed exclusively in the final stages of the feature extraction pipeline. By guiding only the deepest layers of the network, the objective was to allow the student maximum architectural freedom in the initial phase, where low-level features (edges, textures) are extracted. This configuration tests whether forcing an alignment on high-level, abstract semantic representations is sufficient to guide the student towards a better convergence, without restricting its capacity to find an autonomous path for elementary feature extraction.
- **Full:** Imposes a strict, dense multi-layer supervision across the entire backbone of both networks. This aims to evaluate the impact of continuous, step-by-step guidance throughout the whole optimization process.

### 4.5 Attention Transfer

Attention Transfer adds a spatial alignment term on top of KD, in a **single training stage** (no regressors):

$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{KD}} + \mathcal{L}_{\text{AT}}
$$

$$
\mathcal{L}_{\text{AT}} = \frac{\beta}{2} \sum_{j \in \mathcal{J}} \left\| \frac{Q_s^j}{\|Q_s^j\|_2} - \frac{Q_t^j}{\|Q_t^j\|_2} \right\|_2^2
$$

where $Q^j$ is the flattened *F*²_sum attention map (sum of squared activations over channels) at layer $j$, and $\mathcal{J} = \{\text{layer1, layer2, layer3, layer4}\}$. If spatial resolutions differ, the student map is bilinearly interpolated to the teacher resolution before normalization.

**Hyperparameters:** $\alpha = 0.9$ (KD term), $\beta = 100$, distillation temperature **$T \in \{4, 8, 20\}$**, 200 epochs, cosine LR schedule.

### 4.6 Evaluation metrics

For every trained model, the following metrics are reported:

| Metric                           | Description                                                                              |
| :------------------------------- | :--------------------------------------------------------------------------------------- |
| **Top-1 accuracy (%)**           | Classification accuracy on the 10,000-image CIFAR-100 test set                           |
| **Model size (MB)**              | Total parameter memory from `state_dict`                                                 |
| **Inference latency (ms/image)** | Mean forward-pass time per image on GPU, measured over 50 batches after 5 warmup batches |

All runs use `seed = 42` for reproducibility.

---

## 5. Results and Discussion

All reported student runs use **cosine annealing**, **seed = 42**, **200 epochs**, and the same fine-tuned teacher checkpoint (82.16% top-1). Metrics are taken from the `Final(best)` line of each training log (best test-set checkpoint during training). Temperature ablations use **$T \in \{4, 8, 20\}$** for Vanilla KD and AT, and **$T \in \{3, 8, 20\}$** for FitNet Stage 2 (FitNet was not run at $T = 4$).

### 5.1 Main results

**Table 1:** Test accuracy, model size, and inference latency across all methods (cosine scheduler). Best accuracy per method is in **bold**.

| Model     | Method               |  _T_   | Top-1 Acc. (%) | Size (MB) | Latency (ms) |
| :-------- | :------------------- | :----: | :------------: | :-------: | :----------: |
| ResNet-50 | Teacher (fine-tuned) |   —    |   **82.16**    |   90.63   |    0.172     |
| ResNet-18 | Baseline (CE only)   |   —    |     76.75      |   42.84   |    0.038     |
| ResNet-18 | Vanilla KD           |   4    |     80.67      |   42.84   |    0.031     |
| ResNet-18 | Vanilla KD           |   8    |     80.74      |   42.84   |    0.029     |
| ResNet-18 | Vanilla KD           |   20   |   **80.92**    |   42.84   |    0.038     |
| ResNet-18 | FitNet Middle S2     |   3    |     80.71      |   42.84   |    0.026     |
| ResNet-18 | FitNet Middle S2     |   8    |     80.78      |   42.84   |    0.056     |
| ResNet-18 | FitNet Middle S2     |   20   |   **80.95**    |   42.84   |    0.030     |
| ResNet-18 | FitNet Deep S2       |   3    |     80.55      |   42.84   |    0.049     |
| ResNet-18 | FitNet Deep S2       |   8    |     80.95      |   42.84   |    0.058     |
| ResNet-18 | **FitNet Deep S2**   | **20** |   **81.11**    |   42.84   |    0.031     |
| ResNet-18 | FitNet Full S2       |   3    |     80.51      |   42.84   |    0.031     |
| ResNet-18 | FitNet Full S2       |   8    |   **80.76**    |   42.84   |    0.045     |
| ResNet-18 | FitNet Full S2       |   20   |     80.62      |   42.84   |    0.061     |
| ResNet-18 | AT + KD              |   4    |   **79.52**    |   42.84   |    0.030     |
| ResNet-18 | AT + KD              |   8    |     79.45      |   42.84   |    0.051     |
| ResNet-18 | AT + KD              |   20   |     79.62      |   42.84   |    0.039     |

**Overall performance.** Every KD-based method beats the CE-only baseline (76.75%) by **+2.7 to +4.4 pp**. The best student is **FitNet Deep S2 at _T_ = 20 (81.11%)**, closing roughly **58%** of the teacher–student gap. All students share the same footprint (42.84 MB, **2.1× smaller** than the teacher) and remain **~5.5× faster** at inference (best student: 0.031 ms/image vs. 0.172 ms for the teacher).

**Effect of temperature.** Reading Table 1 row-by-row: Vanilla KD improves monotonically from _T_ = 4 to _T_ = 20 (80.67 → 80.74 → 80.92, +0.25 % total). FitNet Deep and Middle peak at _T_ = 20, with the largest single step from _T_ = 3 to _T_ = 8 for Deep (+0.40 %). FitNet Full is best at _T_ = 8 (80.76 %) and degrades at _T_ = 20 (80.62 %). AT + KD is flat across _T_ = 4–20 (79.45–79.52%), which is expected: the AT term acts on intermediate activations with sample-wise $L_2$ normalization, largely decoupling spatial alignment from logit softening. In all cases, gains beyond _T_ = 8 are modest or absent.

**Why FitNet Full and AT underperform: the capacity gap.** When distillation forces the student to replicate the teacher's internal geometry, either by matching attention maps at all four stages or by using hints through all layers, the student's limited representational budget is consumed imitating layer-wise structure rather than learning discriminative boundaries for 100 classes. This **over-regularizes** a network that already operates near its capacity ceiling: extra constraints compete with the classification objective instead of complementing it. By contrast, **FitNet Deep** aligns only the final backbone stage (where semantic abstraction is highest but the channel gap is still bridgeable via a 1×1 regressor), leaving earlier layers free to adapt to the task. **Vanilla KD** avoids intermediate matching altogether and transfers only softened class relations, which scales better across the teacher–student capacity gap. Middle and Deep FitNet sit between these extremes; Full and AT push feature-level fidelity beyond what a ResNet-18 can absorb, explaining why they never surpass vanilla KD despite their added training cost.

### 5.2 Answers to the initial hypotheses

1. **Vanilla KD vs. baseline: confirmed.** Logit-based distillation consistently exceeds CE-only training (+3.9 / +4.0 / +4.2 % at _T_ = 4 / 8 / 20).

2. **Feature-based methods vs. vanilla KD: partially confirmed.** FitNet Deep (_T_ = 20, 81.11%) is the only configuration that clearly beats vanilla KD (+0.19 %). Middle is competitive; **Full and AT do not justify their complexity** relative to simpler KD, because multi-layer feature/attention matching over-constrains the student.

3. **Role of hint layer and _T_: confirmed with nuance.** Hint depth matters more than brute-force multi-scale alignment. Temperature has a non-monotonic but generally modest effect: the largest swing is FitNet Deep _T_ = 3 → 8 (+0.40 %); _T_ = 8 → 20 adds at most +0.17 %. Model size and latency are essentially unchanged across methods; accuracy differences therefore reflect distillation design, not architecture size.

---

## 6. Conclusion

Logit-based and feature-based knowledge distillation were implemented and compared for compressing a fine-tuned ResNet-50 teacher into a ResNet-18 student on CIFAR-100. All distillation methods improved over the CE-only baseline (76.75%). **FitNet Deep Stage 2 with _T_ = 20** achieved the best student accuracy (**81.11%**), followed closely by vanilla KD (_T_ = 20, 80.92%) and FitNet Middle/Deep at _T_ = 8 (~80.95%). Attention Transfer did not outperform simpler logit-based KD.

The student retains a **2.1× smaller** model and **~5.5× lower** inference latency than the teacher, at the cost of ~1 % accuracy. Temperature sweeps show modest gains when moving from low _T_ (_T_ = 3 or 4) to _T_ = 8, and smaller improvements from _T_ = 8 to _T_ = 20 for vanilla KD and FitNet Deep/Middle; AT and FitNet Full do not benefit consistently from higher _T_.

---

## 7. Use of Artificial Intelligence

AI tools were used during this project for:

- **Boilerplate and scaffolding:** YAML config templates, SLURM cluster scripts, argument parsing.
- **Debugging and code review:** Identifying shape mismatches in feature extractors, verifying loss implementations against paper formulas.
- **Documentation:** Drafting sections of this report and inline code comments.

All architectural decisions, experiment design, hyperparameter choices, and interpretation of results are the author's responsibility. AI-generated code was reviewed, tested, and integrated manually before use in experiments.
