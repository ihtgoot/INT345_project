
* **name** : Restoration of the distorted frames (due to atmospheric conditions such as heat) in the captured video
* **outcome** : A system for restoring distorted video frames caused by atmospheric conditions like heat, utilizing advanced image processing and machine learning techniques to enhance visual clarity and quality.
* **objective** : Restore distorted video frames caused by heat and atmospheric conditions using machine learning and image processing 
* **dataset** : https://www.kaggle.com/datasets/animeshmahajan/thermal-image-dataset



| PAPER                | how they simulate distortion                            | what pipeline order they use (alignment before or after restoration?) | what metrics they report | approch |
| -------------------- | ------------------------------------------------------- | --------------------------------------------------------------------- | ------------------------ | ------- |
| [[2207.10040v2.pdf]] | finding data to tain is hard , so simulated [[p1_data]] |                                                                       |                          |         |
|                      |                                                         |                                                                       |                          |         |
|                      |                                                         |                                                                       |                          |         |
|                      |                                                         |                                                                       |                          |         |
|                      |                                                         |                                                                       |                          |         |
|                      |                                                         |                                                                       |                          |         |
|                      |                                                         |                                                                       |                          |         |







# 0. First: understand the core difficulty (from literature)

> Turbulence = **coupled distortion (geometry + blur + noise + time variation)**

From Single Frame Atmospheric Turbulence Mitigation: A Benchmark Study and A New Physics-Inspired Transformer Model:
- It’s not separable into independent problems
- CNNs fail because distortion is **spatially dynamic** ([arXiv](https://arxiv.org/abs/2207.10040?utm_source=chatgpt.com "Single Frame Atmospheric Turbulence Mitigation: A Benchmark Study and A New Physics-Inspired Transformer Model"))    

From diffusion paper:
- distortion = **geometric + spatially varying blur simultaneously** ([arXiv](https://arxiv.org/abs/2305.05077?utm_source=chatgpt.com "Atmospheric Turbulence Correction via Variational Deep ..."))
If you don’t internalize this, your model design will be wrong.

---

# 1. Reading path (don’t jump randomly)

## Stage 1 — Understand the physics + classical methods

### Paper 1 (START HERE)

- Restoration of Atmospheric Turbulence-distorted Images via RPCA and Quasiconformal Maps    
### What to extract:

- Why they use **low-rank decomposition**
- Concept of **reference frame construction**
- Separation:
    - stable background vs distortion

👉 This builds your intuition:

> turbulence = small random displacement over time

---

### Paper 2

- Atmospheric turbulence mitigation using optical flow and CNN
### Key idea:

- First **align frames (optical flow)**
- Then **restore blur**

Pipeline:

```
distorted frames → alignment → deblurring → output
```

👉 This is critical:

> geometry correction comes BEFORE restoration

---

# 2. Stage 2 — Deep learning approaches (non-transformer)

### Paper 3

- Blind Restoration of Atmospheric Turbulence-Degraded Images NSRN
### What matters:
- They treat:
    - noise + blur as **coupled**
- Use:
    - multi-scale attention
    - curriculum learning

👉 Insight
> training strategy matters as much as architecture ([MDPI](https://www.mdpi.com/2072-4292/14/19/4797?utm_source=chatgpt.com "Blind Restoration of Atmospheric Turbulence-Degraded ..."))

---

### Paper 4 (important conceptual shift)

- Atmospheric turbulence removal with complex-valued convolution
    
Key idea:
- Use **phase information**
- Separate:
    - geometric distortion
    - fine detail reconstruction ([ScienceDirect](https://www.sciencedirect.com/science/article/pii/S0167865523001447?utm_source=chatgpt.com "Atmospheric turbulence removal with complex-valued ..."))
👉 Insight:
> distortion ≠ just pixel intensity → it affects phase structure

---

# 3. Stage 3 — Transformer-based methods (your target)

Now you’re allowed to look at transformers.

---

### Paper 5 (CORE for your project)

- Imaging through the Atmosphere using Turbulence Mitigation Transformer
### What it actually does:
- Multi-frame transformer
- Introduces:
    - **Temporal-Channel Joint Attention**
- Learns:
    - which frames are less distorted
👉 Critical line
- trained entirely on **synthetic data** ([arXiv](https://arxiv.org/pdf/2207.06465?utm_source=chatgpt.com "Imaging through the Atmosphere using Turbulence ..."))
    

This should hit you:

> your synthetic generator is NOT optional — it's central

---

### Paper 6 (newer transformer system)

- Restoration of Atmospheric Turbulence-Degraded Images using Vision Transformer ATRN    
Key:
- Two-stage:
    1. transformer restoration
    2. multi-frame fusion ([PubMed](https://pubmed.ncbi.nlm.nih.gov/39876226/?utm_source=chatgpt.com "Restoration of atmospheric turbulence-degraded images ..."))

👉 Insight
> even transformers alone are not enough → need fusion

---

### Paper 7 (modern SOTA direction)

- DeTurb: Atmospheric Turbulence Mitigation with Deformable 3D Convolutions and 3D Swin Transformers    
Key idea:
- Combine:
    - geometric alignment (3D conv)
    - transformer restoration ([arXiv](https://arxiv.org/abs/2407.20855?utm_source=chatgpt.com "DeTurb: Atmospheric Turbulence Mitigation with Deformable 3D Convolutions and 3D Swin Transformers"))

👉 Insight
> hybrid models win, not pure transformers

---

# 4. Stage 4 — Alternative paradigm (optional but powerful)

### Paper 8

- Atmospheric Turbulence Correction via Variational Deep Diffusion    
Key:
- uses **diffusion models**
- conditioned on degradation process
👉 Insight
> future direction = generative restoration, not deterministic

---

# 5. The meta-paper (read this last)

### Review paper
- Deep Learning Techniques for Atmospheric Turbulence Removal A Review
This gives:
- datasets
- metrics
- comparison of:
    - CNN vs Transformer vs diffusion ([ResearchGate](https://www.researchgate.net/publication/388386749_Deep_learning_techniques_for_atmospheric_turbulence_removal_a_review?utm_source=chatgpt.com "Deep learning techniques for atmospheric turbulence ..."))

---

# 6. What you should extract (non-negotiable)

Don’t read like a student. Extract:

## A. Degradation model

- How turbulence is simulated
    
- Zernike polynomials (mentioned in classical papers)
    

## B. Pipeline structure

You will see this pattern everywhere:

```
alignment → restoration → refinement
```

## C. Why transformers help

- handle **non-local distortions**
    
- capture **temporal info**
    

## D. Why they still fail

From TMT:

- poor generalization to real data ([xg416.github.io](https://xg416.github.io/TMT/?utm_source=chatgpt.com "Turbulence Mitigation Transformer"))
    
## Minimal correct design (based on papers)

### Your system should look like:

```
Synthetic turbulence generator
        ↓
Distorted frames (input)
        ↓
(Optional) alignment module
        ↓
Transformer (Restormer / SwinIR)
        ↓
Restored frame
```

---

## If you want to impress evaluator

Add ONE of:
- multi-frame input
- temporal consistency loss
- distortion simulation based on physics    

---

# 8. Brutal reality check

If you
- skip synthetic data → project fails
- skip distortion modeling → model learns garbage
- skip evaluation → looks fake
---

# 9. Your task (don’t dodge this)

Now you respond with:

1. Explain:
    - how turbulence distortion differs from Gaussian blur
2. Describe:
    - a synthetic distortion pipeline (step by step)
3. Decide:
    - single-frame OR multi-frame (and justify)
