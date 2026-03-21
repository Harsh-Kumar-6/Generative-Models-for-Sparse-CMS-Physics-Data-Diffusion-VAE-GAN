# 🚀 CMS Detector Simulation using Generative Models

[![CERN-CMS](https://img.shields.io/badge/Experiment-CMS-blue)](https://cms.cern/) 
[![Python-3.9+](https://img.shields.io/badge/Python-3.9%2B-green)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/Framework-PyTorch-orange)](https://pytorch.org/)
# 🚀 CMS Detector Simulation using Generative Models

This repository presents a **Proof-of-Concept (PoC)** for simulating low-level CMS detector data using modern generative models. The primary focus is on **Denoising Diffusion Probabilistic Models (DDPMs)**, with comparative analysis against **Variational Autoencoders (VAEs)** and **Generative Adversarial Networks (GANs)**.

The dataset consists of high-dimensional sparse tensors of shape **(8 × 128 × 128)** representing energy deposits across detector layers.

---

## 🎯 Objective

To generate synthetic detector data that preserves:

- Spatial structure of particle showers  
- Per-channel energy distributions  
- Sparse high-energy event patterns  

---

## 📊 Summary of Results

| Model | Avg. Wasserstein Distance | Stability | Key Behavior |
|-------|--------------------------|----------|-------------|
| **Diffusion (DDPM)** | **0.18** | High | Best distribution match; slight underestimation of extreme energy peaks |
| **GAN** | 0.55 | Low | Mode collapse; repetitive artifacts |
| **VAE** | 0.85 | Medium | Blurred outputs; sensitive to KL weighting |

---

## 🧠 Key Findings

### 1. Diffusion Models: Robust but Computationally Expensive

The diffusion model successfully captures:

- Spatial structure of energy deposits  
- Sparse activation patterns  
- Channel-wise distributions  

This is attributed to its **iterative denoising process**, which allows gradual refinement of samples.

**Limitations:**
- Slight truncation of high-energy tails  
- High sampling cost due to multi-step inference  

---

### 2. VAE: Sensitivity to KL Regularization

The VAE exhibited strong dependence on KL weighting:

- **High β → Posterior Collapse**: latent variables ignored, outputs converge to near-zero  
- **Low β → Latent Explosion**: unbounded latent space, poor sampling consistency  
- **Balanced β → Partial recovery**: produces smoother but **blurred and energy-averaged outputs**

This demonstrates **complexity bias**, where the model prefers simpler (low-energy) reconstructions over rare high-energy events.

---

### 3. GAN: Mode Collapse and Instability

The GAN struggled to model the full distribution:

- Generated repetitive patterns (mode collapse)  
- Failed to capture diversity of particle showers  
- Training instability despite reasonable loss values  

This highlights limitations of adversarial training on **sparse, high-dimensional physics data**.

---

## 🧪 Statistical Evaluation Framework

To quantitatively compare models, the following metrics were used:

### 1. Per-Channel Wasserstein Distance (WD)
Measures the distance between real and generated distributions:
- Lower WD → better physical fidelity  

### 2. Log-Scale Energy Histograms
Used to verify:
- Presence of long-tail high-energy events  
- Distribution alignment across magnitudes  

### 3. Spatial Occupancy Heatmaps
Visual validation of:
- Energy localization  
- Detector geometry alignment  

### 4. (Optional) Maximum Mean Discrepancy (MMD)
Evaluates global distribution similarity in high-dimensional space.

---

## ⚔️ Comparative Insight

| Property | Diffusion | VAE | GAN |
|----------|----------|-----|-----|
| Handles sparsity | ✅ | ❌ | ⚠️ |
| Captures rare events | ✅ | ❌ | ❌ |
| Training stability | ✅ | ⚠️ | ❌ |
| Sampling speed | ❌ | ✅ | ✅ |

---
---

## 🛠️ Repository Structure

```text
├── models/             # Pre-trained weights (.pth)
├── notebooks/          # Step-by-step experiment logs
│   ├── 1_diffusion.ipynb
│   ├── 2_vae.ipynb
│   └── 3_gan.ipynb
├── src/                # Core implementation
│   ├── arch_ddpm.py    # UNet and Denoising logic
│   ├── baselines.py    # VAE & GAN architectures
│   └── utils.py        # H5 Data loaders & Plotting
├── requirements.txt
└── README.md


git clone [https://github.com/your-username/cms-diffusion-simulation.git](https://github.com/your-username/cms-diffusion-simulation.git)
cd cms-diffusion-simulation
pip install -r requirements.txt

---

## 📝 Technical Note on Data Scaling

Due to computational constraints (T4 GPU), this Proof-of-Concept was trained on a 10% stratified subset (**6,000 samples**) for 8 epochs.  

- This allowed rapid prototyping of the comparative framework.  
- Diffusion required more sampling steps but gave far superior fidelity for sparse CMS detector data.

**Future Work:**
- Transition to a weighted loss function to better capture high-energy peaks  
- Scale to the full 60,000 sample dataset

---

## ⚙️ Installation & Usage

```bash
git clone https://github.com/your-username/cms-diffusion-simulation.git
cd cms-diffusion-simulation
pip install -r requirements.txt
