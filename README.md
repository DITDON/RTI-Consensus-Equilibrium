# Consensus Equilibrium with Deep Denoisers for Radio Tomographic Imaging

A research project on improving **Radio Tomographic Imaging (RTI)** using deep denoising and **Consensus Equilibrium (CE)** for robust spatial reconstruction.

> **Project Status:** 🚧 Under Development  
> **Current Stage:** RTI simulation + Tikhonov baseline reconstruction

---

## 📌 Overview

Radio Tomographic Imaging (RTI) is an imaging technique that reconstructs the location of objects or signal-attenuating regions using changes in **Received Signal Strength (RSS)** between wireless sensor nodes.

The reconstruction problem can be represented as an inverse problem:

$$
y = Wx + n
$$

where:

- $y$ = measured RSS-related observation vector
- $W$ = RTI weight matrix
- $x$ = spatial loss field (SLF)
- $n$ = measurement noise

RTI is an ill-posed inverse problem because the number and quality of measurements are limited and the measurements can be affected by noise and other uncertainties.

This project investigates a reconstruction framework that combines RTI data fidelity with learned image denoising and Consensus Equilibrium.

---

## 🎯 Objectives

The main objectives of this project are:

- Build a reproducible RTI simulation environment.
- Model wireless sensor links and their spatial influence.
- Generate synthetic RSS/measurement observations.
- Reconstruct the spatial loss field from noisy measurements.
- Establish traditional reconstruction methods as baselines.
- Develop a deep-learning-based denoising prior.
- Integrate the denoiser with Consensus Equilibrium.
- Evaluate reconstruction quality under different noise conditions.
- Compare reconstruction performance using quantitative metrics.

---

## 🧠 Current Pipeline

The current implementation follows:

```text
Sensor Network
      ↓
RTI Weight Matrix (W)
      ↓
Ground Truth Spatial Loss Field
      ↓
Measurement Generation
      ↓
Add Measurement Noise
      ↓
Tikhonov Reconstruction
      ↓
Reconstructed Image
      ↓
RMSE / PSNR / SSIM Evaluation
