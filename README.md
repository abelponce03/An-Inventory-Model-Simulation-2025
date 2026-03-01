# 📦 Inventory Model Simulation — (s, S) Policy Analysis

![Language](https://img.shields.io/badge/Language-Python-3776AB?logo=python&logoColor=white)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

A **discrete-event simulation** of *(s, S)* inventory replenishment policies in Python. Simulates a store inventory system with Poisson customer arrivals, configurable demand distributions, and stochastic order lead times. Includes **bootstrap statistical analysis** for comparing policies with confidence intervals.

---

## 📑 Table of Contents

- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [How to Build & Run](#how-to-build--run)
- [How It Works](#how-it-works)
- [Academic Context](#academic-context)

---

## ✨ Features

- **(s, S) Inventory Policy Simulation** — Models when to reorder (stock ≤ s) and how much (up to S)
- **Poisson Customer Arrivals** — Stochastic arrival process for realistic demand modeling
- **Configurable Demand Distributions** — Flexible demand per customer via various probability distributions
- **Stochastic Lead Times** — Random delivery delays for replenishment orders
- **Bootstrap Analysis** — Non-parametric statistical method for constructing confidence intervals
- **Policy Comparison** — Evaluates and compares different (s, S) parameter combinations
- **Cost Metrics** — Tracks holding costs, shortage costs, ordering costs, and total cost

---

## 🛠 Tech Stack

| Component       | Technology             |
|-----------------|------------------------|
| Language         | Python 3               |
| Simulation       | Custom discrete-event  |
| Statistics       | Bootstrap resampling   |
| Libraries        | NumPy, SciPy           |

---

## 📁 Project Structure

```
An-Inventory-Model-Simulation-2025/
├── Inventory_Model.py     # Core discrete-event simulation engine
├── bootstrap.py           # Bootstrap statistical analysis
├── common_functions.py    # Shared utility functions and distributions
└── s_valor.py             # (s, S) policy parameter evaluation
```

---

## 🚀 How to Build & Run

### Prerequisites

- Python 3.8+
- NumPy, SciPy

### Install Dependencies

```bash
pip install numpy scipy
```

### Run the Simulation

```bash
python Inventory_Model.py
```

### Run Bootstrap Analysis

```bash
python bootstrap.py
```

---

## ⚙️ How It Works

### Simulation

The system models a single-product inventory with the *(s, S)* policy:
- When inventory drops to or below **s** (reorder point), an order is placed to bring inventory up to **S** (order-up-to level).
- Customers arrive according to a **Poisson process**, each demanding a random quantity.
- Orders have **stochastic lead times** before delivery.
- The simulation tracks **holding costs**, **shortage costs**, and **ordering costs** over time.

### Statistical Analysis

- **Bootstrap resampling** is used to estimate confidence intervals for cost metrics without assuming a specific distribution.
- Multiple (s, S) parameter pairs are compared to identify the optimal inventory policy.

---

## 🎓 Academic Context

> **Course Project** — Simulation, University of Havana, 2025
