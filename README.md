# Cuckoo: Temporal Clean-Label Backdoor Attacks in Federated Learning

<img width="1369" height="1149" alt="image" src="https://github.com/user-attachments/assets/b9543b1e-1d1b-4619-8b42-af58376e7bbb" />


This repository contains the implementation of **Cuckoo**, a temporal
and adaptive clean-label backdoor attack framework for Federated
Learning.

<img width="1218" height="1291" alt="image" src="https://github.com/user-attachments/assets/7eec74fa-d478-47b0-a441-df17a6d47946" />


Cuckoo introduces a biologically inspired incubation and hatching
strategy. The malicious client first remains hidden by producing updates
similar to honest participants and later activates stronger backdoor
behaviour.

<img width="605" height="269" alt="image" src="https://github.com/user-attachments/assets/9a813a49-c85e-475c-b002-dfaf41a0c12c" />


## Overview

Federated Learning allows multiple clients to collaboratively train a
model without sharing raw data.

    Clients
       |
    Local Updates
       |
    Central Server
       |
    Global Model

Cuckoo changes the attack from a static poisoning strategy into a
temporal optimization problem.

## Attack Mechanism

### Incubation Phase

The attacker reduces poisoning strength:

\[ p\_{incubation}=0.1p \]

and minimizes the difference between malicious and honest updates:

\[ \|\|`\Delta`{=tex}*{cuckoo}-`\Delta`{=tex}*{honest}\|\|\_2
`\rightarrow 0`{=tex} \]

### Hatching Phase

After the activation round:

\[ t=t_h \]

the attacker increases poisoning strength and optimizes the backdoor
objective.

## Clean-label Backdoor

Cuckoo preserves labels:

\[ y\_{poison}=y\_{original} \]

Only the learned representation is modified.

## Framework Extensions

Cuckoo extends federated learning with:

-   Cuckoo phase controller
-   incubation poisoning
-   hatching activation
-   adaptive poison scheduling
-   trigger generation
-   defence evaluation pipeline

Supported defences:

-   Robust Learning Rate
-   Krum
-   Coordinate Median
-   Gradient clipping
-   Noise injection

## Experiments

### Attack Scenarios

  ID   Scenario
  ---- --------------------------
  S1   Client scalability
  S2   Multi-attacker collusion
  S3   Low poison fraction
  S4   Delayed hatching
  S5   Non-IID federation

### Neural Architectures

  Model
  ----------
  CNN
  Deep CNN
  ResNet
  CERN-Net
  LeNet-5
  MLP

### Dataset Evaluation

  Dataset         Domain
  --------------- ---------------------
  MNIST           Digits
  Fashion-MNIST   Clothing
  KMNIST          Japanese characters

## Example Usage

``` bash
python federated.py --data=fmnist --num_agents=10 --rounds=300 --num_corrupt=1 --poison_frac=0.5 --clean_label=1 --cuckoo=1 --cuckoo_start_round=100
```

## Citation

``` bibtex
@mastersthesis{2026cuckoo,
title={Cuckoo: Temporal Clean-Label Backdoor Attacks in Federated Learning},
author={},
year={2026}
}
```

## License

MIT License
