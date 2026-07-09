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

During the incubation phase, the attacker remains hidden inside the federated
training process. Instead of immediately applying the full backdoor objective,
Cuckoo reduces the poisoning intensity to avoid statistical detection.

The poisoning rate is controlled by:

$$
p_{incubation}=0.1p
$$

where \(p\) represents the normal poisoning fraction used during the attack.

The malicious client additionally optimizes its update direction to resemble
honest client behaviour. The objective is to minimize the distance between the
malicious update and the aggregated honest update:

$$
||\Delta_{cuckoo}-\Delta_{honest}||_2 \rightarrow 0
$$

Therefore, during incubation:

- the malicious update has similar magnitude and direction to honest updates,
- distance-based aggregation methods have difficulty identifying the attacker,
- the backdoor remains dormant while the global model learns normally.

---

### Hatching Phase

After the predefined activation round:

$$
t=t_h
$$

the Cuckoo attacker transitions from a stealth phase into an active backdoor
injection phase.

The poisoning strength is increased:

$$
p_{hatch}=p
$$

and the attacker optimizes the model update toward the backdoor objective:

$$
\min_{\Delta_c}
L_{clean}(w+\Delta_c)
+
\lambda L_{backdoor}(w+\Delta_c)
$$

where:

- \(L_{clean}\) maintains normal classification performance,
- \(L_{backdoor}\) increases attack success rate (ASR),
- \(\lambda\) controls the trade-off between stealth and attack strength.

This temporal separation allows Cuckoo to remain hidden during early training
while activating the backdoor after sufficient model adaptation.

---

## Clean-label Backdoor

Unlike traditional dirty-label attacks, Cuckoo does not modify the training
labels.

The clean-label constraint is:

$$
y_{poison}=y_{original}
$$

The attacker only modifies the input representation by adding a trigger pattern:

$$
x_{poison}=x+\delta_{trigger}
$$

while preserving the original semantic label.

Therefore:

- label inspection cannot detect poisoned samples,
- the attacker behaves similarly to an honest participant,
- the learned decision boundary is modified rather than the training labels.

The objective is to create a hidden feature association:

$$
f(x+\delta_{trigger}) \rightarrow y_{target}
$$

while maintaining:

$$
Acc_{clean}\approx Acc_{normal}
$$


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
