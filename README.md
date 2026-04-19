# Self-Pruning Neural Network
### Tredence AI Engineer Intern Task
 
**Submitted by:**
Ishan Goel (RA2311003010156)
SRM Institute of Science and Technology, Kattankulathur, Tamil Nadu
3rd Year
 
---
 
## 📌 Overview
 
This project implements a self-pruning neural network that dynamically learns which of its connections are unnecessary during training.
 
Instead of pruning weights after training, this model:
- Learns gating parameters for each weight
- Suppresses unimportant connections during training itself
- Produces a sparse and efficient model
The implementation is based on a custom PyTorch layer and a modified loss function with sparsity regularization.
 
---
 
## 🧠 Key Idea
 
Each weight in the network is paired with a learnable gate:
 
```
Effective Weight = Weight × Sigmoid(Gate Score)
```
 
- If gate → 0 → connection is **pruned**
- If gate → 1 → connection is **active**
To encourage pruning, we add an L1 penalty on gate values:
 
```
Total Loss = CrossEntropyLoss + λ × SparsityLoss
```
 
---
 
## 🏗️ Project Structure
 
```
├── prunable_net.py              # Main training script
├── report.md                    # Short analysis report
├── results_lambda_*.json        # Results for different λ values
├── gate_distributions.png       # Gate value distribution plot
├── tradeoff_summary.png         # Accuracy vs sparsity plot
└── .gitignore
```
 
---
 
## ⚠️ Important Note (Dataset)
 
The CIFAR-10 dataset is **NOT included** in this repository due to GitHub file size limitations.
 
However, the script automatically downloads it using:
 
```python
torchvision.datasets.CIFAR10(download=True)
```
 
So you do **not** need to manually download anything.
 
---
 
## 🚀 How to Run
 
### 1. Install Dependencies
 
```bash
pip install torch torchvision matplotlib numpy
```
 
### 2. Run the Training Script
 
```bash
python prunable_net.py
```
 
### 3. Optional: Custom Run
 
```bash
python prunable_net.py --epochs 10 --batch-size 128 --lambdas 0.0001 0.001 0.01
```
 
---
 
## 📊 Outputs
 
After training, the following are generated:
 
### 1. Results Table (JSON)
- Accuracy for each λ
- Sparsity level
### 2. Plots
 
**`gate_distributions.png`**
→ Shows how many gates went near zero
 
**`tradeoff_summary.png`**
→ Shows sparsity vs accuracy trade-off
 
---
 
## 📈 Evaluation Metrics
 
**✔ Test Accuracy**
Measures classification performance on CIFAR-10
 
**✔ Sparsity Level**
 
Defined as:
```
% of gates < 1e-2
```
Higher sparsity = more pruning
 
---
 
## ⚖️ Observations
 
- **Low λ** → Higher accuracy, low sparsity
- **High λ** → Lower accuracy, high sparsity
- The model successfully learns to prune itself dynamically
---
 
## 🧩 What is Implemented
 
- ✔ Custom `PrunableLinear` layer
- ✔ Learnable gate mechanism
- ✔ Sparsity regularization (L1 penalty)
- ✔ Custom training loop
- ✔ Lambda sweep experiments
- ✔ Visualization of gate distributions
## ❌ What is NOT Included
 
- CIFAR-10 dataset (auto-downloaded)
- Model checkpoints (`.pt` files) due to size constraints
---
 
## 💡 Future Improvements
 
- Replace MLP with CNN for better accuracy
- Structured pruning (neurons instead of weights)
- Hard threshold pruning after training
- Deployment on edge devices
---
 
## 🏁 Conclusion
 
This project demonstrates that neural networks can:
- Learn their own structure
- Remove unnecessary connections
- Achieve a balance between efficiency and accuracy
