
# 📚 **HEIST**

This project implements a distributed training pipeline for learning representations over spatial cell graphs using PyTorch, PyTorch Geometric, and Distributed Data Parallel (DDP).

---


## Create and Activate Conda Environment

```bash
conda env create -f environment.yml
conda activate HEIST
```
---

## 📂 **Data Preparation**

* Place your dataset in the `data/pretraining/` directory.
* The data should be preprocessed and saved as PyTorch objects using `torch.save`.

---

## ⚙️ **Training the Model**

Run the the full HEIST model on all the available GPUs, use the following command to start distributed training:

```bash
python main_ddp.py --data_dir data/pretraining/ --pe --cross_message_passing
```

If you want to train only on single GPU, change `main_ddp` with `main`:

```bash
python main.py --data_dir data/pretraining/ --pe --cross_message_passing
```

### 📈 **Weights & Biases (wandb) Logging**

To visualize training metrics (loss curves, validation loss, etc.) in real-time:

1. Install wandb: `pip install wandb`
2. Login (first time only): `wandb login`
3. Add `--wandb` flag when training:

```bash
python main.py --data_dir data/pretraining/ --pe --cross_message_passing --blending --wandb
```

Optional wandb arguments:
- `--wandb_project HEIST` - Project name (default: HEIST)
- `--wandb_entity your_username` - Your W&B username/team

### 💡 **Important Arguments:**

| Argument       | Description                 | Default           |
| -------------- | --------------------------- | ----------------- |
| `--data_dir`   | Path to preprocessed data   | data/pretraining/ |
| `--pe_dim`     | Positional Encoding Dim     | 128               |
| `--init_dim`   | Initial MLP Hidden Dim      | 128               |
| `--hidden_dim` | Hidden Dimension            | 128               |
| `--output_dim` | Output Dimension            | 128               |
| `--num_layers` | Number of MLP Layers        | 10                |
| `--num_heads`  | Number of Transformer Heads | 8                 |
| `--batch_size` | Batch Size                  | 128                |
| `--lr`         | Learning Rate               | 1e-3              |
| `--wd`         | Weight Decay                | 3e-3              |
| `--num_epochs` | Number of Training Epochs   | 20                |
| `--wandb`      | Enable W&B logging          | False             |

---

## 📈 **Model Checkpoints**

* Checkpoints are saved automatically under the `saved_models/` directory.
* The best model is saved as `HEIST.pth`.

---

## 🛠 **Resuming Training**

If you want to resume from a saved checkpoint, ensure that the model and optimizer state dictionaries are correctly loaded in the script.

---


## 📊 **Model Evaluation**

After training, you can evaluate the model using the provided evaluation script.

### ✅ **Run the Evaluation**

```bash
bash eval.sh
```

This script will:

1. Initialize and activate the Conda environment `HEIST`.
2. Run a series of evaluations across multiple datasets and tasks:

   * **Representation Space Calculation**

     * `dfci`, `upmc`, `charville`, `sea`, `melanoma`, `placenta`, `lung`
   * **Tissue Classification**

     * Predict primary outcomes and recurrence for clinical datasets.
   * **Melanoma and Cell Clustering Evaluations**
   * **Placenta Dataset Analysis**
   * **Gene Imputation Tasks**

     * Both standard and fine-tuned versions.

---

### 📁 **Generated Outputs**

* Evaluation results will be stored in the corresponding directories or logged to the console.
* Ensure that the trained model checkpoint is available and named correctly (default: `HEIST`).
