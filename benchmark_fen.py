# ============================================================
# FEN vs Baselines Benchmarking on UBFC-rPPG Dataset
# Strictly parameter-matched sequence-to-sequence training
# ============================================================

import os
import time
import random
import shutil
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import matplotlib.pyplot as plt

# Try importing kagglehub for automated dataset download
try:
    import kagglehub
    KAGGLEHUB_AVAILABLE = True
except ImportError:
    KAGGLEHUB_AVAILABLE = False

# ============================================================
# 1. CONFIGURATION BLOCK
# ============================================================
RUN_MODE = "all_quick"
# Options:
#   "all_quick"       -> Runs all 5 models sequentially and prints a summary
#   "rnn"             -> Vanilla RNN baseline
#   "lstm"            -> Vanilla LSTM baseline
#   "rnn_residual"    -> RNN with temporal residual connections
#   "lstm_residual"   -> LSTM with temporal residual connections
#   "fen"             -> Feature-Escrow Network (FEN wrapping LSTM core)

NUM_EPOCHS = 10
BATCH_SIZE = 256
SEQ_LEN = 128
LR = 0.001
TARGET_PARAMS = 105000  # Strictly matched parameter budget
AUTO_MATCH_PARAMS = True

# Masking configuration (Curing the notebook's batch slicing bug)
MASK_EARLY_FRAMES = 60  # Ignore first 60 frames during loss and evaluation

# ============================================================
# 2. DEVICE & SEEDING
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.benchmark = True

seed_everything(42)

# ============================================================
# 3. DATASET PREPARATION & LOADING
# ============================================================
data_path = os.path.join('data', 'UBFC-RPPG-Dataset')

# Auto-download dataset if missing
if not os.path.exists(data_path):
    print(f"Dataset path '{data_path}' not found.")
    if KAGGLEHUB_AVAILABLE:
        print("Downloading UBFC-rPPG dataset via kagglehub...")
        try:
            download_path = kagglehub.dataset_download("malekdinarito/ubfc-rppg-dataset")
            os.makedirs('data', exist_ok=True)
            # Move the downloaded contents to data/UBFC-RPPG-Dataset
            shutil.move(download_path, data_path)
            print("Download and setup complete!")
        except Exception as e:
            print(f"Failed to auto-download dataset: {e}")
            print("Please download it manually or run prepare_datasets.py")
            exit(1)
    else:
        print("Please install 'kagglehub' or manually download the dataset to data/UBFC-RPPG-Dataset")
        exit(1)

subjects = os.listdir(data_path)

class UBFC_Dataset(Dataset):
    def __init__(self, data_path, subjects, seq_len=128):
        self.data_path = data_path 
        self.subjects = subjects 
        self.seq_len = seq_len
        self.possible_ranges = []
        for subject in subjects:
            signal_path = os.path.join(data_path, subject, 'ground_truth.txt')
            # Load signal to determine shape
            signal = np.loadtxt(signal_path)
            # Ground truth shape is [3, length]
            num_starts = signal.shape[-1] - seq_len 
            for i in range(num_starts): 
                self.possible_ranges.append((subject, i)) 

    def __len__(self):
        return len(self.possible_ranges)

    def __getitem__(self, index):
        subject, i = self.possible_ranges[index]
        signal_path = os.path.join(self.data_path, subject, 'ground_truth.txt')
        colors_path = os.path.join(self.data_path, subject, 'roi_colors.txt')
        
        signals = np.loadtxt(signal_path)
        colors = np.loadtxt(colors_path, delimiter=',')
        
        # Take Normalized Blood Volume Pulse (first row)
        signal_seq = signals[0, i : i + self.seq_len]
        color_seq = colors[i : i + self.seq_len]

        # Standard range normalization to avoid color saturation
        min_vals = color_seq.min(axis=0, keepdims=True)
        max_vals = color_seq.max(axis=0, keepdims=True)
        color_seq = (color_seq - min_vals) / (max_vals - min_vals + 1e-6)

        return torch.tensor(color_seq, dtype=torch.float32), torch.tensor(signal_seq, dtype=torch.float32)

# Instantiate loaders
dataset = UBFC_Dataset(data_path, subjects, seq_len=SEQ_LEN)
train_size = int(0.8 * len(dataset))
test_size = len(dataset) - train_size 
train_dataset, test_dataset = random_split(dataset, [train_size, test_size], generator=torch.Generator().manual_seed(42))

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, pin_memory=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, pin_memory=True)

# ============================================================
# 4. MODELS DEFINITION
# ============================================================

def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

# --- 4.1 Vanilla RNN Baseline ---
class RNNBaseline(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, return_stats=False):
        out, _ = self.rnn(x)
        logits = self.fc(out)
        if return_stats:
            # Measure L2 norm across the sequence
            return logits, {"active_norm": out.norm(dim=-1).mean().item()}
        return logits

# --- 4.2 Vanilla LSTM Baseline ---
class LSTMBaseline(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, return_stats=False):
        out, _ = self.lstm(x)
        logits = self.fc(out)
        if return_stats:
            return logits, {"active_norm": out.norm(dim=-1).mean().item()}
        return logits

# --- 4.3 Residual RNN Baseline ---
class ResidualRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cells = nn.ModuleList([
            nn.RNNCell(input_size if l == 0 else hidden_size, hidden_size) 
            for l in range(num_layers)
        ])
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, return_stats=False):
        B, seq_len, _ = x.shape
        h = [torch.zeros(B, self.hidden_size, device=x.device) for _ in range(self.num_layers)]
        outputs = []
        
        for t in range(seq_len):
            xt = x[:, t, :]
            h_next = []
            
            # Layer 0
            h0_n = self.cells[0](xt, h[0])
            h0 = h0_n + h[0]  # Temporal residual
            h_next.append(h0)
            
            # Deeper layers
            for l in range(1, self.num_layers):
                hl_n = self.cells[l](h_next[l-1], h[l])
                hl = hl_n + h[l]  # Temporal residual
                h_next.append(hl)
                
            h = h_next
            outputs.append(h[-1].unsqueeze(1))
            
        outputs = torch.cat(outputs, dim=1)
        logits = self.fc(outputs)
        if return_stats:
            return logits, {"active_norm": outputs.norm(dim=-1).mean().item()}
        return logits

# --- 4.4 Residual LSTM Baseline ---
class ResidualLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers=2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.cells = nn.ModuleList([
            nn.LSTMCell(input_size if l == 0 else hidden_size, hidden_size) 
            for l in range(num_layers)
        ])
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x, return_stats=False):
        B, seq_len, _ = x.shape
        h = [torch.zeros(B, self.hidden_size, device=x.device) for _ in range(self.num_layers)]
        c = [torch.zeros(B, self.hidden_size, device=x.device) for _ in range(self.num_layers)]
        outputs = []
        
        for t in range(seq_len):
            xt = x[:, t, :]
            h_next, c_next = [], []
            
            # Layer 0
            h0_n, c0_n = self.cells[0](xt, (h[0], c[0]))
            h0 = h0_n + h[0]  # Temporal residual
            h_next.append(h0)
            c_next.append(c0_n)
            
            # Deeper layers
            for l in range(1, self.num_layers):
                hl_n, cl_n = self.cells[l](h_next[l-1], (h[l], c[l]))
                hl = hl_n + h[l]  # Temporal residual
                h_next.append(hl)
                c_next.append(cl_n)
                
            h = h_next
            c = c_next
            outputs.append(h[-1].unsqueeze(1))
            
        outputs = torch.cat(outputs, dim=1)
        logits = self.fc(outputs)
        if return_stats:
            return logits, {"active_norm": outputs.norm(dim=-1).mean().item()}
        return logits

# --- 4.5 Feature-Escrow Network (FEN Wrapping LSTM Active Stream) ---
class FeatureEscrowLSTM(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.lstm_cell = nn.LSTMCell(input_size, hidden_size)
        self.gate = nn.Linear(hidden_size, hidden_size)
        self.escrow_proj = nn.Linear(hidden_size, hidden_size)
        self.fc = nn.Linear(hidden_size * 2, 1)

    def forward(self, x, return_stats=False):
        B, seq_len, _ = x.shape
        h = torch.zeros(B, self.hidden_size, device=x.device)
        c = torch.zeros(B, self.hidden_size, device=x.device)
        E = torch.zeros(B, self.hidden_size, device=x.device)
        
        outputs = []
        gate_means = []
        
        for t in range(seq_len):
            xt = x[:, t, :]
            # 1. Active Transformation (LSTM Core)
            h_raw, c = self.lstm_cell(xt, (h, c))
            
            # 2. Escrow Gate evaluation
            g = torch.sigmoid(self.gate(h_raw))
            D = g * h_raw
            
            # 3. Subtractive Routing / Depletion
            h = h_raw - D
            
            # 4. Secure Archiving (Escrow)
            E = E + self.escrow_proj(D)
            
            # 5. Readout Synthesis at step t
            combined = torch.cat([h, E], dim=-1)
            outputs.append(combined.unsqueeze(1))
            
            if return_stats:
                gate_means.append(g.mean().detach())
                
        outputs = torch.cat(outputs, dim=1)
        logits = self.fc(outputs)
        if return_stats:
            stats = {
                "active_norm": h.norm(dim=-1).mean().item(),
                "gate_mean": sum(gate_means)/len(gate_means)
            }
            return logits, stats
        return logits

# ============================================================
# 5. MODEL GENERATOR & AUTO PARAMETER MATCHING
# ============================================================
def build_model(mode, hidden_dim):
    if mode == "rnn":
        return RNNBaseline(input_size=9, hidden_size=hidden_dim, num_layers=2)
    elif mode == "lstm":
        return LSTMBaseline(input_size=9, hidden_size=hidden_dim, num_layers=2)
    elif mode == "rnn_residual":
        return ResidualRNN(input_size=9, hidden_size=hidden_dim, num_layers=2)
    elif mode == "lstm_residual":
        return ResidualLSTM(input_size=9, hidden_size=hidden_dim, num_layers=2)
    elif mode == "fen":
        return FeatureEscrowLSTM(input_size=9, hidden_size=hidden_dim)
    else:
        raise ValueError(f"Unknown mode: {mode}")

def choose_hidden_dim(mode):
    if not AUTO_MATCH_PARAMS:
        return 64
    
    best_h = 8
    best_diff = float('inf')
    
    for h in range(8, 256):
        model = build_model(mode, h)
        params = count_params(model)
        diff = abs(params - TARGET_PARAMS)
        if diff < best_diff:
            best_h = h
            best_diff = diff
            
    return best_h

# ============================================================
# 6. HYBRID RPPG LOSS FUNCTION
# ============================================================
class HybridWeightedRPPGLoss(nn.Module):
    def forward(self, preds, target, alpha=0.2):
        # Subtract mean to focus on rhythmic fluctuations
        preds_mean = preds - torch.mean(preds)
        target_mean = target - torch.mean(target)
        
        # Pearson correlation
        numerator = torch.sum(preds_mean * target_mean)
        denominator = torch.sqrt(torch.sum(preds_mean ** 2)) * torch.sqrt(torch.sum(target_mean ** 2))
        pearson_corr = numerator / (denominator + 1e-8)
        pearson_loss = 1 - pearson_corr

        # MSE Loss
        mse = nn.MSELoss()
        mse_loss = mse(preds_mean, target_mean)
        
        total_loss = (1 - alpha) * pearson_loss + alpha * mse_loss
        return pearson_loss, mse_loss, total_loss

criterion = HybridWeightedRPPGLoss()

# ============================================================
# 7. TRAINING & EVALUATION LOGIC
# ============================================================
def evaluate(model, loader):
    model.eval()
    total_loss, total_pearson, total_mse = 0.0, 0.0, 0.0
    active_norm_sum, count = 0.0, 0

    with torch.no_grad():
        for color_seq, signal_seq in loader:
            color_seq = color_seq.to(device)
            signal_seq = signal_seq.to(device)
            
            # Get predictions and stats
            predictions, stats = model(color_seq, return_stats=True)
            
            # FIXING NOTEBOOK BATCH SLICING BUG: 
            # We slice the sequence/time dimension (dim 1), not the batch dimension (dim 0).
            predictions_masked = predictions[:, MASK_EARLY_FRAMES:, :].flatten()
            signal_seq_masked = signal_seq[:, MASK_EARLY_FRAMES:].flatten()
            
            pearson, mse, loss = criterion(predictions_masked, signal_seq_masked)
            
            total_loss += loss.item()
            total_pearson += pearson.item()
            total_mse += mse.item()
            
            if "active_norm" in stats:
                active_norm_sum += stats["active_norm"]
                count += 1
                
    model.train()
    return {
        "loss": total_loss / len(loader),
        "pearson": total_pearson / len(loader),
        "mse": total_mse / len(loader),
        "active_norm": active_norm_sum / max(count, 1)
    }

def train_one_model(mode):
    hidden_dim = choose_hidden_dim(mode)
    model = build_model(mode, hidden_dim).to(device)
    params = count_params(model)
    
    print("\n" + "=" * 80)
    print(f"START TRAINING: Model={mode.upper()} | Hidden={hidden_dim} | Params={params:,}")
    print("=" * 80)
    
    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    
    best_val_loss = float('inf')
    best_metrics = {}
    
    start_time = time.time()
    
    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        train_loss_sum = 0.0
        
        for i, (color_seq, signal_seq) in enumerate(train_loader):
            color_seq = color_seq.to(device)
            signal_seq = signal_seq.to(device)
            
            optimizer.zero_grad()
            predictions = model(color_seq)
            
            # Mask early frames (fixing batch slicing bug)
            predictions_masked = predictions[:, MASK_EARLY_FRAMES:, :].flatten()
            signal_seq_masked = signal_seq[:, MASK_EARLY_FRAMES:].flatten()
            
            _, _, loss = criterion(predictions_masked, signal_seq_masked)
            loss.backward()
            optimizer.step()
            
            train_loss_sum += loss.item()
            
        scheduler.step()
        
        # Validate
        val = evaluate(model, test_loader)
        
        if val["loss"] < best_val_loss:
            best_val_loss = val["loss"]
            best_metrics = dict(val)
            # Save checkpoint
            os.makedirs('checkpoints', exist_ok=True)
            torch.save(model.state_dict(), f"checkpoints/best_{mode}_rppg.pth")
            
        print(f"Epoch {epoch:02d}/{NUM_EPOCHS} | Train Loss: {train_loss_sum/len(train_loader):.4f} | "
              f"Val Loss: {val['loss']:.4f} (Pearson: {val['pearson']:.3f}, MSE: {val['mse']:.4f}) | "
              f"Active Norm: {val['active_norm']:.2f}")
              
    elapsed = time.time() - start_time
    print(f"Finished {mode} | Best Val Loss: {best_val_loss:.4f} | Time: {elapsed:.1f}s")
    
    # Save a visual plot of predictions
    plot_predictions(model, mode)
    
    return {
        "mode": mode,
        "params": params,
        "hidden": hidden_dim,
        "best_val_loss": best_val_loss,
        "best_pearson": best_metrics.get("pearson", 0.0),
        "best_mse": best_metrics.get("mse", 0.0),
        "active_norm": best_metrics.get("active_norm", 0.0),
        "time": elapsed
    }

def plot_predictions(model, mode):
    model.eval()
    color_seq, signal_seq = next(iter(test_loader))
    with torch.no_grad():
        color_seq = color_seq.to(device)
        predictions = model(color_seq)
        
    preds_np = predictions[0].detach().cpu().numpy()
    sign_np = signal_seq[0].numpy()
    
    plt.figure(figsize=(14, 5))
    plt.plot(sign_np, label='Ground Truth (Target)', color='blue', linewidth=2)
    plt.plot(preds_np, label=f'{mode.upper()} Prediction', color='orange', linewidth=2, linestyle='--')
    
    # Highlight the masked region
    plt.axvspan(0, MASK_EARLY_FRAMES, color='red', alpha=0.1, label='Masked early frames (60 frames)')
    
    plt.title(f'{mode.upper()} rPPG Prediction vs Ground Truth')
    plt.xlabel('Frame Index')
    plt.ylabel('Amplitude')
    plt.legend(loc='upper right')
    plt.grid(True, alpha=0.4)
    plt.savefig(f'{mode}_rppg_prediction.png')
    plt.close()

# ============================================================
# 8. EXECUTION
# ============================================================
if __name__ == "__main__":
    if RUN_MODE == "all_quick":
        modes = ["rnn", "lstm", "rnn_residual", "lstm_residual", "fen"]
    else:
        modes = [RUN_MODE]
        
    results = []
    for mode in modes:
        res = train_one_model(mode)
        results.append(res)
        
    print("\n\n" + "#" * 80)
    print("FINAL BENCHMARK SUMMARY (UBFC-rPPG)")
    print("#" * 80)
    print(f"{'Model':<15} | {'Params':<8} | {'Best Val Loss':<13} | {'Pearson Loss':<12} | {'MSE Loss':<8} | {'Active Norm':<11} | {'Time':<6}")
    print("-" * 85)
    for r in results:
        print(f"{r['mode'].upper():<15} | {r['params']:<8,} | {r['best_val_loss']:<13.4f} | {r['best_pearson']:<12.4f} | {r['best_mse']:<8.4f} | {r['active_norm']:<11.2f} | {r['time']:<6.1f}s")
    print("#" * 80)
