"""
================================================================================
Binary Pattern Learning Using a Restricted Boltzmann Machine (RBM)
================================================================================
A complete from-scratch implementation of an Energy-Based Restricted 
Boltzmann Machine using NumPy, Pandas, and Matplotlib.

Author: Academic Project
Topic: Binary Pattern Learning & Reconstruction
Dataset: 5x5 Binary Pattern Grid (25 visible units)
================================================================================
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Set random seed for reproducibility
np.random.seed(42)


class RestrictedBoltzmannMachine:
    """
    Restricted Boltzmann Machine (RBM) trained with Contrastive Divergence (CD-k).
    
    Mathematical Formulation:
    -------------------------
    Energy Function:
        E(v, h) = - sum(a_i * v_i) - sum(b_j * h_j) - sum(v_i * W_ij * h_j)
                = - v^T * a - h^T * b - v^T * W * h
                
    Conditional Probabilities:
        P(h_j = 1 | v) = sigmoid(b_j + sum_i(v_i * W_ij))
        P(v_i = 1 | h) = sigmoid(a_i + sum_j(h_j * W_ij))
        
    Contrastive Divergence (CD-1) Gradients:
        dW = (v_pos^T * h_pos - v_neg^T * h_neg) / batch_size - weight_decay * W
        da = mean(v_pos - v_neg)
        db = mean(h_pos - h_neg)
    """
    def __init__(self, n_visible=25, n_hidden=16, learning_rate=0.08, 
                 momentum=0.5, weight_decay=0.0001, random_state=42):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
        
        # Xavier-style weight initialization
        self.W = self.rng.normal(0.0, 0.05, size=(n_visible, n_hidden))
        self.a = np.zeros(n_visible)  # Visible bias
        self.b = np.zeros(n_hidden)   # Hidden bias
        
        # Velocity buffers for momentum
        self.v_W = np.zeros_like(self.W)
        self.v_a = np.zeros_like(self.a)
        self.v_b = np.zeros_like(self.b)
        
        self.history = {
            'epoch': [],
            'mean_squared_error': [],
            'reconstruction_error': [],
            'free_energy': []
        }

    @staticmethod
    def sigmoid(z):
        """Numerically stable logistic sigmoid activation function."""
        z_clipped = np.clip(z, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-z_clipped))

    def sample_bernoulli(self, probabilities):
        """Perform stochastic Bernoulli sampling given activation probabilities."""
        return (self.rng.rand(*probabilities.shape) < probabilities).astype(np.float64)

    def propagate_up(self, v):
        """
        Compute hidden unit activation probabilities P(h_j = 1 | v) and sample binary states.
        """
        h_probs = self.sigmoid(np.dot(v, self.W) + self.b)
        h_states = self.sample_bernoulli(h_probs)
        return h_probs, h_states

    def propagate_down(self, h):
        """
        Compute visible unit activation probabilities P(v_i = 1 | h) and sample binary states.
        """
        v_probs = self.sigmoid(np.dot(h, self.W.T) + self.a)
        v_states = self.sample_bernoulli(v_probs)
        return v_probs, v_states

    def free_energy(self, v):
        """
        Compute theoretical Free Energy of a visible vector v:
        F(v) = - v^T * a - sum_j ln(1 + exp(b_j + v * W_j))
        """
        v_bias_term = np.dot(v, self.a)
        wx_b = np.dot(v, self.W) + self.b
        hidden_term = np.sum(np.log1p(np.exp(np.clip(wx_b, -30.0, 30.0))), axis=-1)
        return -v_bias_term - hidden_term

    def contrastive_divergence(self, v_batch, k=1):
        """
        Execute 1-step or k-step Contrastive Divergence (CD-k) with momentum.
        """
        batch_size = v_batch.shape[0]
        
        # Positive Phase (Data-driven clamp)
        h_probs_0, h_states_0 = self.propagate_up(v_batch)
        pos_associations = np.dot(v_batch.T, h_probs_0)
        
        # Negative Phase (Gibbs sampling CD-k)
        v_current = v_batch
        h_states = h_states_0
        for _ in range(k):
            v_probs_k, v_states_k = self.propagate_down(h_states)
            h_probs_k, h_states_k = self.propagate_up(v_probs_k)
            h_states = h_states_k
            v_current = v_probs_k
        
        neg_associations = np.dot(v_current.T, h_probs_k)
        
        # Compute Gradients with L2 Weight Decay
        dW = (pos_associations - neg_associations) / batch_size - self.weight_decay * self.W
        da = np.mean(v_batch - v_current, axis=0)
        db = np.mean(h_probs_0 - h_probs_k, axis=0)
        
        # Update Velocities with Momentum
        self.v_W = self.momentum * self.v_W + self.learning_rate * dW
        self.v_a = self.momentum * self.v_a + self.learning_rate * da
        self.v_b = self.momentum * self.v_b + self.learning_rate * db
        
        # Update Parameters
        self.W += self.v_W
        self.a += self.v_a
        self.b += self.v_b
        
        return np.mean((v_batch - v_current) ** 2)

    def fit(self, X, n_epochs=150, batch_size=16, k=1, verbose=True):
        """
        Train the RBM using mini-batch Contrastive Divergence over multiple epochs.
        """
        n_samples = X.shape[0]
        if verbose:
            print("=" * 72)
            print(f"  TRAINING RESTRICTED BOLTZMANN MACHINE (Contrastive Divergence CD-{k})")
            print(f"  Visible Units: {self.n_visible} (5x5 pixels) | Hidden Units: {self.n_hidden}")
            print(f"  Training Samples: {n_samples} | Batch Size: {batch_size} | Epochs: {n_epochs}")
            print(f"  Learning Rate: {self.learning_rate} | Momentum: {self.momentum} -> 0.90")
            print("=" * 72)

        start_time = time.time()
        for epoch in range(1, n_epochs + 1):
            # Dynamic momentum schedule
            self.momentum = 0.90 if epoch > 25 else 0.50
            
            indices = self.rng.permutation(n_samples)
            X_shuffled = X[indices]
            
            for i in range(0, n_samples, batch_size):
                v_batch = X_shuffled[i:i + batch_size]
                self.contrastive_divergence(v_batch, k=k)
            
            # Epoch evaluation metrics
            recon_probs, _ = self.reconstruct(X, steps=1, return_probabilities=True)
            epoch_mse = np.mean((X - recon_probs) ** 2)
            epoch_bce = -np.mean(X * np.log(np.clip(recon_probs, 1e-10, 1.0)) + 
                                 (1.0 - X) * np.log(np.clip(1.0 - recon_probs, 1e-10, 1.0)))
            avg_free_energy = np.mean(self.free_energy(X))
            
            self.history['epoch'].append(epoch)
            self.history['mean_squared_error'].append(epoch_mse)
            self.history['reconstruction_error'].append(epoch_bce)
            self.history['free_energy'].append(avg_free_energy)
            
            if verbose and (epoch % 15 == 0 or epoch == 1 or epoch == n_epochs):
                print(f"  Epoch [{epoch:3d}/{n_epochs:3d}] | "
                      f"MSE: {epoch_mse:.5f} | "
                      f"Recon Loss (BCE): {epoch_bce:.4f} | "
                      f"Free Energy: {avg_free_energy:.2f}")

        elapsed = time.time() - start_time
        if verbose:
            print("-" * 72)
            print(f"  Training Completed in {elapsed:.2f}s | Final Reconstruction MSE: {self.history['mean_squared_error'][-1]:.6f}")
            print("=" * 72)
        return self.history

    def reconstruct(self, v, steps=1, return_probabilities=True):
        """
        Reconstruct visible patterns using forward-backward Gibbs sampling.
        """
        v_curr = np.copy(v)
        for _ in range(steps):
            h_probs, h_states = self.propagate_up(v_curr)
            v_probs, v_states = self.propagate_down(h_probs)
            v_curr = v_probs
        
        binary_output = (v_probs >= 0.5).astype(np.float64)
        if return_probabilities:
            return v_probs, binary_output
        return binary_output


# ==============================================================================
# Helper Functions: Data Loading, Noise Injection, Metrics & Plotting
# ==============================================================================

def load_dataset(csv_path):
    """
    Load the binary pattern dataset from CSV.
    """
    df = pd.read_csv(csv_path)
    pixel_cols = [c for c in df.columns if c.startswith('pixel_')]
    X = df[pixel_cols].values.astype(np.float64)
    y = df['pattern'].values
    return df, X, y, pixel_cols


def get_canonical_patterns(df, pixel_cols):
    """
    Extract the clean canonical 5x5 pattern prototype for each class.
    """
    canonical = {}
    for pattern_name, group in df.groupby('pattern'):
        avg_pixels = group[pixel_cols].mean().values
        binary_proto = (avg_pixels >= 0.5).astype(np.float64)
        canonical[pattern_name] = binary_proto
    return canonical


def add_noise(X, noise_rate=0.20, noise_type='bit_flip', seed=42):
    """
    Inject noise into binary pattern vectors.
    """
    rng = np.random.RandomState(seed)
    X_noisy = np.copy(X)
    
    if noise_type == 'bit_flip':
        flip_mask = rng.rand(*X.shape) < noise_rate
        X_noisy[flip_mask] = 1.0 - X_noisy[flip_mask]
    elif noise_type == 'occlusion':
        mask = rng.rand(*X.shape) < noise_rate
        X_noisy[mask] = 0.0
    return X_noisy


def compute_metrics(y_true, y_pred, y_noisy=None):
    """
    Compute quantitative evaluation metrics between ground-truth and predictions.
    """
    pixel_accuracy = np.mean(y_true == y_pred) * 100.0
    mse = np.mean((y_true - y_pred) ** 2)
    hamming_dist = np.sum(y_true != y_pred)
    ber = np.mean(y_true != y_pred)
    
    results = {
        'Pixel Accuracy (%)': pixel_accuracy,
        'Mean Squared Error': mse,
        'Bit Error Rate': ber,
        'Hamming Distance': hamming_dist
    }
    
    if y_noisy is not None:
        initial_acc = np.mean(y_true == y_noisy) * 100.0
        acc_gain = pixel_accuracy - initial_acc
        results['Initial Noisy Accuracy (%)'] = initial_acc
        results['Accuracy Recovery Gain (%)'] = acc_gain
    
    return results


# ==============================================================================
# Visualization Routines
# ==============================================================================

def setup_plot_style():
    """Configure clean matplotlib aesthetic parameters."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
        'axes.edgecolor': '#cbd5e1',
        'axes.linewidth': 1.2,
        'grid.color': '#f1f5f9',
        'grid.linestyle': '--',
        'figure.facecolor': '#ffffff',
        'axes.facecolor': '#ffffff',
    })


def plot_original_patterns(canonical_dict, save_path):
    """Plot and save canonical ground-truth binary patterns in a 2x4 grid."""
    setup_plot_style()
    patterns = list(canonical_dict.keys())
    fig, axes = plt.subplots(2, 4, figsize=(11, 6), dpi=300)
    fig.suptitle("Ground Truth 5×5 Binary Patterns (25 Visible Units)", 
                 fontsize=14, fontweight='bold', y=0.98, color='#0f172a')
    
    for idx, name in enumerate(patterns):
        ax = axes[idx // 4, idx % 4]
        grid = canonical_dict[name].reshape(5, 5)
        ax.imshow(grid, cmap='Blues', vmin=0, vmax=1, interpolation='nearest')
        
        for r in range(5):
            for c in range(5):
                val = int(grid[r, c])
                color = 'white' if val == 1 else '#475569'
                ax.text(c, r, str(val), ha='center', va='center', fontsize=9, 
                        fontweight='bold', color=color)
                
        ax.set_title(f"Pattern: {name}", fontsize=11, fontweight='bold', pad=8, color='#0f172a')
        ax.set_xticks(np.arange(-0.5, 5, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 5, 1), minor=True)
        ax.grid(which='minor', color='#94a3b8', linestyle='-', linewidth=1.2)
        ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
        for spine in ax.spines.values():
            spine.set_color('#64748b')
            spine.set_linewidth(1.5)
            
    plt.tight_layout(rect=[0, 0.03, 1, 0.94])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  [SAVED] Original patterns figure: {save_path}")


def plot_noisy_patterns(noisy_dict, noise_rate, save_path):
    """Plot and save corrupted/noisy binary patterns."""
    setup_plot_style()
    patterns = list(noisy_dict.keys())
    fig, axes = plt.subplots(2, 4, figsize=(11, 6), dpi=300)
    fig.suptitle(f"Corrupted Input Patterns (Bit-Flip Noise Rate: {int(noise_rate*100)}%)", 
                 fontsize=14, fontweight='bold', y=0.98, color='#991b1b')
    
    for idx, name in enumerate(patterns):
        ax = axes[idx // 4, idx % 4]
        grid = noisy_dict[name].reshape(5, 5)
        ax.imshow(grid, cmap='Reds', vmin=0, vmax=1, interpolation='nearest')
        
        for r in range(5):
            for c in range(5):
                val = int(grid[r, c])
                color = 'white' if val == 1 else '#7f1d1d'
                ax.text(c, r, str(val), ha='center', va='center', fontsize=9, 
                        fontweight='bold', color=color)
                
        ax.set_title(f"Noisy {name}", fontsize=11, fontweight='bold', pad=8, color='#7f1d1d')
        ax.set_xticks(np.arange(-0.5, 5, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 5, 1), minor=True)
        ax.grid(which='minor', color='#f87171', linestyle='-', linewidth=1.2)
        ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
        for spine in ax.spines.values():
            spine.set_color('#b91c1c')
            spine.set_linewidth(1.5)
            
    plt.tight_layout(rect=[0, 0.03, 1, 0.94])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  [SAVED] Noisy patterns figure: {save_path}")


def plot_reconstructed_patterns(recon_dict, save_path):
    """Plot and save reconstructed/denoised binary patterns."""
    setup_plot_style()
    patterns = list(recon_dict.keys())
    fig, axes = plt.subplots(2, 4, figsize=(11, 6), dpi=300)
    fig.suptitle("RBM Reconstructed Patterns (Energy-Based Associative Recall)", 
                 fontsize=14, fontweight='bold', y=0.98, color='#065f46')
    
    for idx, name in enumerate(patterns):
        ax = axes[idx // 4, idx % 4]
        grid = recon_dict[name].reshape(5, 5)
        ax.imshow(grid, cmap='Greens', vmin=0, vmax=1, interpolation='nearest')
        
        for r in range(5):
            for c in range(5):
                val = int(grid[r, c])
                color = 'white' if val == 1 else '#064e3b'
                ax.text(c, r, str(val), ha='center', va='center', fontsize=9, 
                        fontweight='bold', color=color)
                
        ax.set_title(f"Reconstructed {name}", fontsize=11, fontweight='bold', pad=8, color='#065f46')
        ax.set_xticks(np.arange(-0.5, 5, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, 5, 1), minor=True)
        ax.grid(which='minor', color='#34d399', linestyle='-', linewidth=1.2)
        ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
        for spine in ax.spines.values():
            spine.set_color('#059669')
            spine.set_linewidth(1.5)
            
    plt.tight_layout(rect=[0, 0.03, 1, 0.94])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  [SAVED] Reconstructed patterns figure: {save_path}")


def plot_comparison(orig_dict, noisy_dict, recon_dict, metrics_per_pattern, save_path):
    """
    Side-by-side comparison across all 8 patterns:
    Column 1: Original Clean Pattern
    Column 2: Noisy Corrupted Input
    Column 3: RBM Reconstructed Output
    Column 4: Difference / Error Map
    """
    setup_plot_style()
    patterns = list(orig_dict.keys())
    n_patterns = len(patterns)
    
    fig, axes = plt.subplots(n_patterns, 4, figsize=(12, 2.5 * n_patterns), dpi=300)
    fig.suptitle("Pattern Restoration Comparison: Original vs Noisy Input vs RBM Output", 
                 fontsize=15, fontweight='bold', y=0.995, color='#0f172a')
    
    col_titles = ["1. Original Ground Truth", "2. Noisy Input (20% Noise)", 
                  "3. RBM Reconstruction", "4. Error Map (Original vs Recon)"]
    
    for col_idx, title in enumerate(col_titles):
        axes[0, col_idx].set_title(title, fontsize=11, fontweight='bold', pad=10, color='#1e293b')
        
    for row_idx, name in enumerate(patterns):
        orig = orig_dict[name].reshape(5, 5)
        noisy = noisy_dict[name].reshape(5, 5)
        recon = recon_dict[name].reshape(5, 5)
        diff = np.abs(orig - recon)
        
        row_metrics = metrics_per_pattern[name]
        acc = row_metrics['Pixel Accuracy (%)']
        hamming = int(row_metrics['Hamming Distance'])
        
        # 1. Original
        ax_orig = axes[row_idx, 0]
        ax_orig.imshow(orig, cmap='Blues', vmin=0, vmax=1)
        ax_orig.set_ylabel(f"{name}", fontsize=11, fontweight='bold', color='#1e293b', rotation=0, labelpad=40)
        
        # 2. Noisy
        ax_noisy = axes[row_idx, 1]
        ax_noisy.imshow(noisy, cmap='Reds', vmin=0, vmax=1)
        
        # 3. Reconstructed
        ax_recon = axes[row_idx, 2]
        ax_recon.imshow(recon, cmap='Greens', vmin=0, vmax=1)
        
        # 4. Difference Map (0 = Match [white], 1 = Mismatch [red])
        ax_diff = axes[row_idx, 3]
        ax_diff.imshow(diff, cmap='YlOrRd', vmin=0, vmax=1)
        
        status_text = f"Acc: {acc:.0f}%\nErrors: {hamming}/25"
        status_color = '#065f46' if hamming == 0 else '#b91c1c'
        ax_diff.text(5.5, 2.5, status_text, fontsize=9, fontweight='bold', 
                     color=status_color, va='center')
        
        # Format grid cells
        for ax, grid_data in zip([ax_orig, ax_noisy, ax_recon, ax_diff], [orig, noisy, recon, diff]):
            for r in range(5):
                for c in range(5):
                    v = int(grid_data[r, c])
                    ax.text(c, r, str(v), ha='center', va='center', fontsize=8,
                            color='#334155' if v == 0 else 'white', fontweight='bold')
            ax.set_xticks(np.arange(-0.5, 5, 1), minor=True)
            ax.set_yticks(np.arange(-0.5, 5, 1), minor=True)
            ax.grid(which='minor', color='#cbd5e1', linestyle='-', linewidth=0.8)
            ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)

    plt.tight_layout(rect=[0, 0.01, 0.90, 0.98])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  [SAVED] Comparison figure: {save_path}")


def plot_training_error(history, save_path):
    """Plot training curves showing MSE, Reconstruction Error, and Free Energy."""
    setup_plot_style()
    epochs = history['epoch']
    mse = history['mean_squared_error']
    bce = history['reconstruction_error']
    free_energy = history['free_energy']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    fig.suptitle("Restricted Boltzmann Machine Training Convergence (Contrastive Divergence CD-1)", 
                 fontsize=13, fontweight='bold', y=0.98, color='#0f172a')
    
    # Left subplot: MSE & Binary Cross Entropy
    color_mse = '#2563eb'
    color_bce = '#dc2626'
    
    line1 = ax1.plot(epochs, mse, color=color_mse, linewidth=2.2, label='Mean Squared Error (MSE)')
    ax1.set_xlabel('Training Epochs', fontsize=11, fontweight='bold', color='#1e293b')
    ax1.set_ylabel('Reconstruction MSE', color=color_mse, fontsize=11, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=color_mse)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    # Twin axis for Cross Entropy
    ax1_twin = ax1.twinx()
    line2 = ax1_twin.plot(epochs, bce, color=color_bce, linewidth=2.0, linestyle='-.', label='Cross-Entropy Loss')
    ax1_twin.set_ylabel('Binary Cross-Entropy (BCE)', color=color_bce, fontsize=11, fontweight='bold')
    ax1_twin.tick_params(axis='y', labelcolor=color_bce)
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right', framealpha=0.9)
    ax1.set_title("Reconstruction Loss Progression", fontsize=11, fontweight='bold', pad=10)
    
    # Right subplot: Free Energy minimization
    ax2.plot(epochs, free_energy, color='#059669', linewidth=2.2, label='Average Free Energy F(v)')
    ax2.set_xlabel('Training Epochs', fontsize=11, fontweight='bold', color='#1e293b')
    ax2.set_ylabel('Free Energy', fontsize=11, fontweight='bold', color='#059669')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.set_title("Energy-Based Convergence: Free Energy Minimization", fontsize=11, fontweight='bold', pad=10)
    ax2.legend(loc='lower right', framealpha=0.9)
    
    final_mse = mse[-1]
    ax1.annotate(f'Final MSE: {final_mse:.5f}', 
                 xy=(epochs[-1], final_mse), xytext=(epochs[-1] * 0.65, final_mse + 0.04),
                 arrowprops=dict(facecolor=color_mse, shrink=0.08, width=1, headwidth=6),
                 fontsize=9, fontweight='bold', color=color_mse)
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  [SAVED] Training error figure: {save_path}")


def plot_full_dashboard(orig_dict, noisy_dict, recon_dict, history, metrics_df, weights, save_path):
    """
    Generate an all-in-one visual summary dashboard for presentation and README showcase.
    Saves to screenshots/output.png.
    """
    setup_plot_style()
    fig = plt.figure(figsize=(16, 11), dpi=300)
    gs = gridspec.GridSpec(3, 3, figure=fig, height_ratios=[1.1, 1.0, 1.0])
    
    fig.suptitle("Boltzmann Machine Binary Pattern Learning — Full Execution & Evaluation Dashboard", 
                 fontsize=16, fontweight='bold', y=0.98, color='#0f172a')
    
    # Panel 1: Original Patterns Grid (Top Left & Center)
    ax_orig = fig.add_subplot(gs[0, 0:2])
    ax_orig.set_title("1. Ground Truth Binary Patterns (5×5, 8 Classes)", fontsize=11, fontweight='bold', pad=8)
    patterns = list(orig_dict.keys())
    
    combined_orig = np.zeros((10, 20))
    for idx, p in enumerate(patterns):
        r_offset = (idx // 4) * 5
        c_offset = (idx % 4) * 5
        combined_orig[r_offset:r_offset+5, c_offset:c_offset+5] = orig_dict[p].reshape(5, 5)
    
    ax_orig.imshow(combined_orig, cmap='Blues', vmin=0, vmax=1)
    for idx, p in enumerate(patterns):
        r_offset = (idx // 4) * 5
        c_offset = (idx % 4) * 5
        ax_orig.text(c_offset + 2, r_offset - 0.5, p, ha='center', va='bottom', fontsize=8, fontweight='bold', color='#1e293b')
    ax_orig.set_xticks([])
    ax_orig.set_yticks([])
    
    # Panel 2: Performance Summary Bar Chart (Top Right)
    ax_bar = fig.add_subplot(gs[0, 2])
    ax_bar.set_title("2. Reconstruction Accuracy by Pattern (%)", fontsize=11, fontweight='bold', pad=8)
    names = metrics_df['Pattern'].values
    accs = metrics_df['Recon Accuracy (%)'].values
    initial_accs = metrics_df['Noisy Accuracy (%)'].values
    
    x = np.arange(len(names))
    width = 0.35
    ax_bar.bar(x - width/2, initial_accs, width, label='Noisy Input', color='#f87171')
    ax_bar.bar(x + width/2, accs, width, label='RBM Recon', color='#10b981')
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(names, rotation=45, ha='right', fontsize=8, fontweight='bold')
    ax_bar.set_ylim(0, 110)
    ax_bar.set_ylabel('Pixel Accuracy (%)', fontsize=9, fontweight='bold')
    ax_bar.legend(loc='lower right', fontsize=8)
    ax_bar.grid(True, linestyle='--', alpha=0.4)
    
    # Panel 3: Training Error Convergence (Middle Left)
    ax_train = fig.add_subplot(gs[1, 0:2])
    ax_train.set_title("3. Contrastive Divergence Training Loss & Free Energy", fontsize=11, fontweight='bold', pad=8)
    ax_train.plot(history['epoch'], history['mean_squared_error'], color='#2563eb', lw=2, label='Reconstruction MSE')
    ax_train.set_xlabel('Epochs', fontsize=9, fontweight='bold')
    ax_train.set_ylabel('MSE Loss', color='#2563eb', fontsize=9, fontweight='bold')
    ax_train.grid(True, linestyle='--', alpha=0.5)
    
    ax_train_fe = ax_train.twinx()
    ax_train_fe.plot(history['epoch'], history['free_energy'], color='#059669', lw=2, linestyle='--', label='Free Energy')
    ax_train_fe.set_ylabel('Free Energy F(v)', color='#059669', fontsize=9, fontweight='bold')
    
    lines_1, labels_1 = ax_train.get_legend_handles_labels()
    lines_2, labels_2 = ax_train_fe.get_legend_handles_labels()
    ax_train.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper right', fontsize=8)
    
    # Panel 4: Learned Hidden Receptive Fields (Middle Right)
    ax_weights = fig.add_subplot(gs[1, 2])
    ax_weights.set_title("4. Learned RBM Weight Filters (16 Hidden Units)", fontsize=11, fontweight='bold', pad=8)
    weight_canvas = np.zeros((4 * 6, 4 * 6))
    for h in range(min(16, weights.shape[1])):
        hr = h // 4
        hc = h % 4
        w_patch = weights[:, h].reshape(5, 5)
        weight_canvas[hr*6:hr*6+5, hc*6:hc*6+5] = w_patch
    
    im_w = ax_weights.imshow(weight_canvas, cmap='coolwarm', interpolation='nearest')
    ax_weights.set_xticks([])
    ax_weights.set_yticks([])
    plt.colorbar(im_w, ax=ax_weights, fraction=0.046, pad=0.04, label='Weight Strength')
    
    # Panel 5: Triplet Comparison (Bottom Span)
    ax_comp = fig.add_subplot(gs[2, :])
    ax_comp.set_title("5. Triplet Restoration Visualizer: [Ground Truth  |  Corrupted Input  |  Denoised RBM Output]", 
                      fontsize=11, fontweight='bold', pad=8)
    
    selected = ['X', 'PLUS', 'SQUARE', 'DIAMOND']
    pan_w = len(selected) * 18
    pan_img = np.zeros((5, pan_w))
    
    for idx, p in enumerate(selected):
        c_base = idx * 18
        pan_img[:, c_base:c_base+5] = orig_dict[p].reshape(5, 5)
        pan_img[:, c_base+6:c_base+11] = noisy_dict[p].reshape(5, 5)
        pan_img[:, c_base+12:c_base+17] = recon_dict[p].reshape(5, 5)
        
    ax_comp.imshow(pan_img, cmap='Blues', vmin=0, vmax=1)
    for idx, p in enumerate(selected):
        c_base = idx * 18
        ax_comp.text(c_base + 8.5, -0.8, f"Pattern: {p}", ha='center', fontsize=9, fontweight='bold', color='#1e293b')
        ax_comp.text(c_base + 2, 5.8, "Original", ha='center', fontsize=7.5, color='#0284c7', fontweight='bold')
        ax_comp.text(c_base + 8, 5.8, "Noisy", ha='center', fontsize=7.5, color='#dc2626', fontweight='bold')
        ax_comp.text(c_base + 14, 5.8, "Reconstructed", ha='center', fontsize=7.5, color='#059669', fontweight='bold')
        
    ax_comp.set_xticks([])
    ax_comp.set_yticks([])
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  [SAVED] Output dashboard screenshot: {save_path}")


# ==============================================================================
# Main Pipeline Execution
# ==============================================================================

def run_pipeline(csv_path='dataset/binary_patterns_dataset.csv', 
                 results_dir='results', 
                 screenshots_dir='screenshots'):
    """
    Execute end-to-end binary pattern learning and evaluation pipeline.
    """
    print("\n" + "=" * 75)
    print("      BINARY PATTERN LEARNING USING RESTRICTED BOLTZMANN MACHINE")
    print("=" * 75)
    
    # 1. Load Dataset
    print(f"\n[STEP 1] Loading Dataset from '{csv_path}'...")
    df, X, y, pixel_cols = load_dataset(csv_path)
    print(f"  Loaded {X.shape[0]} samples with {X.shape[1]} binary pixels each (5x5 grid).")
    print(f"  Pattern distribution:\n{df['pattern'].value_counts().to_string(header=False)}")
    
    # Extract clean canonical prototypes
    canonical_dict = get_canonical_patterns(df, pixel_cols)
    print(f"  Extracted {len(canonical_dict)} canonical pattern prototypes.")
    
    # 2. Initialize & Train RBM
    print("\n[STEP 2] Initializing and Training Restricted Boltzmann Machine...")
    rbm = RestrictedBoltzmannMachine(
        n_visible=25,
        n_hidden=16,
        learning_rate=0.08,
        momentum=0.5,
        weight_decay=0.0001,
        random_state=42
    )
    
    history = rbm.fit(X, n_epochs=150, batch_size=16, k=1, verbose=True)
    
    # 3. Create Noisy Patterns for Testing
    noise_rate = 0.20  # 20% bit flip noise
    print(f"\n[STEP 3] Generating Corrupted/Noisy Test Patterns (Noise Level: {int(noise_rate*100)}%)...")
    
    canonical_X = np.array([canonical_dict[p] for p in canonical_dict.keys()])
    noisy_canonical_X = add_noise(canonical_X, noise_rate=noise_rate, noise_type='bit_flip', seed=42)
    
    noisy_dict = {}
    for idx, p in enumerate(canonical_dict.keys()):
        noisy_dict[p] = noisy_canonical_X[idx]
        
    # 4. Reconstruct / Denoise Using Trained RBM
    print("\n[STEP 4] Performing Associative Recall & Pattern Denoising via RBM...")
    recon_probs, recon_binary = rbm.reconstruct(noisy_canonical_X, steps=1, return_probabilities=True)
    
    recon_dict = {}
    for idx, p in enumerate(canonical_dict.keys()):
        recon_dict[p] = recon_binary[idx]
        
    # 5. Evaluate Quantitative Metrics
    print("\n[STEP 5] Computing Quantitative Evaluation Metrics...")
    metrics_per_pattern = {}
    table_rows = []
    
    for idx, p in enumerate(canonical_dict.keys()):
        orig_v = canonical_X[idx]
        noisy_v = noisy_canonical_X[idx]
        recon_v = recon_binary[idx]
        
        m = compute_metrics(orig_v, recon_v, noisy_v)
        metrics_per_pattern[p] = m
        
        noisy_hamming = int(np.sum(orig_v != noisy_v))
        recon_hamming = int(np.sum(orig_v != recon_v))
        
        table_rows.append({
            'Pattern': p,
            'Noisy Accuracy (%)': np.mean(orig_v == noisy_v) * 100.0,
            'Recon Accuracy (%)': m['Pixel Accuracy (%)'],
            'Noisy Hamming': f"{noisy_hamming}/25",
            'Recon Hamming': f"{recon_hamming}/25",
            'MSE': m['Mean Squared Error'],
            'Recovery Gain (%)': m['Accuracy Recovery Gain (%)']
        })
        
    metrics_df = pd.DataFrame(table_rows)
    print("\n" + "-" * 75)
    print("                     PATTERN RECONSTRUCTION PERFORMANCE TABLE")
    print("-" * 75)
    print(metrics_df.to_string(index=False))
    print("-" * 75)
    
    overall_orig_noisy_acc = np.mean(canonical_X == noisy_canonical_X) * 100.0
    overall_recon_acc = np.mean(canonical_X == recon_binary) * 100.0
    print(f"  Overall Initial Corrupted Accuracy:  {overall_orig_noisy_acc:.2f}%")
    print(f"  Overall RBM Denoised Accuracy:       {overall_recon_acc:.2f}%")
    print(f"  Net Accuracy Improvement:            +{overall_recon_acc - overall_orig_noisy_acc:.2f}%")
    print("-" * 75)
    
    # 6. Generate and Save All Visualization Figures
    print("\n[STEP 6] Generating Visual Artifacts and Saving to Disk...")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(screenshots_dir, exist_ok=True)
    
    plot_original_patterns(canonical_dict, os.path.join(results_dir, 'original_patterns.png'))
    plot_noisy_patterns(noisy_dict, noise_rate, os.path.join(results_dir, 'noisy_patterns.png'))
    plot_reconstructed_patterns(recon_dict, os.path.join(results_dir, 'reconstructed_patterns.png'))
    plot_comparison(canonical_dict, noisy_dict, recon_dict, metrics_per_pattern, os.path.join(results_dir, 'comparison.png'))
    plot_training_error(history, os.path.join(results_dir, 'training_error.png'))
    
    # Master dashboard output screenshot
    plot_full_dashboard(canonical_dict, noisy_dict, recon_dict, history, metrics_df, rbm.W, 
                        os.path.join(screenshots_dir, 'output.png'))
    
    print("\n" + "=" * 75)
    print("  PIPELINE EXECUTION COMPLETED SUCCESSFULLY!")
    print("=" * 75)
    return rbm, metrics_df


if __name__ == '__main__':
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    csv_file = os.path.join(base_dir, 'dataset', 'binary_patterns_dataset.csv')
    res_dir = os.path.join(base_dir, 'results')
    ss_dir = os.path.join(base_dir, 'screenshots')
    
    run_pipeline(csv_path=csv_file, results_dir=res_dir, screenshots_dir=ss_dir)
