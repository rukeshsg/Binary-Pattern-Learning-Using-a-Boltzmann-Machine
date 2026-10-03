"""
================================================================================
Binary Pattern Learning Using a Restricted Boltzmann Machine (RBM)
================================================================================
A complete from-scratch implementation of an Energy-Based Restricted 
Boltzmann Machine (RBM) using NumPy, Pandas, and Matplotlib.

Author: Academic Project
Domain: Machine Learning / Generative Models / Associative Memory
Architecture: 25 Visible Units (5x5 Binary Grid) -> 24 Hidden Units
Training: Contrastive Divergence (CD-1) with Momentum & Weight Decay
Evaluation: 20% Bit-Flip Noise Injection & Associative Denoising
================================================================================
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Global seed for exact scientific reproducibility
SEED = 42
np.random.seed(SEED)


class RestrictedBoltzmannMachine:
    """
    Restricted Boltzmann Machine (RBM) implemented from scratch in pure NumPy.
    
    Mathematical Formulation:
    -------------------------
    1. Energy Function:
       E(v, h) = - sum_i(a_i * v_i) - sum_j(b_j * h_j) - sum_i sum_j(v_i * W_ij * h_j)
               = - v^T * W * h - a^T * v - b^T * h

    2. Conditional Probabilities:
       P(h_j = 1 | v) = sigmoid(b_j + sum_i(v_i * W_ij))
       P(v_i = 1 | h) = sigmoid(a_i + sum_j(h_j * W_ij))

    3. Contrastive Divergence (CD-k) Updates:
       dW = (v_0^T * P(h_0 | v_0) - v_k^T * P(h_k | v_k)) / N - weight_decay * W
       da = mean(v_0 - v_k)
       db = mean(P(h_0 | v_0) - P(h_k | v_k))
    """
    def __init__(self, n_visible=25, n_hidden=24, learning_rate=0.04, 
                 momentum=0.5, weight_decay=0.0001, random_state=42):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.weight_decay = weight_decay
        self.random_state = random_state
        self.rng = np.random.RandomState(random_state)
        
        # Xavier / He style Gaussian initialization
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
            'reconstruction_loss_bce': [],
            'free_energy': []
        }

    @staticmethod
    def sigmoid(z):
        """Numerically stable logistic sigmoid activation."""
        z_clipped = np.clip(z, -30.0, 30.0)
        return 1.0 / (1.0 + np.exp(-z_clipped))

    def sample_bernoulli(self, probabilities):
        """Stochastic binary sampling from Bernoulli distribution."""
        return (self.rng.rand(*probabilities.shape) < probabilities).astype(np.float64)

    def propagate_up(self, v):
        """
        Compute hidden unit activation probabilities P(h_j = 1 | v) and sample states.
        """
        h_probs = self.sigmoid(np.dot(v, self.W) + self.b)
        h_states = self.sample_bernoulli(h_probs)
        return h_probs, h_states

    def propagate_down(self, h):
        """
        Compute visible unit activation probabilities P(v_i = 1 | h) and sample states.
        """
        v_probs = self.sigmoid(np.dot(h, self.W.T) + self.a)
        v_states = self.sample_bernoulli(v_probs)
        return v_probs, v_states

    def free_energy(self, v):
        """
        Calculate analytical Free Energy of visible vector v:
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
        
        # --- Positive Phase (Data Driven) ---
        h_probs_0, h_states_0 = self.propagate_up(v_batch)
        pos_associations = np.dot(v_batch.T, h_probs_0)
        
        # --- Negative Phase (Model Reconstruction via Gibbs Sampling) ---
        v_current_sample = v_batch
        h_current_sample = h_states_0
        
        for _ in range(k):
            v_probs_k, v_states_k = self.propagate_down(h_current_sample)
            h_probs_k, h_states_k = self.propagate_up(v_states_k)
            v_current_sample = v_states_k
            h_current_sample = h_states_k
        
        neg_associations = np.dot(v_current_sample.T, h_probs_k)
        
        # --- Compute Parameter Gradients ---
        dW = (pos_associations - neg_associations) / batch_size - self.weight_decay * self.W
        da = np.mean(v_batch - v_current_sample, axis=0)
        db = np.mean(h_probs_0 - h_probs_k, axis=0)
        
        # --- Momentum Updates ---
        self.v_W = self.momentum * self.v_W + self.learning_rate * dW
        self.v_a = self.momentum * self.v_a + self.learning_rate * da
        self.v_b = self.momentum * self.v_b + self.learning_rate * db
        
        self.W += self.v_W
        self.a += self.v_a
        self.b += self.v_b
        
        return np.mean((v_batch - v_probs_k) ** 2)

    def fit(self, X_train, n_epochs=500, batch_size=16, k=1, verbose=True):
        """
        Train the RBM using mini-batch Contrastive Divergence.
        """
        n_samples = X_train.shape[0]
        if verbose:
            print("=" * 72)
            print(f"  TRAINING RESTRICTED BOLTZMANN MACHINE (Contrastive Divergence CD-{k})")
            print(f"  Visible Units: {self.n_visible} (5x5 pixels) | Hidden Units: {self.n_hidden}")
            print(f"  Training Samples: {n_samples} | Batch Size: {batch_size} | Epochs: {n_epochs}")
            print(f"  Learning Rate: {self.learning_rate} | Momentum: 0.50 -> 0.90")
            print("=" * 72)

        start_time = time.time()
        for epoch in range(1, n_epochs + 1):
            # Dynamic momentum schedule: lower initial momentum for stability
            self.momentum = 0.90 if epoch > 30 else 0.50
            
            # Shuffle training batches
            indices = self.rng.permutation(n_samples)
            X_shuffled = X_train[indices]
            
            for i in range(0, n_samples, batch_size):
                v_batch = X_shuffled[i:i + batch_size]
                self.contrastive_divergence(v_batch, k=k)
            
            # Record convergence metrics on training data
            recon_probs, _ = self.reconstruct(X_train, steps=1, return_probabilities=True)
            epoch_mse = np.mean((X_train - recon_probs) ** 2)
            epoch_bce = -np.mean(X_train * np.log(np.clip(recon_probs, 1e-10, 1.0)) + 
                                 (1.0 - X_train) * np.log(np.clip(1.0 - recon_probs, 1e-10, 1.0)))
            avg_fe = np.mean(self.free_energy(X_train))
            
            self.history['epoch'].append(epoch)
            self.history['mean_squared_error'].append(epoch_mse)
            self.history['reconstruction_loss_bce'].append(epoch_bce)
            self.history['free_energy'].append(avg_fe)
            
            if verbose and (epoch % 50 == 0 or epoch == 1 or epoch == n_epochs):
                print(f"  Epoch [{epoch:3d}/{n_epochs:3d}] | "
                      f"MSE: {epoch_mse:.5f} | "
                      f"BCE Loss: {epoch_bce:.4f} | "
                      f"Free Energy: {avg_fe:.2f}")

        elapsed = time.time() - start_time
        if verbose:
            print("-" * 72)
            print(f"  Training Completed in {elapsed:.2f}s | Final Training MSE: {self.history['mean_squared_error'][-1]:.6f}")
            print("=" * 72)
        return self.history

    def reconstruct(self, v_input, steps=5, return_probabilities=True):
        """
        Reconstruct patterns through multi-step Gibbs sampling / mean-field relaxation.
        Allows the RBM to settle into the nearest learned energy minimum.
        """
        v_curr = np.copy(v_input)
        for _ in range(steps):
            h_probs, h_states = self.propagate_up(v_curr)
            v_probs, v_states = self.propagate_down(h_probs)
            v_curr = v_probs
        
        binary_output = (v_probs >= 0.5).astype(np.float64)
        if return_probabilities:
            return v_probs, binary_output
        return binary_output


# ==============================================================================
# Data Handling & Controlled Synthetic Noise Generation
# ==============================================================================

def load_canonical_dataset(csv_path):
    """
    Load dataset and extract the pristine canonical ground-truth patterns.
    """
    df = pd.read_csv(csv_path)
    pixel_cols = [c for c in df.columns if c.startswith('pixel_')]
    
    # Extract pristine canonical prototypes (mode / thresholded average)
    canonical_dict = {}
    for pattern_name, group in df.groupby('pattern'):
        avg = group[pixel_cols].mean().values
        proto = (avg >= 0.5).astype(np.float64)
        canonical_dict[pattern_name] = proto
        
    return df, canonical_dict, pixel_cols


def prepare_training_data(canonical_dict, replicates=25, add_single_pixel_variations=True):
    """
    Prepare clean training dataset.
    Trains the RBM on clean patterns and legitimate single-pixel variations to form
    wide, stable attractor basins in energy space.
    """
    clean_patterns = list(canonical_dict.values())
    X_train_list = []
    
    for p in clean_patterns:
        # Replicate clean prototype
        for _ in range(replicates):
            X_train_list.append(p.copy())
            
        # Add legitimate single-pixel variations to teach local basin curvature
        if add_single_pixel_variations:
            for pixel_idx in range(25):
                variant = p.copy()
                variant[pixel_idx] = 1.0 - variant[pixel_idx]
                X_train_list.append(variant)
                
    return np.array(X_train_list, dtype=np.float64)


def add_controlled_noise(clean_matrix, noise_fraction=0.20, seed=42):
    """
    Corrupt patterns by flipping EXACTLY noise_fraction of pixels (e.g. 5 out of 25 = 20%).
    Ensures fair, uniform corruption across all pattern classes.
    """
    rng = np.random.RandomState(seed)
    noisy_matrix = np.copy(clean_matrix)
    n_samples, n_pixels = clean_matrix.shape
    k_flips = int(round(noise_fraction * n_pixels))  # Exactly 5 pixels for 25 units
    
    for i in range(n_samples):
        flip_indices = rng.choice(n_pixels, size=k_flips, replace=False)
        noisy_matrix[i, flip_indices] = 1.0 - noisy_matrix[i, flip_indices]
        
    return noisy_matrix, k_flips


def compute_metrics(y_true, y_pred, y_noisy=None):
    """
    Compute rigorous quantitative evaluation metrics.
    Accuracy is measured strictly against the CLEAN ground truth.
    """
    pixel_accuracy = np.mean(y_true == y_pred) * 100.0
    incorrect_pixels = int(np.sum(y_true != y_pred))
    mse = np.mean((y_true - y_pred) ** 2)
    ber = np.mean(y_true != y_pred)
    
    results = {
        'Pixel Accuracy (%)': pixel_accuracy,
        'Incorrect Pixels': incorrect_pixels,
        'Mean Squared Error': mse,
        'Bit Error Rate': ber
    }
    
    if y_noisy is not None:
        initial_acc = np.mean(y_true == y_noisy) * 100.0
        initial_incorrect = int(np.sum(y_true != y_noisy))
        results['Initial Noisy Accuracy (%)'] = initial_acc
        results['Initial Incorrect Pixels'] = initial_incorrect
        results['Accuracy Gain (%)'] = pixel_accuracy - initial_acc
        
    return results


# ==============================================================================
# Publication-Grade Visualizations
# ==============================================================================

def setup_plot_style():
    """Configure matplotlib styling."""
    plt.rcParams.update({
        'font.family': 'sans-serif',
        'font.sans-serif': ['DejaVu Sans', 'Arial', 'Helvetica'],
        'axes.edgecolor': '#cbd5e1',
        'axes.linewidth': 1.2,
        'grid.color': '#f1f5f9',
        'figure.facecolor': '#ffffff',
        'axes.facecolor': '#ffffff',
    })


def plot_original_patterns(canonical_dict, save_path):
    """Plot and save ground-truth 5x5 binary patterns in 2x4 grid."""
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
    """Plot and save corrupted 20% noisy patterns."""
    setup_plot_style()
    patterns = list(noisy_dict.keys())
    fig, axes = plt.subplots(2, 4, figsize=(11, 6), dpi=300)
    fig.suptitle(f"Corrupted Test Input Patterns (20% Bit-Flip Noise = 5 Flips/Pattern)", 
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
    """Plot and save RBM reconstructed patterns."""
    setup_plot_style()
    patterns = list(recon_dict.keys())
    fig, axes = plt.subplots(2, 4, figsize=(11, 6), dpi=300)
    fig.suptitle("RBM Reconstructed Patterns (Energy-Based Associative Denoising)", 
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
    Compact, non-overlapping comparison figure with 8 rows and 4 columns:
    Column 1: Original Ground Truth
    Column 2: 20% Noisy Input
    Column 3: RBM Reconstruction
    Column 4: Error Map
    """
    setup_plot_style()
    patterns = list(orig_dict.keys())
    n_patterns = len(patterns)
    
    fig, axes = plt.subplots(n_patterns, 4, figsize=(11, 2.2 * n_patterns), dpi=300)
    fig.suptitle("Pattern Restoration Analysis: Original vs Noisy Input vs RBM Output", 
                 fontsize=14, fontweight='bold', y=0.995, color='#0f172a')
    
    col_titles = ["Column 1: Original Ground Truth", 
                  "Column 2: 20% Noisy Input", 
                  "Column 3: RBM Reconstruction", 
                  "Column 4: Error Map (True vs Recon)"]
    
    for col_idx, title in enumerate(col_titles):
        axes[0, col_idx].set_title(title, fontsize=10, fontweight='bold', pad=10, color='#1e293b')
        
    for row_idx, name in enumerate(patterns):
        orig = orig_dict[name].reshape(5, 5)
        noisy = noisy_dict[name].reshape(5, 5)
        recon = recon_dict[name].reshape(5, 5)
        diff = np.abs(orig - recon)
        
        row_metrics = metrics_per_pattern[name]
        acc = row_metrics['Pixel Accuracy (%)']
        incorrect = row_metrics['Incorrect Pixels']
        
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
        
        # 4. Difference Map (0 = Match [White], 1 = Error [Red])
        ax_diff = axes[row_idx, 3]
        ax_diff.imshow(diff, cmap='YlOrRd', vmin=0, vmax=1)
        
        # Right annotation
        status_text = f"Accuracy: {acc:.0f}%\nErrors: {incorrect}/25"
        status_color = '#065f46' if incorrect == 0 else '#b91c1c'
        ax_diff.text(5.5, 2.0, status_text, fontsize=9, fontweight='bold', 
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

    plt.tight_layout(rect=[0, 0.01, 0.88, 0.98])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  [SAVED] Comparison figure: {save_path}")


def plot_training_error(history, save_path):
    """
    Plot training reconstruction loss and diagnostic Free Energy graph.
    """
    setup_plot_style()
    epochs = history['epoch']
    mse = history['mean_squared_error']
    bce = history['reconstruction_loss_bce']
    free_energy = history['free_energy']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=300)
    fig.suptitle("Restricted Boltzmann Machine Training Convergence (Contrastive Divergence CD-1)", 
                 fontsize=13, fontweight='bold', y=0.98, color='#0f172a')
    
    # Left: Reconstruction Loss (MSE & BCE)
    color_mse = '#2563eb'
    color_bce = '#dc2626'
    
    line1 = ax1.plot(epochs, mse, color=color_mse, linewidth=2.2, label='Mean Squared Error (MSE)')
    ax1.set_xlabel('Training Epochs', fontsize=11, fontweight='bold', color='#1e293b')
    ax1.set_ylabel('Reconstruction MSE', color=color_mse, fontsize=11, fontweight='bold')
    ax1.tick_params(axis='y', labelcolor=color_mse)
    ax1.grid(True, linestyle='--', alpha=0.5)
    
    ax1_twin = ax1.twinx()
    line2 = ax1_twin.plot(epochs, bce, color=color_bce, linewidth=1.8, linestyle='-.', label='Cross-Entropy Loss (BCE)')
    ax1_twin.set_ylabel('Binary Cross-Entropy Loss', color=color_bce, fontsize=11, fontweight='bold')
    ax1_twin.tick_params(axis='y', labelcolor=color_bce)
    
    lines = line1 + line2
    labels = [l.get_label() for l in lines]
    ax1.legend(lines, labels, loc='upper right', framealpha=0.9)
    ax1.set_title("Reconstruction Loss Progression vs Epoch", fontsize=11, fontweight='bold', pad=10)
    
    # Right: Diagnostic Free Energy Minimization
    ax2.plot(epochs, free_energy, color='#059669', linewidth=2.2, label='Average Free Energy F(v)')
    ax2.set_xlabel('Training Epochs', fontsize=11, fontweight='bold', color='#1e293b')
    ax2.set_ylabel('Free Energy (Lower = Higher Probability)', fontsize=11, fontweight='bold', color='#059669')
    ax2.grid(True, linestyle='--', alpha=0.5)
    ax2.set_title("Diagnostic: Thermodynamic Free Energy Minimization", fontsize=11, fontweight='bold', pad=10)
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
    Consolidated master overview dashboard saving to screenshots/output.png.
    """
    setup_plot_style()
    fig = plt.figure(figsize=(16, 11), dpi=300)
    gs = gridspec.GridSpec(3, 3, figure=fig, height_ratios=[1.1, 1.0, 1.0])
    
    fig.suptitle("Boltzmann Machine Binary Pattern Learning — Master Execution Dashboard", 
                 fontsize=15, fontweight='bold', y=0.98, color='#0f172a')
    
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
    initial_accs = metrics_df['Initial Noisy Accuracy (%)'].values
    
    x = np.arange(len(names))
    width = 0.35
    ax_bar.bar(x - width/2, initial_accs, width, label='Noisy Input (20%)', color='#f87171')
    ax_bar.bar(x + width/2, accs, width, label='RBM Recon', color='#10b981')
    ax_bar.set_xticks(x)
    ax_bar.set_xticklabels(names, rotation=45, ha='right', fontsize=8, fontweight='bold')
    ax_bar.set_ylim(0, 115)
    ax_bar.set_ylabel('Pixel Accuracy (%)', fontsize=9, fontweight='bold')
    ax_bar.legend(loc='lower right', fontsize=8)
    ax_bar.grid(True, linestyle='--', alpha=0.4)
    
    # Panel 3: Training Error Convergence (Middle Left)
    ax_train = fig.add_subplot(gs[1, 0:2])
    ax_train.set_title("3. Reconstruction Loss & Free Energy vs Epoch", fontsize=11, fontweight='bold', pad=8)
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
    
    # Panel 4: Learned Weight Filters (Middle Right)
    ax_weights = fig.add_subplot(gs[1, 2])
    ax_weights.set_title("4. Learned RBM Weight Filters (24 Hidden Units)", fontsize=11, fontweight='bold', pad=8)
    weight_canvas = np.zeros((4 * 6, 6 * 6))
    for h in range(min(24, weights.shape[1])):
        hr = h // 6
        hc = h % 6
        w_patch = weights[:, h].reshape(5, 5)
        weight_canvas[hr*6:hr*6+5, hc*6:hc*6+5] = w_patch
    
    im_w = ax_weights.imshow(weight_canvas, cmap='coolwarm', interpolation='nearest')
    ax_weights.set_xticks([])
    ax_weights.set_yticks([])
    plt.colorbar(im_w, ax=ax_weights, fraction=0.046, pad=0.04, label='Weight Value')
    
    # Panel 5: Restoration Panorama (Bottom Span)
    ax_comp = fig.add_subplot(gs[2, :])
    ax_comp.set_title("5. Sample Restorations: [Clean Ground Truth  |  20% Noisy Input  |  Denoised RBM Output]", 
                      fontsize=11, fontweight='bold', pad=8)
    
    selected = ['DIAMOND', 'PLUS', 'SQUARE', 'X']
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
        ax_comp.text(c_base + 8, 5.8, "Noisy (20%)", ha='center', fontsize=7.5, color='#dc2626', fontweight='bold')
        ax_comp.text(c_base + 14, 5.8, "Reconstructed", ha='center', fontsize=7.5, color='#059669', fontweight='bold')
        
    ax_comp.set_xticks([])
    ax_comp.set_yticks([])
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.95])
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"  [SAVED] Master dashboard screenshot: {save_path}")


# ==============================================================================
# Main Pipeline Execution
# ==============================================================================

def run_pipeline(csv_path='dataset/binary_patterns_dataset.csv', 
                 results_dir='results', 
                 screenshots_dir='screenshots',
                 n_hidden=24,
                 n_epochs=500,
                 learning_rate=0.04,
                 noise_rate=0.20):
    """
    Execute end-to-end binary pattern learning, evaluation and visualization.
    """
    print("\n" + "=" * 78)
    print("      BINARY PATTERN LEARNING USING RESTRICTED BOLTZMANN MACHINE (RBM)")
    print("=" * 78)
    
    # 1. Load Clean Dataset
    print(f"\n[STEP 1] Loading Clean Dataset & Prototypes from '{csv_path}'...")
    df, canonical_dict, pixel_cols = load_canonical_dataset(csv_path)
    pattern_names = list(canonical_dict.keys())
    print(f"  Extracted {len(canonical_dict)} Clean Canonical Pattern Prototypes:")
    print(f"  Patterns: {', '.join(pattern_names)}")
    
    # 2. Prepare Training Data
    print("\n[STEP 2] Preparing Training Dataset (Clean Patterns + Structural Basis Variations)...")
    X_train = prepare_training_data(canonical_dict, replicates=25, add_single_pixel_variations=True)
    print(f"  Generated {X_train.shape[0]} training patterns (5x5 pixels = 25 visible units).")
    
    # 3. Initialize & Train RBM
    print(f"\n[STEP 3] Initializing RBM (Visible: 25, Hidden: {n_hidden}, lr: {learning_rate})...")
    rbm = RestrictedBoltzmannMachine(
        n_visible=25,
        n_hidden=n_hidden,
        learning_rate=learning_rate,
        momentum=0.5,
        weight_decay=0.0001,
        random_state=SEED
    )
    
    history = rbm.fit(X_train, n_epochs=n_epochs, batch_size=16, k=1, verbose=True)
    
    # 4. Generate Corrupted Test Inputs AFTER Training
    print(f"\n[STEP 4] Generating Corrupted Test Patterns ({int(noise_rate*100)}% Random Bit-Flip Noise)...")
    canonical_X = np.array([canonical_dict[p] for p in pattern_names])
    noisy_canonical_X, k_flips = add_controlled_noise(canonical_X, noise_fraction=noise_rate, seed=SEED)
    
    noisy_dict = {p: noisy_canonical_X[idx] for idx, p in enumerate(pattern_names)}
    print(f"  Flipped exactly {k_flips} out of 25 pixels (20% bit corruption) per pattern.")
    
    # 5. Denoise & Reconstruct via Gibbs Sampling
    print("\n[STEP 5] Performing Energy-Based Denoising & Associative Recall via Gibbs Sampling...")
    recon_probs, recon_binary = rbm.reconstruct(noisy_canonical_X, steps=5, return_probabilities=True)
    recon_dict = {p: recon_binary[idx] for idx, p in enumerate(pattern_names)}
    
    # 6. Quantitative Evaluation
    print("\n[STEP 6] Computing Quantitative Metrics (Evaluated vs Clean Ground Truth)...")
    metrics_per_pattern = {}
    table_rows = []
    
    for idx, p in enumerate(pattern_names):
        orig_v = canonical_X[idx]
        noisy_v = noisy_canonical_X[idx]
        recon_v = recon_binary[idx]
        
        m = compute_metrics(orig_v, recon_v, noisy_v)
        metrics_per_pattern[p] = m
        
        table_rows.append({
            'Pattern': p,
            'Initial Noisy Accuracy (%)': m['Initial Noisy Accuracy (%)'],
            'Recon Accuracy (%)': m['Pixel Accuracy (%)'],
            'Noisy Incorrect Pixels': f"{m['Initial Incorrect Pixels']}/25",
            'Recon Incorrect Pixels': f"{m['Incorrect Pixels']}/25",
            'Mean Squared Error': m['Mean Squared Error'],
            'Recovery Gain (%)': f"+{m['Accuracy Gain (%)']:.1f}%" if m['Accuracy Gain (%)'] >= 0 else f"{m['Accuracy Gain (%)']:.1f}%"
        })
        
    metrics_df = pd.DataFrame(table_rows)
    print("\n" + "-" * 78)
    print("                     PATTERN RECONSTRUCTION PERFORMANCE TABLE")
    print("-" * 78)
    print(metrics_df.to_string(index=False))
    print("-" * 78)
    
    overall_noisy_acc = np.mean(canonical_X == noisy_canonical_X) * 100.0
    overall_recon_acc = np.mean(canonical_X == recon_binary) * 100.0
    total_noisy_errors = int(np.sum(canonical_X != noisy_canonical_X))
    total_recon_errors = int(np.sum(canonical_X != recon_binary))
    
    print(f"\n  ================================================================")
    print(f"  FINAL SUMMARY EVALUATION RESULTS:")
    print(f"  • Overall Initial Corrupted Accuracy:  {overall_noisy_acc:.2f}% ({total_noisy_errors}/200 total errors)")
    print(f"  • Overall RBM Reconstructed Accuracy:  {overall_recon_acc:.2f}% ({total_recon_errors}/200 total errors)")
    print(f"  • Net Accuracy Recovery Gain:          +{overall_recon_acc - overall_noisy_acc:.2f}%")
    print(f"  • Perfectly Restored Patterns (100%):  {sum(1 for m in metrics_per_pattern.values() if m['Incorrect Pixels'] == 0)} / {len(pattern_names)}")
    print(f"  ================================================================\n")
    
    # 7. Generate Visual Artifacts
    print("[STEP 7] Generating High-Resolution Result Figures...")
    os.makedirs(results_dir, exist_ok=True)
    os.makedirs(screenshots_dir, exist_ok=True)
    
    plot_original_patterns(canonical_dict, os.path.join(results_dir, 'original_patterns.png'))
    plot_noisy_patterns(noisy_dict, noise_rate, os.path.join(results_dir, 'noisy_patterns.png'))
    plot_reconstructed_patterns(recon_dict, os.path.join(results_dir, 'reconstructed_patterns.png'))
    plot_comparison(canonical_dict, noisy_dict, recon_dict, metrics_per_pattern, os.path.join(results_dir, 'comparison.png'))
    plot_training_error(history, os.path.join(results_dir, 'training_error.png'))
    
    plot_full_dashboard(canonical_dict, noisy_dict, recon_dict, history, metrics_df, rbm.W, 
                        os.path.join(screenshots_dir, 'output.png'))
    
    print("\n" + "=" * 78)
    print("  RBM TRAINING & RECONSTRUCTION PIPELINE COMPLETED SUCCESSFULLY!")
    print("=" * 78)
    return rbm, metrics_df, metrics_per_pattern


if __name__ == '__main__':
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    csv_file = os.path.join(base_dir, 'dataset', 'binary_patterns_dataset.csv')
    res_dir = os.path.join(base_dir, 'results')
    ss_dir = os.path.join(base_dir, 'screenshots')
    
    run_pipeline(
        csv_path=csv_file, 
        results_dir=res_dir, 
        screenshots_dir=ss_dir,
        n_hidden=24,
        n_epochs=500,
        learning_rate=0.04,
        noise_rate=0.20
    )
