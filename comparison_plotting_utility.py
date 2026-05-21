import re
import matplotlib.pyplot as plt

def parse_log_file(filepath):
    """Parses the log file and returns lists of metrics."""
    epochs = []
    times = []
    train_losses = []
    test_l2s = []
    
    train_pattern = re.compile(r"\[(\d+)\] time=([\d\.]+), avg_loss=([\d\.]+), train_err=([\d\.]+)")
    eval_pattern = re.compile(r"Eval: test_l2=([\d\.]+)")
    
    try:
        with open(filepath, 'r') as f:
            for line in f:
                line = line.strip()
                
                train_match = train_pattern.search(line)
                if train_match:
                    epochs.append(int(train_match.group(1)))
                    times.append(float(train_match.group(2)))
                    train_losses.append(float(train_match.group(3)))
                    continue
                
                eval_match = eval_pattern.search(line)
                if eval_match:
                    test_l2s.append(float(eval_match.group(1)))
                    
    except FileNotFoundError:
        print(f"Warning: File '{filepath}' not found. Skipping.")
        
    return epochs, times, train_losses, test_l2s

def style_axis(ax, title, xlabel, ylabel):
    """Applies a clean, modern aesthetic to a matplotlib axis."""
    ax.set_title(title, fontsize=15, fontweight='bold', pad=15, color='#333333')
    ax.set_xlabel(xlabel, fontsize=12, fontweight='medium', color='#555555')
    ax.set_ylabel(ylabel, fontsize=12, fontweight='medium', color='#555555')
    
    # Soft grid
    ax.grid(True, linestyle=':', linewidth=1.5, color='#E0E0E0', zorder=0)
    
    # Remove top and right borders (despining)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_color('#DDDDDD')
    ax.spines['bottom'].set_color('#DDDDDD')
    
    # Tick color
    ax.tick_params(colors='#555555', length=0, pad=8)

def main():
    torch_file = 'log_run_torch.txt'
    jax_file = 'log_run_jax.txt'
    
    torch_ep, _, torch_train, torch_test = parse_log_file(torch_file)
    jax_ep, _, jax_train, jax_test = parse_log_file(jax_file)
    
    if not torch_ep and not jax_ep:
        print("No data found to plot. Please check your text files.")
        return

    # --- DOWNSAMPLING PARAMETER ---
    N = 7  # Plot every Nth point

    # Modern color palette
    c_torch = '#2A6B99' # Deep Slate Blue
    c_jax = '#E05D2D'   # Burnt Orange
    
    # Set up the figure with a slightly off-white background for a premium look
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.5), facecolor='#FCFCFC')
    fig.patch.set_facecolor('#FCFCFC')
    
    # ---------------------------------------------------------
    # PLOT 1: TRAINING LOSS
    # ---------------------------------------------------------
    style_axis(ax1, 'Training Loss vs Epochs', 'Epoch', 'Average Training Loss')
    
    if torch_ep:
        ep_sub, train_sub = torch_ep[::N], torch_train[::N]
        ax1.plot(ep_sub, train_sub, marker='o', color=c_torch, label='GINO PyTorch (Reference)',
                 linewidth=2.5, markersize=8, markeredgecolor='white', markeredgewidth=2, zorder=3)
        ax1.fill_between(ep_sub, train_sub, alpha=0.1, color=c_torch, zorder=2)
        
    if jax_ep:
        ep_sub, train_sub = jax_ep[::N], jax_train[::N]
        ax1.plot(ep_sub, train_sub, marker='s', color=c_jax, label='GINO JAX (Developed)',
                 linewidth=2.5, markersize=8, markeredgecolor='white', markeredgewidth=2, zorder=3)
        ax1.fill_between(ep_sub, train_sub, alpha=0.1, color=c_jax, zorder=2)
        
    ax1.legend(frameon=True, facecolor='white', edgecolor='#DDDDDD', fontsize=11, loc='upper right')

    # ---------------------------------------------------------
    # PLOT 2: TEST L2 ERROR
    # ---------------------------------------------------------
    style_axis(ax2, 'Test L2 Error vs Epochs', 'Epoch', 'Test L2 Error')
    
    if torch_ep and len(torch_test) > 0:
        # Match lengths safely before slicing
        ep_test = torch_ep[:len(torch_test)]
        ep_sub, test_sub = ep_test[::N], torch_test[::N]
        
        ax2.plot(ep_sub, test_sub, marker='o', color=c_torch, label='GINO PyTorch (Reference)',
                 linewidth=2.5, markersize=8, markeredgecolor='white', markeredgewidth=2, zorder=3)
        ax2.fill_between(ep_sub, test_sub, alpha=0.1, color=c_torch, zorder=2)
        
    if jax_ep and len(jax_test) > 0:
        ep_test = jax_ep[:len(jax_test)]
        ep_sub, test_sub = ep_test[::N], jax_test[::N]
        
        ax2.plot(ep_sub, test_sub, marker='s', color=c_jax, label='GINO JAX (Developed)',
                 linewidth=2.5, markersize=8, markeredgecolor='white', markeredgewidth=2, zorder=3)
        ax2.fill_between(ep_sub, test_sub, alpha=0.1, color=c_jax, zorder=2)
        
    ax2.legend(frameon=True, facecolor='white', edgecolor='#DDDDDD', fontsize=11, loc='upper right')

    # Final Layout Adjustments
    plt.suptitle('GINO Training Dynamics: PyTorch vs JAX', fontsize=18, color='#222222', y=0.98)
    plt.tight_layout(pad=3.0)
    plt.subplots_adjust(top=0.88) # Make room for the suptitle
    
    # Save the beautiful plot
    plt.savefig('gino_comparison_beautiful.png', dpi=300, bbox_inches='tight')
    print("Beautiful plot saved successfully as 'gino_comparison_beautiful.png'")
    # plt.show()

if __name__ == "__main__":
    main()
