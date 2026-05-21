import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import re

# 1. Initialize lists to store our extracted data
epochs = []
train_errors = []
test_errors = []

# 2. Open and parse the log file line by line
with open('training_log_precomputed_neighbor_search.txt', 'r') as file:
    for line in file:
        # Look for the training line e.g., "[0] time=1.31, avg_loss=0.8547, train_err=0.8547"
        train_match = re.search(r'\[(\d+)\] .*train_err=([0-9.]+)', line)
        if train_match:
            epochs.append(int(train_match.group(1)))
            train_errors.append(float(train_match.group(2)))
        
        # Look for the testing line e.g., "Eval: test_l2=0.5466"
        test_match = re.search(r'Eval: test_l2=([0-9.]+)', line)
        if test_match:
            test_errors.append(float(test_match.group(1)))

# Quick safety check to ensure data lengths match
if len(train_errors) != len(test_errors):
    print("Warning: Mismatch in number of training and testing logs found!")

# 3. Create the plot
plt.figure(figsize=(10, 6))

# Plot train and test curves with distinct markers and line styles
# Removed markers ('o' and 's') so it looks cleaner with hundreds of epochs
plt.plot(epochs, train_errors, label='Train Error', color='blue', linewidth=2)
plt.plot(epochs, test_errors, label='Test Error (L2)', color='red', linestyle='--', linewidth=2)

# 4. Format the plot to make it publication/report ready
plt.title('GINO CFD Model: Training vs Testing Loss', fontsize=16, fontweight='bold')
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('L2 Error', fontsize=14)

# Force a maximum of 10 ticks on the X-axis
plt.gca().xaxis.set_major_locator(MaxNLocator(integer=True, nbins=10)) 

plt.grid(True, linestyle='--', alpha=0.7)
plt.legend(fontsize=12)

# Adjust layout to prevent clipping
plt.tight_layout()

# 5. Save the plot as a high-res image AND display it
plt.savefig('loss_curve_precomputed_neighbor_search.png', dpi=300)
print("Plot saved successfully as 'loss_curve_precomputed_neighbor_search.png'")
plt.show()
