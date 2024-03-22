import os
import pandas as pd
import matplotlib.pyplot as plt

# Load the data from CSV
folder_name = "./"
test_name = "log"
folder_name = os.path.join(folder_name, test_name)

non_data = pd.read_csv(os.path.join(folder_name, 'non.csv'))
pos_data = pd.read_csv(os.path.join(folder_name, 'pos.csv'))
neg_data = pd.read_csv(os.path.join(folder_name, 'neg.csv'))
x = range(1, len(pos_data['asr'])+1)

# Plot the data with different line styles
plt.figure(figsize=(15, 10))
plt.plot(x, pos_data['psr'], label='passing - pos-TE', linestyle='-', marker='o', color='b')  # Solid line with circle markers
plt.plot(x, pos_data['msr'], label='meeting - pos-TE', linestyle='--', marker='s', color='b')  # Dashed line with square markers
plt.plot(x, pos_data['asr'], label='average - pos-TE', linestyle='-.', marker='^', color='b')  # Dash-dot line with triangle markers

plt.plot(x, non_data['psr'], label='passing - non-TE', linestyle='-', marker='o', color='g')  # Solid line with circle markers
plt.plot(x, non_data['msr'], label='meeting - non-TE', linestyle='--', marker='s', color='g')  # Dashed line with square markers
plt.plot(x, non_data['asr'], label='average - non-TE', linestyle='-.', marker='^', color='g')  # Dash-dot line with triangle markers

plt.plot(x, neg_data['psr'], label='passing - neg-TE', linestyle='-', marker='o', color='r')  # Solid line with circle markers
plt.plot(x, neg_data['msr'], label='meeting - neg-TE', linestyle='--', marker='s', color='r')  # Dashed line with square markers
plt.plot(x, neg_data['asr'], label='average - neg-TE', linestyle='-.', marker='^', color='r')  # Dash-dot line with triangle markers

plt.xlabel('Number of rounds')
plt.ylabel('Success rate')
plt.title(f'My performance playing against different agents - {test_name}')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(folder_name, "result.pdf"), format='pdf')
plt.show()
