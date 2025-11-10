"""
Week 1 – Python Foundations for Scientists
Course: Python for Neuroscience
Author: [Your Name]
------------------------------------------------------
Goals:
- Learn Python syntax, variables, loops, and functions
- Use lists and dictionaries
- Save and load simple data files
- Apply coding to neuroscience-style data
"""

# === 1. Welcome & Setup ===
# Make sure to install dependencies if needed:
# pip install numpy pandas matplotlib

import numpy as np
import matplotlib.pyplot as plt
import csv

# === 2. Python Basics ===
neuron_name = "V1_cell_01"
spike_count = 120
duration_sec = 60
firing_rate = spike_count / duration_sec

print(f"{neuron_name} firing rate: {firing_rate:.2f} Hz")

# === 3. Lists and Dictionaries ===
spikes = [0.01, 0.15, 0.34, 0.51, 0.89]  # spike times in seconds
metadata = {"neuron": "V1_cell_01", "region": "V1", "animal_id": "M001"}

print("Spike times:", spikes)
print("Neuron metadata:", metadata)

# === 4. Loops and Conditionals ===
for t in spikes:
    if t > 0.5:
        print(f"Late spike detected at {t}s")

# === 5. Functions ===
def firing_rate(spike_times, duration):
    """Compute firing rate (Hz) given spike times and duration."""
    return len(spike_times) / duration

rate = firing_rate(spikes, 1.0)
print(f"Firing rate from function: {rate:.2f} Hz")

# === 6. File I/O ===
with open("spike_data.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["neuron_id", "spike_time_s"])
    for t in spikes:
        writer.writerow([metadata["neuron"], t])

print("Saved spike_data.csv!")

# === 7. Mini Neuroscience Task ===
def mean_firing_rate(neuron_spikes, duration):
    """
    Compute mean firing rate across multiple neurons.
    neuron_spikes: list of lists containing spike times
    duration: total recording duration in seconds
    """
    rates = [firing_rate(s, duration) for s in neuron_spikes]
    return np.mean(rates)

neuron1 = np.random.uniform(0, 1, 10).tolist()
neuron2 = np.random.uniform(0, 1, 12).tolist()
neuron3 = np.random.uniform(0, 1, 8).tolist()

mean_rate = mean_firing_rate([neuron1, neuron2, neuron3], 1.0)
print(f"Mean firing rate across neurons: {mean_rate:.2f} Hz")

# === 8. Exercises ===

# 1. Compute inter-spike intervals (ISIs)
def inter_spike_intervals(spike_times):
    """Return list of ISIs given spike timestamps."""
    return np.diff(spike_times)

isis = inter_spike_intervals(sorted(spikes))
print("Inter-spike intervals (s):", isis)

# 2. Detect ISI outliers (> mean + 2 SD)
threshold = np.mean(isis) + 2 * np.std(isis)
outliers = [isi for isi in isis if isi > threshold]
print("Outlier ISIs:", outliers)

# 3. Simulate multiple neurons with random spike counts
neuron_rates = {}
for i in range(5):
    spikes_i = np.random.uniform(0, 1, np.random.randint(8, 15))
    rate_i = firing_rate(spikes_i, 1.0)
    neuron_rates[f"Neuron_{i+1}"] = rate_i

print("Neuron firing rates:", neuron_rates)

# Save to CSV
with open("firing_rates.csv", "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["neuron_id", "firing_rate_Hz"])
    for n, r in neuron_rates.items():
        writer.writerow([n, r])

print("Saved firing_rates.csv!")

# 4. Plot histogram of firing rates
plt.hist(list(neuron_rates.values()), bins=5, edgecolor='black')
plt.xlabel("Firing Rate (Hz)")
plt.ylabel("Count")
plt.title("Distribution of Neuron Firing Rates")
plt.show()


