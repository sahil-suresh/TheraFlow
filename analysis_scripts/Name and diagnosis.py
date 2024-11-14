import numpy as np
import matplotlib.pyplot as plt

# Name binaries
data_raw = [
    1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    1, 1, 1, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, '--', 1, 1, '--', 1, 
    1, 0, 1, 1, 1, 1, 1, 1, '--', 1, 1, '--', 1, 1, 1, 1, 0, 0, 1, 1
]
data = [int(value) for value in data_raw if value != '--']

# Diagnosis binaries
diagnosis_data = [
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
    1, 1, 1, 1, 1, 1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
]

# Mean and SEM for name 
mean_correctness = np.mean(data)
sem_correctness = np.sqrt(mean_correctness * (1 - mean_correctness) / len(data))

# Mean and SEM for diagnosis accuracy
mean_diagnosis = np.mean(diagnosis_data)
sem_diagnosis = np.sqrt(mean_diagnosis * (1 - mean_diagnosis) / len(diagnosis_data))

# Convert proportions to percentages for display
mean_correctness_pct = mean_correctness * 100
sem_correctness_pct = sem_correctness * 100
mean_diagnosis_pct = mean_diagnosis * 100
sem_diagnosis_pct = sem_diagnosis * 100

# Plotting
plt.figure(figsize=(5, 6))
bars = plt.bar(['Name', 'Diagnosis'], 
               [mean_correctness_pct, mean_diagnosis_pct], 
               yerr=[sem_correctness_pct, sem_diagnosis_pct], 
               capsize=10, color=['#5aff91', '#808080'],  # Medical green and grey colors
               width=0.7)

# Adding percentage labels at the bottom of each bar, overlayed within the bars
for bar, height in zip(bars, [mean_correctness_pct, mean_diagnosis_pct]):
    plt.text(bar.get_x() + bar.get_width() / 2, height * 0.01, f'{height:.2f}%', 
             ha='center', va='bottom', fontsize=12, color='black')  # Positioned within each bar near the base

plt.ylim(0, 100.1)  # Keep ylim close to 100 to avoid extra space
plt.ylabel('Percent Accuracy')
plt.title('Dialogue Extraction Accuracy')
plt.tight_layout()  
plt.show()

# Print accuracy values
print(f"Mean accuracy for Name: {mean_correctness_pct:.2f}%")
print(f"Mean accuracy for Diagnosis: {mean_diagnosis_pct:.2f}%")
