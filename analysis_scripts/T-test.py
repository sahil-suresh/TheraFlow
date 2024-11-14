import pandas as pd
import math
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import ttest_rel

# Data in minutes
data = {
    'LLM_correct': [
        "03:00", "02:30", "08:30", "09:30", "01:00", "02:00", "04:00", "06:00", "07:00", "06:30", 
        "04:30", "00:30", "01:00", "07:00", "04:30", "05:00", "02:30", "05:00", "02:30", "00:00", 
        "00:30", "00:00", "00:00", "01:30", "11:00", "06:30", "05:30", "00:00", "00:30", "00:00", 
        "00:00", "09:00", "00:00", "00:00", "00:00", "09:30", "08:00", "00:00", "00:30", "01:00", 
        "00:30", "00:30", "00:00", "00:00", "09:00", "00:30", "05:30", "04:00", 
        "07:00", "05:00", "06:30", "06:30", "06:30", "00:00", "00:00", "06:00", 
        "00:30", "00:30", "01:00", "00:00", "09:30", "00:00"
    ],
    'in_dialogue': [
        "09:00", "08:30", "08:30", "09:30", "04:00", "10:00", "07:30", "06:00", "07:00", "06:30", 
        "09:00", "06:00", "07:00", "07:00", "04:30", "05:00", "05:00", "05:00", "02:30", "12:00", 
        "05:00", "05:00", "05:30", "09:00", "10:30", "06:00", "05:30", "11:00", "09:30", "05:00", 
        "05:30", "07:00", "05:30", "04:30", "05:30", "09:30", "08:00", "08:30", "07:00", "09:00", 
        "08:00", "06:30", "04:00", "07:30", "08:30", "10:30", "05:00", "07:30", 
        "06:00", "05:00", "06:00", "06:30", "06:30", "08:00", "08:00", "05:30", 
        "06:00", "06:30", "08:30", "07:00", "09:30", "09:30"
    ]
}

df = pd.DataFrame(data)

# Convert to minutes in decimal for calculation
def time_to_minutes(time_str):
    parts = time_str.split(':')
    minutes = int(parts[0])
    seconds = int(parts[1]) / 60
    return minutes + seconds

df['LLM_minutes'] = df['LLM_correct'].apply(time_to_minutes)
df['Dialogue_minutes'] = df['in_dialogue'].apply(time_to_minutes)

# Melt dataframe for seaborn
df_melted = pd.melt(df, value_vars=['LLM_minutes', 'Dialogue_minutes'], var_name='Condition', value_name='Time')

# Function to format y-axis to show mm:ss
def format_minutes(x, pos):
    m = int(x)
    s = int((x - m) * 60)
    return f"{m}:{s:02d}"

# Define updated colors
llm_color = "#5aff91"  # Flashier teal for LLM
doctor_color = "#a8a8a8"  # More subdued gray for Doctor
point_color = "#4d4d4d"  # Darker color for points to improve contrast

# Plotting
plt.figure(figsize=(10, 6))
sns.boxplot(x='Condition', y='Time', data=df_melted, palette=[llm_color, doctor_color], width=0.4)
sns.swarmplot(x='Condition', y='Time', data=df_melted, color=point_color, size=5)

# Paired T-test
t_stat, p_value = ttest_rel(df['LLM_minutes'], df['Dialogue_minutes'])

# Titles and labels
plt.title(f'Treatment prediction time by LLM\nPaired T-test p-value = {p_value:.3g}')
plt.ylabel("Time (minutes:seconds)")
plt.xlabel("")
plt.xticks([0, 1], ["Correct tx predicted by LLM", "Tx stated in dialogue"])

# Set y-axis to display mm:ss format
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(format_minutes))

plt.tight_layout()
plt.show()
