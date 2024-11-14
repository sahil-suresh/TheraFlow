import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Sample data construction (replace this with your actual data load)
data = {
    "Timestamp": [
    "00:00:00", "00:00:30", "00:01:00", "00:01:30", "00:02:00", "00:02:30",
    "00:03:00", "00:03:30", "00:04:00", "00:04:30", "00:05:00", "00:05:30",
    "00:06:00", "00:06:30", "00:07:00", "00:07:30", "00:08:00", "00:08:30",
    "00:09:00", "00:09:30", "00:10:00", "00:10:30", "00:11:00", "00:11:30",
    "00:12:00", "00:12:30", "00:13:00"],

    "HPC freq": [3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 3, 3, 3, 3, 3, 3, 3, 3, 3, 3, 2, 2, 2, 2, 3],
    "FHx freq":  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0],
    "Surg. Hx freq":  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "DHx freq":  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
    "SHx freq":  [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    "Summary": ["HPC", "HPC", "HPC", "HPC", "HPC", "HPC", "HPC", "HPC", "HPC", "FHx", "FHx", "Surg. Hx", "Exam", "Exam", "Exam", "Exam",
    "SHx", "SHx", "Exam + Lab explanation", "Exam + Lab explanation", "Exam + Lab explanation", "Exam + Lab explanation", "Exam + Lab explanation", "Exam + Lab explanation", "DHx", "DHx", "DHx"]
}

df = pd.DataFrame(data)
df["Timestamp"] = pd.to_datetime(df["Timestamp"], format="%H:%M:%S")

# Define colors for each theme and frequency line
theme_colors = {
    "HPC": "#4B0082",         # Indigo
    "FHx": "#1E90FF",          # Dodger Blue
    "Surg. Hx": "#32CD32",    # Lime Green
    "Exam": "#FF6347",        # Tomato Red
    "SHx": "#FFD700",          # Gold
    "Exam + Lab explanation": "#FF4500", # Orange Red
    "DHx": "#708090"           # Slate Grey
}


# Plotting
plt.figure(figsize=(12, 6))

# Plot each frequency line with matching colors
plt.plot(df["Timestamp"], df["HPC freq"], label="HPC freq", color=theme_colors["HPC"])
plt.plot(df["Timestamp"], df["FHx freq"], label="FHx freq", color=theme_colors["FHx"])
plt.plot(df["Timestamp"], df["Surg. Hx freq"], label="Surg. Hx freq", color=theme_colors["Surg. Hx"])
plt.plot(df["Timestamp"], df["SHx freq"], label="SHx freq", color=theme_colors["SHx"])
plt.plot(df["Timestamp"], df["DHx freq"], label="DHx freq", color=theme_colors["DHx"])

# Color-coding for themes
added_labels = set()  # Track themes added to legend

for theme, color in theme_colors.items():
    # Highlight background for each theme, adding a small offset to close gaps
    theme_rows = df[df["Summary"] == theme]
    start = theme_rows["Timestamp"].min()
    end = theme_rows["Timestamp"].max() + pd.Timedelta(seconds=30)  # Slightly extend to close gaps
    label = theme if theme not in added_labels else None  # Add label only once
    plt.axvspan(start, end, color=color, alpha=0.2, label=label)
    added_labels.add(theme)

# Labels and legend
plt.xlabel("Conversation Timestamp")
plt.ylabel("Frequency")
plt.title("Frequency by Question Type Over Time")

# Customizing y-axis to display integers only
plt.yticks(np.arange(0, 4, 1))  # Set y-ticks to integer values only

# Adjusting the x-axis format to start at 00:00
plt.gca().xaxis.set_major_formatter(plt.matplotlib.dates.DateFormatter("%H:%M:%S"))

# Move legend to outside the plot area to avoid obstruction
plt.legend(loc="upper left", bbox_to_anchor=(1, 1))

plt.xticks(rotation=45)
plt.tight_layout()

plt.show()