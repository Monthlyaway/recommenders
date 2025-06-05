import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Data provided in the image
data = {
    'Category': ['AIME 2024', 'Codeforces', 'GPQA Diamond', 'MATH-500', 'MMLU', 'SWE-bench'],
    'Metric_Type': ['(Pass@1)', '(Percentile)', '(Pass@1)', '(Pass@1)', '(Pass@1)', '(Pass@1)'],
    'DeepSeek-R1': [79.8, 96.3, 71.5, 97.3, 90.8, 49.2],
    'OpenAI-o1-1217': [79.2, 96.6, 75.7, 96.4, 91.8, 48.9],
    'DeepSeek-R1-32B': [72.6, 93.4, 62.1, 94.3, 87.4, 36.8],
    'OpenAI-o1-mini': [63.6, 58.7, 60.0, 90.0, 85.2, 41.6],
    # None for missing data
    'DeepSeek-V3': [39.2, None, 59.1, 90.2, None, 42.0]
}

df = pd.DataFrame(data)

# Reshape the DataFrame for Seaborn's barplot
df_melted = df.melt(id_vars=['Category', 'Metric_Type'],
                    var_name='Model',
                    value_name='Accuracy / Percentile (%)')

# Define custom colors to mimic the image
# The colors in the image are blue (DeepSeek-R1, DeepSeek-R1-32B, DeepSeek-V3)
# and grey (OpenAI-o1-1217, OpenAI-o1-mini) with varying shades/patterns.
# We'll approximate this.
palette_colors = {
    'DeepSeek-R1': '#4285F4',           # A shade of blue
    'OpenAI-o1-1217': '#9E9E9E',        # Grey
    'DeepSeek-R1-32B': '#8AB4F8',       # Lighter blue
    'OpenAI-o1-mini': '#C0C0C0',        # Lighter grey
    'DeepSeek-V3': '#E8F0FE'            # Very light blue
}


plt.figure(figsize=(15, 8))
sns.barplot(x='Category', y='Accuracy / Percentile (%)',
            hue='Model', data=df_melted, palette=palette_colors)

# Add value labels on top of the bars
for p in plt.gca().patches:
    if not pd.isna(p.get_height()):  # Check if height is not NaN
        plt.gca().annotate(f'{p.get_height():.1f}',
                           (p.get_x() + p.get_width() / 2., p.get_height()),
                           ha='center', va='center',
                           xytext=(0, 10),
                           textcoords='offset points',
                           fontsize=9)

# Customize the plot
plt.ylabel('Accuracy / Percentile (%)', fontsize=12)
plt.xlabel('')  # No general x-label, categories are self-explanatory
plt.ylim(0, 100)  # Y-axis from 0 to 100
plt.grid(axis='y', linestyle='--', alpha=0.7)  # Add horizontal grid lines

# Add the metric type under each category label
ax = plt.gca()
xtick_labels = []
for i, category in enumerate(df['Category']):
    metric_type = df['Metric_Type'][i]
    xtick_labels.append(f'{category}\n{metric_type}')
ax.set_xticklabels(xtick_labels)


# Customize legend
plt.legend(title='', loc='upper left', bbox_to_anchor=(
    0.02, 1.05), ncol=5, frameon=True, fontsize=10)

plt.tight_layout()
plt.show()
