#for data
import numpy as np
import pandas as pd
#for plotting
import matplotlib.pyplot as plt
import seaborn as sns
from pandas.plotting import parallel_coordinates

#Read in Datasets
mushrooms_dataset_path = "C:\\Users\\conor\\Documents\\msiss\\information systems\\Group Project\\mushrooms.csv"
df = pd.read_csv(mushrooms_dataset_path)

# Check Target Clas Balances
print(df['class'].value_counts())
print(df['class'].value_counts(normalize=True))
# Visualize
df['class'].value_counts().plot(kind='bar')
plt.title('Class Distribution')
plt.show()

#alternative way to show above information
# Counter Plots
features = ['cap-shape', 'cap-color', 'odor', 'gill-size']
fig, axes = plt.subplots(2, 2, figsize=(14,10))
for ax, feature in zip(axes.flatten(), features):
    sns.countplot(data=df, x=feature, hue='class', ax=ax)
    ax.set_title(feature)
plt.tight_layout()
plt.show()

# HEATMAP, alternative way to show above information
# Crosstab (normalized)
ct = pd.crosstab(df['cap-shape'], df['class'], normalize='index')
# Set style
sns.set_theme(style="whitegrid", font_scale=1.2)
# Create figure
plt.figure(figsize=(8, 6))
# Heatmap
ax = sns.heatmap(
    ct,
    annot=True,
    fmt=".2f",
    cmap="RdYlBu_r",        # nicer diverging palette
    linewidths=0.5,
    linecolor='white',
    cbar_kws={'label': 'Proportion'},
    annot_kws={"weight": "bold"}
)
# Titles and labels
ax.set_title("Cap Shape vs Class (Normalized)", fontsize=16, weight='bold', pad=15)
ax.set_xlabel("Class", fontsize=12)
ax.set_ylabel("Cap Shape", fontsize=12)
# Improve tick labels
ax.set_xticklabels(['Edible', 'Poisonous'], rotation=0)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
# Remove spines for cleaner look
for spine in ax.spines.values():
    spine.set_visible(False)
plt.tight_layout()
plt.show()


# For each categorical feature, see how it relates to target
def plot_feature_target_relationship(df, feature, target):
    """Shows how each category predicts the target"""
    cross_tab = pd.crosstab(df[feature], df[target], normalize='index')
    
    # Plot
    cross_tab.plot(kind='bar', stacked=True, figsize=(10, 6))
    plt.title(f'{feature} vs Target')
    plt.xlabel(feature)
    plt.ylabel('Proportion')
    plt.legend(title=target)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    return cross_tab

# Apply to all features
plot_feature_target_relationship(df, 'cap-shape', 'class')
# delete quotation marks below to run for all features, 
# but it is a lot of plots to look at
"""
for col in df.columns:
    if col != 'class':
        plot_feature_target_relationship(df, col, 'class')
"""





# Convert categories to numbers first (required for parallel coordinates)
df_numeric = df.copy()
for col in df_numeric.columns:
    df_numeric[col] = df_numeric[col].astype('category').cat.codes

# Create parallel coordinates plot
plt.figure(figsize=(12, 6))
parallel_coordinates(df_numeric, class_column='class', alpha=0.5)
plt.title('Parallel Coordinates Plot')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

