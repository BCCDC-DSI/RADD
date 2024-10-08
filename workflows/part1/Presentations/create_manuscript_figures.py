import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Define directories
dates_dir = '/arc/project/st-ashapi01-1/RADD'
rds_dir = '/scratch/st-ashapi01-1/rds_files/data_2024nps-db'
db_dir = '/arc/project/st-ashapi01-1/RADD_libraries'

# Read in the main file
# Ensure ms1 and ms2 are DataFrames with necessary columns
# Precursor ion
ms1 = pd.read_csv(os.path.join(rds_dir, 'ms1/combined_ms1.csv'))
# Fragments
ms2 = pd.read_csv(os.path.join(rds_dir, 'ms2/combined_ms2.csv'))
print(ms1.head())
print(ms2.head())
# Read databases
db1 = pd.read_csv(os.path.join(db_dir, 'NPS_DB-240705.csv'), index_col=False, skiprows=5)
print(db1.head())
print(db1.columns.to_list())
# Remove columns with a single unique value
keep = db1.apply(lambda col: col.nunique() > 1)
db1 = db1.loc[:, keep]

databases = {"NPS": db1}

# Function to calculate PPM range
def calc_ppm_range(theor_mass, err_ppm=10):
    theor_mass = float(theor_mass)  # Ensure theor_mass is float
    lower_bound = (-err_ppm / 1e6 * theor_mass) + theor_mass
    upper_bound = (err_ppm / 1e6 * theor_mass) + theor_mass
    return lower_bound, upper_bound

# Calculate evidence for each peak
def calculate_matches(row):
    lower_bound, upper_bound = calc_ppm_range(row['Product m/z'], err_ppm=20)
    return lower_bound <= row['m.z'] <= upper_bound

# Filter fragments and drop duplicates
fragments = db1[db1['Workflow'] == 'Fragment'].drop_duplicates(subset=['Compound Name', 'Product m/z'])

# Calculate evidence for each peak
evidence_per_fragment = ms2.merge(fragments, left_on='compound_name', right_on='Compound Name', how='left')
evidence_per_fragment['match'] = evidence_per_fragment.apply(calculate_matches, axis=1)

# Ensure 'm/z' is included for grouping
evidence_per_fragment = evidence_per_fragment[['filename', 'compound_name', 'spectrum', 'm/z', 'Product m/z', 'match']]
evidence_per_fragment_original = evidence_per_fragment.copy()
# Group by file, compound, spectrum, and m/z
evidence_per_fragment = (
    evidence_per_fragment
    .groupby(['filename', 'compound_name', 'spectrum', 'm/z'])
    .agg(
        match=('match', 'sum'),
        product_mz=('Product m/z', 'first')  # or use lambda x: x.iloc[0]
    )
    .reset_index()
)

evidence_per_fragment = evidence_per_fragment.reset_index()
print(evidence_per_fragment)
# Aggregate evidence per parent
evidence_per_parent = (evidence_per_fragment
                       .groupby(['filename', 'compound_name', 'spectrum'])
                       .agg(matches=('match', 'sum'),
                       dbmz=('m/z', 'first'))
                       ).reset_index()

# Calculate total fragments
total_fragments = fragments.groupby('Compound Name').size().reset_index(name='total_fragments')

# Merge total_fragments into evidence_per_parent
evidence_per_parent = evidence_per_parent.merge(total_fragments, left_on='compound_name', right_on='Compound Name', how='left')

print(ms1.columns.to_list())

# Merge with ms1 (including m/z)
ms1 = ms1.merge(evidence_per_parent, on=["filename", "compound_name"], how="left")

print(ms1.columns.to_list())
# Modify this back to m.z and mz
ms1["i"] = ms1["i"].astype(float)
ms1["rt"] = ms1["rt"].astype(float)
# Tag ppm error
# mass that was found in the sample compared to precursor ion in the database

# Modify this back to m.z and mz
ms1["ppm_err"] = 1e6 * (ms1["i"] - ms1["rt"]) / ms1["rt"]

# Keep only best evidence per file
best = ms1.sort_values(by=["matches"], ascending=False).groupby(["filename", "compound_name"]).head(1).reset_index(drop=True)
best2 = best[best["matches"] >= 2]

# Calculate proportion of reference product ions detected
best2.loc[:, "prop_reference_ions"] = best2["matches"] / best2["total_fragments"]

print(best2.columns.to_list())
## Fragment matching

# Plotting
plt.figure(figsize=(14, 10))  # Adjust size if needed
plt.tight_layout(rect=[0, 0.03, 1, 0.95])  # Adjust margins if necessary


# Scatter plot of proportion of reference product ions detected vs. number of matches
sns.scatterplot(data=best2, x='prop_reference_ions', y='matches', hue='compound_name', style='filename', palette='viridis')

plt.title('Evidence and Proportion of Reference Product Ions Detected')
plt.xlabel('Proportion of Reference Product Ions Detected')
plt.ylabel('Number of Matches')
plt.legend(title='Compound Name', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.savefig('Figures/scatter.png')

# Prepare data
best2['pct'] = round(100 * best2['matches'] / best2['total_fragments'])
best2['pct_cut'] = pd.qcut(best2['matches'] / (best2['total_fragments']+1e-10), 5)
# Set up color palettes
pal1 = ['#e0e0e0', '#1f78b4', '#33a02c', '#fb9a99', '#e31a1c']  # Example colors
pal2 = sns.cubehelix_palette(100, start=2, rot=0, dark=0.3, light=0.7)

# Bar chart: number of fragments
plt.figure(figsize=(12, 5))
sns.countplot(data=best2, x='Compound Name', hue='matches', palette=pal1,
order=best2['matches'].value_counts(ascending=False).index)
plt.title('Number of Samples of compounds by # of matches')
plt.xticks(rotation=45, ha='right')
plt.ylabel('# of samples')
plt.ylim(0, 500)
plt.tight_layout()
plt.savefig('Figures/Compound Counts.png')

# Bar chart: proportion of fragments
plt.figure(figsize=(12, 5))
sns.histplot(data=best2, x='Compound Name', hue='pct', multiple='stack', palette=pal2)
plt.title('Proportion of Fragments')
plt.xticks(rotation=45, ha='right')
plt.ylabel('# of samples')
plt.ylim(0, 500)
plt.tight_layout()
plt.savefig('Figures/proportionfragments.png')

# Boxplot: retention times
best2['rt_secs'] = 60*best2['rt']
plt.figure(figsize=(12, 5))
sns.boxplot(data=best2, x='Compound Name', y='rt_secs', color='grey', fliersize=0)
sns.stripplot(data=best2, x='Compound Name', y='rt_secs', color='black', size=3, jitter=True)
plt.title('Retention Time by Compound Name')
plt.xticks(rotation=45, ha='right')
plt.ylabel('Retention time (s)')
plt.tight_layout()
plt.savefig('Figures/retention_times.png')

# Boxplot: mass accuracy
plt.figure(figsize=(12, 5))
sns.boxplot(data=best2, x='Compound Name', y='ppm_err', color='grey', fliersize=0)
sns.stripplot(data=best2, x='Compound Name', y='ppm_err', color='black', size=3, jitter=True)
plt.title('Mass Accuracy by Compound Name')
plt.xticks(rotation=45, ha='right')
plt.ylabel('Mass accuracy (ppm)')
plt.tight_layout()
plt.savefig('Figures/mass_accuracy.png')


# Save the combined figure
plt.savefig("figure2-revised.pdf", format='pdf', bbox_inches='tight')