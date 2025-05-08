import pandas as pd

# Input and output file paths
input_csv = '/Users/carlosperez/Desktop/GordonLab/consolidated_cell_types.csv'
output_csv = './filtered_orn.csv'

# Read the CSV
df = pd.read_csv(input_csv)

# Filter rows where primary_type starts with 'ORN'
filtered_df = df[df['primary_type'].str.startswith('ORN', na=False)]
sorted_df = filtered_df.sort_values(by='primary_type')

# Save to new CSV
sorted_df.to_csv(output_csv, index=False)

print(f"Filtered and sorted {len(sorted_df)} rows saved to {output_csv}")