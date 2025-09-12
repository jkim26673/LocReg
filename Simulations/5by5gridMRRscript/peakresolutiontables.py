import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pandas.plotting import table

# Read the CSV file using pandas
df = pd.read_csv('/home/kimjosy/LocReg_Regularization-1/data/peakresolution/peakresolution_2024-12-14.csv')  # Replace 'your_file.csv' with your actual file path
filepath = "/home/kimjosy/LocReg_Regularization-1/data/peakresolution"
# Save the figure as PNG

styled_df = df.style.set_table_styles(
    [{'selector': 'thead th', 'props': [('background-color', 'lightblue'), ('font-weight', 'bold')]}, 
     {'selector': 'td', 'props': [('text-align', 'center')]},
     {'selector': 'tr:nth-child(odd)', 'props': [('background-color', 'lightgray')]}]
)
# Step 3: Create a Matplotlib figure
fig, ax = plt.subplots(figsize=(6, 3))  # Set the size of the image

# Step 4: Plot the table using Pandas table function
ax.axis('off')  # Turn off the axis
tbl = table(ax, df, loc='center', colWidths=[0.2]*len(df.columns))

# Customize table appearance (optional)
tbl.auto_set_font_size(False)
tbl.set_fontsize(12)
tbl.auto_set_column_width(col=list(range(len(df.columns))))

# Step 5: Save the image as a PNG
plt.savefig(f'{filepath}/styled_table_image.png', bbox_inches='tight', dpi=300)

latex_code = df.to_latex(index=False)  # `index=False` to avoid printing row indices

# Step 3: Print or save the LaTeX code
print(latex_code)

# Optional: Save the LaTeX code to a .tex file
with open(f'{filepath}/resolutiontable.tex', 'w') as f:
    f.write(latex_code)