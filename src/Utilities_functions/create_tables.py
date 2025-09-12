import matplotlib.pyplot as plt
import cv2 as cv
from matplotlib import pyplot as plt

def pathname(classical_prob, method):
    path = f"/Users/steveh/Downloads/NIH 23-24/LocReg_Python/{classical_prob}_prob_nrun_1_{method}.png"
    return path

classical_prob_names = ['baart', 'blur', 'deriv2', 'foxgood', 'gravity', 'heat', 'phillips', 'shaw', 'wing']

# List of image paths for locreg_gt method and classical_prob_names
locreg_images = [pathname(classical_prob, 'locreg_gt') for classical_prob in classical_prob_names]

# List of image paths for other methods (gcv_gt, lc_gt, dp_gt, lambda_plt)
gcv_images = [pathname(classical_prob, 'gcv_gt') for classical_prob in classical_prob_names]
lcurve_images = [pathname(classical_prob, 'lc_gt') for classical_prob in classical_prob_names]
dp_images = [pathname(classical_prob, 'dp_gt') for classical_prob in classical_prob_names]
lambda_images = [pathname(classical_prob, 'lambda_plt') for classical_prob in classical_prob_names]

import cv2 as cv
import matplotlib.pyplot as plt

# Define the list of image file paths for each category
categories = [locreg_images, gcv_images, dp_images, lcurve_images, lambda_images]
num_categories = len(categories)

# Define the number of columns for each category
num_columns = [1, 2, 3, 4, 5]

# Define the row and column names
classical_prob_names = ['baart', 'blur', 'deriv2', 'foxgood', 'gravity', 'heat', 'phillips', 'shaw', 'wing']
column_names = ['LocReg', 'GCV', 'L-Curve', 'DP', 'Lambda']

# Calculate the total number of subplots
total_subplots = sum(num_columns)

plt.figure(figsize=(55, 50))

# Create a figure with a reasonable figure size
for category_index, image_paths in enumerate(categories):
    for i, image_path in enumerate(image_paths):
        # Read the image
        image = cv.imread(image_path)
        
        # Calculate the subplot position based on category and index
        subplot_position = i * num_categories + category_index + 1
        
        # Create the subplot with adjusted size and display the image
        ax = plt.subplot(total_subplots, num_categories, subplot_position)
        ax.imshow(image, 'gray')
        
        # Hide the axes and labels
        ax.axis('off')
        
        # Add column names to the top row of subplots
        if subplot_position <= num_categories:
            ax.annotate(column_names[category_index], xy=(0.5, 1.2), xycoords='axes fraction',
                        fontsize=30, ha='center')
        
        # Add row names to the leftmost column of subplots
        if subplot_position % num_categories == 1:
            ax.annotate(classical_prob_names[i], xy=(-0.3, 0.5), xycoords='axes fraction',
                        fontsize=30, va='center', rotation=90)

plt.subplots_adjust(wspace=0)  # Adjust the spacing as needed (0.1 is an example)
plt.subplots_adjust(hspace=0)  # Adjust the spacing as needed (0.1 is an example)

plt.grid(True, linestyle='--', alpha=0.5, zorder=0)  # Set zorder to ensure visibility
plt.tight_layout()
# Save the figure without white space and rescale to fit
plt.savefig("output_figure.png", bbox_inches='tight', dpi=300)  # Adjust file format and DPI as needed





# Set up the grid for the images
num_rows = len(classical_prob_names) 
num_columns = 5

# Create a black background
fig = plt.figure(figsize=(16, 16), dpi=600, facecolor='black')  # Adjust the figsize parameter

# Set up the ImageGrid with reduced padding
# grid = ImageGrid(fig, 111, nrows_ncols=(num_rows, num_columns+1), axes_pad=0.3)  # Adjust the axes_pad value

# Add column names
column_names = ['LocReg', 'GCV', 'L-Curve', 'DP', 'Lambda']
for i, col_name in enumerate(column_names):
    ax = fig.add_subplot(1, num_columns, i+1)
    ax.set_title(col_name, fontsize=18, color='white')
    ax.grid(False)  # Remove gridlines

# Add row names
row_names = classical_prob_names
for i, row_name in enumerate(row_names):
    ax = fig.add_subplot(num_rows, 1, i+1)
    ax.set_ylabel(row_name, fontsize=18, color='white')
    ax.grid(False)  # Remove gridlines

# Add the images to the grid
for i in range(num_rows):
    grid[i * (num_columns+1)].imshow(plt.imread(locreg_images[i]), aspect='auto')
    grid[i * (num_columns+1) + 1].imshow(plt.imread(gcv_images[i]), aspect='auto')
    grid[i * (num_columns+1) + 2].imshow(plt.imread(lcurve_images[i]), aspect='auto')
    grid[i * (num_columns+1) + 3].imshow(plt.imread(dp_images[i]), aspect='auto')
    grid[i * (num_columns+1) + 4].imshow(plt.imread(lambda_images[i]), aspect='auto')

# Turn off axis labels and ticks for all subplots
# for ax in grid:
#     ax.axis('on')

plt.savefig('multi_row_image_table_with_names.png', dpi=600, transparent=True, bbox_inches='tight')