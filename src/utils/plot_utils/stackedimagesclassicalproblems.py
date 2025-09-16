import matplotlib.pyplot as plt

def pathname(classical_prob, method):
    path = f"/Users/steveh/Downloads/NIH 23-24/LocReg_Python/figure_data/Creating Publication Figures Classical Problem/Classical problems figures 1023/{classical_prob}_prob_nrun_1_{method}.png"
    return path

# classical_prob_names = ['baart', 'blur', 'deriv2', 'foxgood', 'gravity', 'heat', 'phillips', 'shaw', 'wing']
classical_prob_names = ['deriv2']
# classical_prob_names = ['foxgood']


# List of image paths for locreg_gt method and classical_prob_names
locreg_images = [pathname(classical_prob, 'locreg_gt') for classical_prob in classical_prob_names]

# List of image paths for other methods (gcv_gt, lc_gt, dp_gt, lambda_plt)
gcv_images = [pathname(classical_prob, 'gcv_gt') for classical_prob in classical_prob_names]
lcurve_images = [pathname(classical_prob, 'lc_gt') for classical_prob in classical_prob_names]
dp_images = [pathname(classical_prob, 'dp_gt') for classical_prob in classical_prob_names]
lambda_images = [pathname(classical_prob, 'lambda_plt') for classical_prob in classical_prob_names]

# Define the list of image file paths for each category
categories = [locreg_images, gcv_images, dp_images, lcurve_images, lambda_images]
num_categories = len(categories)

# Define the number of columns for each category
num_columns = [1, 2, 3, 4, 5]

# Define the row and column names
# classical_prob_names = ['baart', 'blur', 'deriv2', 'foxgood', 'gravity', 'heat', 'phillips', 'shaw', 'wing']
classical_prob_names = ['deriv2']
# classical_prob_names = ['foxgood']


column_names = ['LocReg', 'GCV', 'L-Curve', 'DP', 'Lambda']

import matplotlib.pyplot as plt
import matplotlib.image as mpimg

# Create a 1x5 grid of subplots
fig, axes = plt.subplots(1, 5, figsize=(12, 12))
fig.patch.set_facecolor('white')

# Check if the lengths of column_names and image lists match
if len(column_names) != len(locreg_images) or \
   len(column_names) != len(gcv_images) or \
   len(column_names) != len(lcurve_images) or \
   len(column_names) != len(dp_images) or \
   len(column_names) != len(lambda_images):
    raise ValueError("The number of column names and images should match.")

# Loop through the columns (assuming classical_prob_names and column_names are appropriately defined)
for j, column_name in enumerate(column_names):
    # Get the image paths based on the column name
    if column_name == 'LocReg':
        image_path = locreg_images[j]
    elif column_name == 'GCV':
        image_path = gcv_images[j]
    elif column_name == 'L-Curve':
        image_path = lcurve_images[j]
    elif column_name == 'DP':
        image_path = dp_images[j]
    elif column_name == 'Lambda':
        image_path = lambda_images[j]
    
    # Load and display the image
    img = mpimg.imread(image_path)
    axes[j].imshow(img)
    axes[j].axis('off')

# Set column names to bold
for j, col in enumerate(column_names):
    axes[j].set_title(col, weight='bold')

# Add row labels
for i, row in enumerate(classical_prob_names):
    label = axes[i].text(-0.28, 0.5, row, rotation=0, size='large', transform=axes[i].transAxes)
    label.set_weight('bold')  # Set row names to bold

# Adjust layout
plt.tight_layout()

# Display the plot
plt.show()


# # Remove the empty last row
# fig.delaxes(axes[5, 0])
# fig.delaxes(axes[5, 1])
# fig.delaxes(axes[5, 2])
# fig.delaxes(axes[5, 3])
# fig.delaxes(axes[5, 4])

# Adjust layout
# plt.tight_layout(rect=[0, 0, 1, 0.95])  # Adjust rect parameter to control white space

# plt.subplots_adjust(hspace=0.2)  # Adjust vertical spacing between subplots
plt.tight_layout()

plt.savefig('testingstackedimages1023.png', dpi=600, transparent=True, bbox_inches='tight')

# Display the plot
plt.show()
