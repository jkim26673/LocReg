import os
from PIL import Image
import matplotlib.pyplot as plt

# Path to the directory containing the images
image_directory = "/Users/steveh/Downloads/NIH 23-24/LocReg_Python/MRR figures 1023 no spanreg"
image_directory =  "/Users/steveh/Downloads/NIH 23-24/LocReg_Python/figure_data/MRR data/11_21_23 compare old expression and new expression/11_21_23_MRR_Simulation_Comparing_Chuan's_Old_Expression_With_New_Expression"

# Get a list of image file names in the directory and sort them
image_files = sorted([f for f in os.listdir(image_directory) if f.endswith(".png")])

# Define the number of rows and columns for the grid
num_rows = 5
num_columns = 5

# Calculate the number of images to display
total_images = num_rows * num_columns

# Create a wider grid of subplots without gridlines and larger figures
fig, axes = plt.subplots(num_rows, num_columns, figsize=(15, 15))  # Adjust the figsize as needed

# Loop through rows and columns
for i in range(num_rows):
    for j in range(num_columns):
        index = i * num_columns + j
        if index < total_images:
            # Open and display the image without gridlines
            image_path = os.path.join(image_directory, image_files[index])
            img = Image.open(image_path)
            
            # Get the dimensions of the original image
            width, height = img.size
            
            cropped_img = img
            # # Crop the left half of the image
            # cropped_img = img.crop((0, 0, width // 2, height))
            
            axes[i, j].imshow(cropped_img)
            axes[i, j].axis('off')  # Turn off axis (gridlines and labels)

# # Set custom column labels and center them
# column_labels = ["1st Peak Separation", "2nd Peak Separation", "3rd Peak Separation", "4th Peak Separation", "5th Peak Separation"]

# # Set column labels
# for j, col_label in enumerate(column_labels):
#     axes[0, j].set_title(col_label, fontweight='bold', ha='center', va='center')

# # Set custom row labels with adjusted padding and moved to the right
# row_labels = ["1st Gaussian Sigma", "2nd Gaussian Sigma", "3rd Gaussian Sigma", "4th Gaussian Sigma", "5th Gaussian Sigma"]

# for i, ax in enumerate(axes[:, 0]):
#     ax.annotate(row_labels[i], xy=(0.5, 0.5), xytext=(70, 0),
#                 xycoords=axes[i, 0].yaxis.label, textcoords='offset points', fontweight='bold',
#                 size='medium', ha='center', va='center')

# Adjust spacing between subplots
plt.tight_layout()

# Show the figure
plt.savefig("/Users/steveh/Downloads/NIH 23-24/LocReg_Python/SpanReg_paper_figure_11_21.png")
plt.show()
