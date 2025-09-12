from PIL import Image
import os
import re

from PIL import Image
import os
import re

# def crop_left_side(img, width_per_section):
#     """Crop the left side of an image."""
#     return img.crop((0, 0, width_per_section, img.height))

# def create_image_grid(image_folder, grid_size=(5, 5), image_size=(2000, 600)):
#     # Define the regex pattern to capture the parts of the filename
#     # pattern = re.compile(r'Simulation(\d+)_Sigma(\d+)_RPS(\d+)\.png')
#     pattern = re.compile(r'Simulation0_Sigma(\d+)_RPS(\d+)\.png')

#     output_file=f'{image_folder}.png'
#     # List and sort the image files based on the numeric parts
#     image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if pattern.match(f)]
#     image_files.sort(key=lambda f: tuple(map(int, pattern.search(f).groups())))

#     # Check that we have the correct number of images
#     if len(image_files) != grid_size[0] * grid_size[1]:
#         raise ValueError(f"Number of images should be {grid_size[0] * grid_size[1]}")
    
#     # Calculate width per section if images are horizontally split into 3 parts
#     width_per_image = image_size[0]
#     # width_per_section = width_per_image // 3
#     width_per_section = width_per_image // 3
#     # Create a blank canvas for the grid
#     grid_width = grid_size[0] * width_per_section
#     grid_height = grid_size[1] * image_size[1]
    
#     grid_image = Image.new('RGB', (grid_width, grid_height), (255, 255, 255))  # White background
    
#     # Arrange images in grid
#     for index, image_file in enumerate(image_files):
#         img = Image.open(image_file)
#         img = img.resize(image_size)  # Resize if necessary
#         # left_img = crop_left_side(img, width_per_section)
#         left_img = img
#         row = index // grid_size[0]
#         col = index % grid_size[0]
        
#         # Calculate position for placing the image
#         x_position = col * width_per_section
#         y_position = row * image_size[1]
        
#         # Paste the image into the grid
#         grid_image.paste(left_img, (x_position, y_position))
    
#     # Save the resulting image
#     grid_image.save(output_file)
#     print(f"Grid image saved as {output_file}1")

from PIL import Image
import os
import re

def create_image_grid(image_folder, grid_size=(5, 5), image_size=(600, 600)):
    pattern = re.compile(r'Simulation0_Sigma(\d+)_RPS(\d+)\.png')

    output_file = f'{image_folder}.png'
    image_files = [os.path.join(image_folder, f) for f in os.listdir(image_folder) if pattern.match(f)]
    image_files.sort(key=lambda f: tuple(map(int, pattern.search(f).groups())))

    if len(image_files) != grid_size[0] * grid_size[1]:
        raise ValueError(f"Number of images should be {grid_size[0] * grid_size[1]}")

    width, height = image_size
    grid_width = grid_size[0] * width
    grid_height = grid_size[1] * height
    grid_image = Image.new('RGB', (grid_width, grid_height), (255, 255, 255))

    for index, image_file in enumerate(image_files):
        img = Image.open(image_file)
        img = img.resize(image_size)  # Resize to square or balanced shape

        row = index // grid_size[0]
        col = index % grid_size[0]
        x_position = col * width
        y_position = row * height

        grid_image.paste(img, (x_position, y_position))

    grid_image.save(output_file)
    print(f"Grid image saved as {output_file}")

# # Example usage
# # image_folder = '/home/kimjosy/LocReg_Regularization-1/SimulationSets/MRR/SpanRegFig/MRR_1D_LocReg_Comparison_2024-09-17_SNR_100_lamini_LCurve_dist_narrowL_broadR_15iter'
# # narrowL_broadR1000 = "/home/kimjosy/LocReg_Regularization-1/SimulationsSets/MRR/SpanRegFig/est_table_SNR1000_iter50_lamini_LCurve_dist_narrowL_broadR_400maxiter_50NR_07Oct24.pkl"
# # narrowL_broadR1000 = "/home/kimjosy/LocReg_Regularization-1/SimulationSets/MRR/SpanRegFig/MRR_1D_LocReg_Comparison_2024-10-06_SNR_1000_lamini_LCurve_dist_narrowL_broadR_400maxiter_50NR"
# narrowL_broadR1000 = "/home/kimjosy/LocReg_Regularization-1/SimulationSets/MRR/SpanRegFig/MRR_1D_LocReg_Comparison_2024-10-08_SNR_1000_lamini_LCurve_dist_narrowL_broadR_SNR1000_1iter"
# # narrowL_broadR300 = "/home/kimjosy/LocReg_Regularization-1/SimulationsSets/MRR/SpanRegFig/est_table_SNR300_iter50_lamini_LCurve_dist_narrowL_broadR_400maxiter_50NR_06Oct24.pkl"
# # narrowL_broadR300 = "/home/kimjosy/LocReg_Regularization-1/SimulationSets/MRR/SpanRegFig/MRR_1D_LocReg_Comparison_2024-10-06_SNR_300_lamini_LCurve_dist_narrowL_broadR_400maxiter_50NR"
# narrowL_broadR300 = "/home/kimjosy/LocReg_Regularization-1/SimulationSets/MRR/SpanRegFig/MRR_1D_LocReg_Comparison_2024-10-08_SNR_300_lamini_LCurve_dist_narrowL_broadR_SNR300_1iter"
# # broadL_narrowR1000 = "/home/kimjosy/LocReg_Regularization-1/SimulationsSets/MRR/SpanRegFig/est_table_SNR1000_iter50_lamini_LCurve_dist_broadL_narrowR_08Oct24.pkl"
# broadL_narrowR1000 = "/home/kimjosy/LocReg_Regularization-1/SimulationSets/MRR/SpanRegFig/MRR_1D_LocReg_Comparison_2024-10-07_SNR_1000_lamini_LCurve_dist_broadL_narrowR_400maxiter_50NR"
# # broadL_narrowR300 = "/home/kimjosy/LocReg_Regularization-1/SimulationsSets/MRR/SpanRegFig/est_table_SNR300_iter50_lamini_LCurve_dist_broadL_narrowR_50NR_08Oct24.pkl"
# broadL_narrowR300 = "/home/kimjosy/LocReg_Regularization-1/SimulationSets/MRR/SpanRegFig/MRR_1D_LocReg_Comparison_2024-10-08_SNR_300_lamini_LCurve_dist_broadL_narrowR_50NR"



# narrowL_broadR1000 = "/home/kimjosy/LocReg_Regularization-1/SimulationSets/MRR/SpanRegFig/MRR_1D_LocReg_Comparison_2024-10-08_SNR_1000_lamini_LCurve_dist_narrowL_broadR_SNR1000_1iter"
# narrowL_broadR300 = "/home/kimjosy/LocReg_Regularization-1/SimulationSets/MRR/SpanRegFig/MRR_1D_LocReg_Comparison_2024-10-08_SNR_300_lamini_LCurve_dist_narrowL_broadR_SNR300_1iter"
# broadL_narrowR1000 = "/home/kimjosy/LocReg_Regularization-1/SimulationSets/MRR/SpanRegFig/MRR_1D_LocReg_Comparison_2024-10-07_SNR_1000_lamini_LCurve_dist_broadL_narrowR_400maxiter_50NR"
# broadL_narrowR300 = "/home/kimjosy/LocReg_Regularization-1/SimulationSets/MRR/SpanRegFig/MRR_1D_LocReg_Comparison_2024-10-08_SNR_300_lamini_LCurve_dist_broadL_narrowR_50NR"


# NL_BR1000 = "/home/kimjosy/LocReg_Regularization-1/SimulationSets/MRR/SpanRegFig/iterationval1801e5/MRR_1D_LocReg_Comparison_2024-10-15_SNR_1000_lamini_LCurve_dist_narrowL_broadR_parallel_nsim50_SNR_1000_new"
# BL_NR1000 = "/home/kimjosy/LocReg_Regularization-1/SimulationSets/MRR/SpanRegFig/iterationval1801e5/MRR_1D_LocReg_Comparison_2024-10-15_SNR_1000_lamini_LCurve_dist_broadL_narrowR_parallel_nsim50_SNR_1000"
# NL_BR300 = "/home/kimjosy/LocReg_Regularization-1/SimulationSets/MRR/SpanRegFig/iterationval1801e5/MRR_1D_LocReg_Comparison_2024-10-15_SNR_300_lamini_LCurve_dist_narrowL_broadR_parallel_nsim50_SNR_300"
# BL_NR300 = "/home/kimjosy/LocReg_Regularization-1/SimulationSets/MRR/SpanRegFig/iterationval1801e5/MRR_1D_LocReg_Comparison_2024-10-15_SNR_300_lamini_LCurve_dist_broadL_narrowR_parallel_nsim50_SNR_300"
# NL_BR50 = "/home/kimjosy/LocReg_Regularization-1/SimulationSets/MRR/SpanRegFig/iterationval1801e5/MRR_1D_LocReg_Comparison_2024-10-15_SNR_50_lamini_LCurve_dist_narrowL_broadR_parallel_nsim50_SNR_50"
# BL_NR50 = "/home/kimjosy/LocReg_Regularization-1/SimulationSets/MRR/SpanRegFig/iterationval1801e5/MRR_1D_LocReg_Comparison_2024-10-15_SNR_50_lamini_LCurve_dist_broadL_narrowR_parallel_nsim50_SNR_50"


# NL_BR1000 = "/home/kimjosy/LocReg_Regularization-1/SimulationSets/MRR/SpanRegFig/iterationval200_1e2/MRR_1D_LocReg_Comparison_2024-10-14_SNR_1000_lamini_LCurve_dist_narrowL_broadR_parallel_nsim50_SNR_1000"
# BL_NR1000 = "/home/kimjosy/LocReg_Regularization-1/SimulationSets/MRR/SpanRegFig/iterationval200_1e2/MRR_1D_LocReg_Comparison_2024-10-14_SNR_1000_lamini_LCurve_dist_broadL_narrowR_parallel_nsim50_SNR_1000"
# NL_BR300 = "/home/kimjosy/LocReg_Regularization-1/SimulationSets/MRR/SpanRegFig/iterationval200_1e2/MRR_1D_LocReg_Comparison_2024-10-14_SNR_300_lamini_LCurve_dist_narrowL_broadR_parallel_nsim50_SNR_300"
# BL_NR300 = "/home/kimjosy/LocReg_Regularization-1/SimulationSets/MRR/SpanRegFig/iterationval200_1e2/MRR_1D_LocReg_Comparison_2024-10-14_SNR_300_lamini_LCurve_dist_broadL_narrowR_parallel_nsim50_SNR_300"
# NL_BR50 = "/home/kimjosy/LocReg_Regularization-1/SimulationSets/MRR/SpanRegFig/iterationval200_1e2/MRR_1D_LocReg_Comparison_2024-10-14_SNR_50_lamini_LCurve_dist_narrowL_broadR_parallel_nsim50_SNR_50"
# BL_NR50 = "/home/kimjosy/LocReg_Regularization-1/SimulationSets/MRR/SpanRegFig/iterationval200_1e2/MRR_1D_LocReg_Comparison_2024-10-14_SNR_50_lamini_LCurve_dist_broadL_narrowR_parallel_nsim50_SNR_50"

NL_BR10001018 = "/home/kimjosy/LocReg_Regularization-1/SimulationSets/MRR/SpanRegFig/MRR_1D_LocReg_Comparison_2024-10-17_SNR_1000_lamini_LCurve_dist_narrowL_broadR_parallel_nsim50_SNR_1000"
NL_BR10001017 = "/home/kimjosy/LocReg_Regularization-1/SimulationSets/MRR/SpanRegFig/MRR_1D_LocReg_Comparison_2024-10-17_SNR_1000_lamini_LCurve_dist_narrowL_broadR_parallel_nsim50_SNR_1000"
NL_BR3001018= "/home/kimjosy/LocReg_Regularization-1/SimulationSets/MRR/SpanRegFig/MRR_1D_LocReg_Comparison_2024-10-18_SNR_300_lamini_LCurve_dist_narrowL_broadR_parallel_nsim50_SNR_300"
NL_BR5001018 = "/home/kimjosy/LocReg_Regularization-1/SimulationSets/MRR/SpanRegFig/MRR_1D_LocReg_Comparison_2024-10-18_SNR_50_lamini_LCurve_dist_narrowL_broadR_parallel_nsim50_SNR_50"

BL_NR10001018 = "/home/kimjosy/LocReg_Regularization-1/SimulationSets/MRR/SpanRegFig/MRR_1D_LocReg_Comparison_2024-10-18_SNR_1000_lamini_LCurve_dist_broadL_narrowR_parallel_nsim50_SNR_1000"
BL_NR3001018 = "/home/kimjosy/LocReg_Regularization-1/SimulationSets/MRR/SpanRegFig/MRR_1D_LocReg_Comparison_2024-10-18_SNR_300_lamini_LCurve_dist_broadL_narrowR_parallel_nsim50_SNR_300"
BL_NR501018 = "/home/kimjosy/LocReg_Regularization-1/SimulationSets/MRR/SpanRegFig/MRR_1D_LocReg_Comparison_2024-10-18_SNR_50_lamini_LCurve_dist_broadL_narrowR_parallel_nsim50_SNR_50"

NL_BR10001022W = "/home/kimjosy/LocReg_Regularization-1/SimulationSets/MRR/SpanRegFig/MRR_1D_LocReg_Comparison_2024-10-22_SNR_1000_lamini_LCurve_dist_narrowL_broadR_parallel_nsim10_SNR_1000_errtype_Wass. Score"
NL_BR10001022L2 = "/home/kimjosy/LocReg_Regularization-1/SimulationSets/MRR/SpanRegFig/MRR_1D_LocReg_Comparison_2024-10-22_SNR_1000_lamini_LCurve_dist_narrowL_broadR_parallel_nsim10_SNR_1000_errtype_Rel. L2 Norm"



# BL_NR1000 = "/home/kimjosy/LocReg_Regularization-1/SimulationSets/MRR/SpanRegFig/MRR_1D_LocReg_Comparison_2024-10-28_SNR_1000_lamini_LCurve_dist_broadL_narrowR_parallel_nsim50_SNR_1000_errtype_Wass. Score"
# BL_NR300 = "/home/kimjosy/LocReg_Regularization-1/SimulationSets/MRR/SpanRegFig/MRR_1D_LocReg_Comparison_2024-10-28_SNR_300_lamini_LCurve_dist_broadL_narrowR_parallel_nsim50_SNR_300_errtype_Wass. Score"
# BL_NR50 = "/home/kimjosy/LocReg_Regularization-1/SimulationSets/MRR/SpanRegFig/MRR_1D_LocReg_Comparison_2024-10-28_SNR_50_lamini_LCurve_dist_broadL_narrowR_parallel_nsim50_SNR_50_errtype_Wass. Score"

# NL_BR1000 = "/home/kimjosy/LocReg_Regularization-1/SimulationSets/MRR/SpanRegFig/MRR_1D_LocReg_Comparison_2024-10-26_SNR_1000_lamini_LCurve_dist_narrowL_broadR_parallel_nsim50_SNR_1000_errtype_Wass. Score"
# NL_BR300 = "/home/kimjosy/LocReg_Regularization-1/SimulationSets/MRR/SpanRegFig/MRR_1D_LocReg_Comparison_2024-10-26_SNR_300_lamini_LCurve_dist_narrowL_broadR_parallel_nsim50_SNR_300_errtype_Wass. Score"
# NL_BR50 = "/home/kimjosy/LocReg_Regularization-1/SimulationSets/MRR/SpanRegFig/MRR_1D_LocReg_Comparison_2024-10-26_SNR_50_lamini_LCurve_dist_narrowL_broadR_parallel_nsim50_SNR_50_errtype_Wass. Score"

im_3by3 = "/home/kimjosy/LocReg_Regularization-1/SimulationSets/MRR/SpanRegFig/MRR_1D_LocReg_Comparison_2024-11-14_SNR_1000_lamini_LCurve_dist_narrowL_broadR_parallel_nsim100_SNR_1000_errtype_Wass. Score"

BL_NR1000 = "/home/kimjosy/LocReg_Regularization-1/SimulationSets/MRR/SpanRegFig/MRR_1D_LocReg_Comparison_2024-11-22_SNR_1000_lamini_LCurve_dist_broadL_narrowR_parallel_nsim50_SNR_1000_errtype_Wass. Score"
BL_NR300 = "/home/kimjosy/LocReg_Regularization-1/SimulationSets/MRR/SpanRegFig/MRR_1D_LocReg_Comparison_2024-11-22_SNR_300_lamini_LCurve_dist_broadL_narrowR_parallel_nsim50_SNR_300_errtype_Wass. Score"
BL_NR50 = "/home/kimjosy/LocReg_Regularization-1/SimulationSets/MRR/SpanRegFig/MRR_1D_LocReg_Comparison_2024-11-22_SNR_50_lamini_LCurve_dist_broadL_narrowR_parallel_nsim50_SNR_50_errtype_Wass. Score"

NL_BR1000 = "/home/kimjosy/LocReg_Regularization-1/SimulationSets/MRR/SpanRegFig/MRR_1D_LocReg_Comparison_2024-11-22_SNR_1000_lamini_LCurve_dist_narrowL_broadR_parallel_nsim50_SNR_1000_errtype_Wass. Score"
NL_BR300 = "/home/kimjosy/LocReg_Regularization-1/SimulationSets/MRR/SpanRegFig/MRR_1D_LocReg_Comparison_2024-11-21_SNR_300_lamini_LCurve_dist_narrowL_broadR_parallel_nsim50_SNR_300_errtype_Wass. Score"
NL_BR50 = "/home/kimjosy/LocReg_Regularization-1/SimulationSets/MRR/SpanRegFig/MRR_1D_LocReg_Comparison_2024-11-21_SNR_50_lamini_LCurve_dist_narrowL_broadR_parallel_nsim50_SNR_50_errtype_Wass. Score"

LocReg_Deriv = "/home/kimjosy/LocReg_Regularization-1/SimulationSets/MRR/SpanRegFig/LocRegDeriviative/MRR_1D_LocReg_Comparison_2025-01-31_SNR_1000_lamini_LCurve_dist_narrowL_broadR_parallel_nsim1_SNR_1000_errtype_Wass. Score"

file_name = LocReg_Deriv

# image = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\SimulationSets\MRR\SpanRegFig\MRR_1D_LocReg_Comparison_May0225_nsim10\savefolder_LocReg"
# image = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\SimulationSets\MRR\SpanRegFig\MRR_1D_LocReg_Comparison_May0225_nsim10\savefolder_LocReg1D"
image = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\SimulationSets\MRR\SpanRegFig\MRR_1D_LocReg_Comparison_May0225_nsim10\savefolder_LocReg2D"
file_name = image

create_image_grid(file_name)
