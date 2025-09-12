from PIL import Image
import os
import re

def create_image_grid(image_folder, deriv, nsim, grid_size=(5, 5), image_size=(800, 800)):
    if deriv == "0":
        pattern = re.compile(rf'Simulation{nsim}_Sigma(\d+)_RPS(\d+)_0th_Derivative\.png')
    elif deriv == "1":
        pattern = re.compile(rf'Simulation{nsim}_Sigma(\d+)_RPS(\d+)_1st_Derivative\.png')
    elif deriv == "2":
        pattern = re.compile(rf'Simulation{nsim}_Sigma(\d+)_RPS(\d+)_2nd_Derivative\.png')
    output_file = f'{image_folder}\Derivative{deriv}_nsim{nsim}.png'
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




NL_BR10001018 = "/home/kimjosy/LocReg_Regularization-1/SimulationSets/MRR/SpanRegFig/MRR_1D_LocReg_Comparison_2024-10-17_SNR_1000_lamini_LCurve_dist_narrowL_broadR_parallel_nsim50_SNR_1000"
NL_BR10001017 = "/home/kimjosy/LocReg_Regularization-1/SimulationSets/MRR/SpanRegFig/MRR_1D_LocReg_Comparison_2024-10-17_SNR_1000_lamini_LCurve_dist_narrowL_broadR_parallel_nsim50_SNR_1000"
NL_BR3001018= "/home/kimjosy/LocReg_Regularization-1/SimulationSets/MRR/SpanRegFig/MRR_1D_LocReg_Comparison_2024-10-18_SNR_300_lamini_LCurve_dist_narrowL_broadR_parallel_nsim50_SNR_300"
NL_BR5001018 = "/home/kimjosy/LocReg_Regularization-1/SimulationSets/MRR/SpanRegFig/MRR_1D_LocReg_Comparison_2024-10-18_SNR_50_lamini_LCurve_dist_narrowL_broadR_parallel_nsim50_SNR_50"

BL_NR10001018 = "/home/kimjosy/LocReg_Regularization-1/SimulationSets/MRR/SpanRegFig/MRR_1D_LocReg_Comparison_2024-10-18_SNR_1000_lamini_LCurve_dist_broadL_narrowR_parallel_nsim50_SNR_1000"
BL_NR3001018 = "/home/kimjosy/LocReg_Regularization-1/SimulationSets/MRR/SpanRegFig/MRR_1D_LocReg_Comparison_2024-10-18_SNR_300_lamini_LCurve_dist_broadL_narrowR_parallel_nsim50_SNR_300"
BL_NR501018 = "/home/kimjosy/LocReg_Regularization-1/SimulationSets/MRR/SpanRegFig/MRR_1D_LocReg_Comparison_2024-10-18_SNR_50_lamini_LCurve_dist_broadL_narrowR_parallel_nsim50_SNR_50"

NL_BR10001022W = "/home/kimjosy/LocReg_Regularization-1/SimulationSets/MRR/SpanRegFig/MRR_1D_LocReg_Comparison_2024-10-22_SNR_1000_lamini_LCurve_dist_narrowL_broadR_parallel_nsim10_SNR_1000_errtype_Wass. Score"
NL_BR10001022L2 = "/home/kimjosy/LocReg_Regularization-1/SimulationSets/MRR/SpanRegFig/MRR_1D_LocReg_Comparison_2024-10-22_SNR_1000_lamini_LCurve_dist_narrowL_broadR_parallel_nsim10_SNR_1000_errtype_Rel. L2 Norm"



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
# image = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\SimulationSets\MRR\SpanRegFig\MRR_1D_LocReg_Comparison_May0225_nsim10\savefolder_LocReg2D"
# file_folder = image

# image_folder = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\SimulationSets\MRR\SpanRegFig\MRR_1D_LocReg_Comparison_2025-05-23_nsim2"
# image_folder = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\SimulationSets\MRR\SpanRegFig\MRR_1D_LocReg_Comparison_2025-05-23_nsim10_SNR1000"
image_folder = r"C:\Users\kimjosy\Downloads\LocReg_Regularization-1\SimulationSets\MRR\SpanRegFig\MRR_1D_LocReg_Comparison_2025-05-23_nsim10_SNR1000_1\moreNRs"
create_image_grid(image_folder,  deriv = '0',nsim=4, grid_size=(5, 5), image_size=(1000, 800))
create_image_grid(image_folder, deriv = '1', nsim=4,grid_size=(5, 5), image_size=(1000, 800))
create_image_grid(image_folder, deriv = '2',nsim=4,grid_size=(5, 5), image_size=(1000, 800))

# create_image_grid(file_folder)