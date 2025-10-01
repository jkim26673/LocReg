# import matplotlib.pyplot as plt
# from mpl_toolkits.axes_grid1 import ImageGrid
# import os

# images_lowSNR=os.listdir("/Users/kimjosy/Downloads/LocReg/results/brain/noise_addition_exp/Sep1925/wass_score_plot")
# images_highSNR=os.listdir("/Users/kimjosy/Downloads/LocReg/results/brain/noise_addition_exp/Sep2625/wass_score_plot")
# images = images_lowSNR + images_highSNR
# fig = plt.figure(figsize=(4., 4.))
# grid = ImageGrid(fig, 111,  # similar to subplot(111)
#                  nrows_ncols=(2, 4),  # creates 2x2 grid of axes
#                  axes_pad=0.1,  # pad between axes in inch.
#                  )

# for ax, im in zip(grid, images):
#     # Iterating over the grid returns the Axes.
#     plt.plot(im)
#     # ax.imshow(im)

# plt.show()
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.image as mpimg
import os

# Directories
dir_lowSNR = "/Users/kimjosy/Downloads/LocReg/results/brain/noise_addition_exp/Sep1925/wass_score_plot"
dir_highSNR = "/Users/kimjosy/Downloads/LocReg/results/brain/noise_addition_exp/Sep2625/wass_score_plot"

# Collect file paths
images_lowSNR = [os.path.join(dir_lowSNR, f) for f in os.listdir(dir_lowSNR) if f.endswith((".png", ".jpg"))]
images_highSNR = [os.path.join(dir_highSNR, f) for f in os.listdir(dir_highSNR) if f.endswith((".png", ".jpg"))]
images = images_lowSNR + images_highSNR
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import matplotlib.image as mpimg

rows, cols = 2, 5
fig_width, fig_height = cols * 2.5, rows * 2.5 + 3.5  # extra height for lower colorbar
fig = plt.figure(figsize=(fig_width, fig_height))

# Image grid
grid = ImageGrid(
    fig, 111,
    nrows_ncols=(rows, cols),
    axes_pad=0.2,
)

# Plot images
for ax, im_path in zip(grid, images):
    data = mpimg.imread(im_path)
    im = ax.imshow(
        data,
        cmap="viridis",
        vmin=0,
        vmax=0.01,
        aspect="equal"
    )
    ax.axis("off")  # turn off axes, ticks, grids

# Horizontal colorbar centered under the grid
cbar_width = 0.6  # fraction of figure width
cbar_height = 0.03
cbar_left = (1 - cbar_width) / 2  # center horizontally
cbar_bottom = 0.02
cbar_ax = fig.add_axes([cbar_left, cbar_bottom, cbar_width, cbar_height])
cbar = fig.colorbar(im, cax=cbar_ax, orientation="horizontal")

# Turn off all colorbar decorations except label
cbar.ax.set_frame_on(False)
cbar.ax.tick_params(size=0, labelsize=10)
cbar.ax.xaxis.set_ticks_position('none')
cbar.ax.set_xlabel("Value", labelpad=5, fontsize=10)

# Remove whitespace around figure
plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

# plt.show()
plt.savefig("/Users/kimjosy/Downloads/LocReg/pub_figs/brain/stacked_brain_figs.png")



# rows, cols = 2, 5  # compact 2x5 layout

# fig = plt.figure(figsize=(cols * 4, rows * 4))
# grid = ImageGrid(fig, 111, nrows_ncols=(rows, cols), axes_pad=0.0)

# # Plot images
# for ax, im_path in zip(grid, images):
#     img = mpimg.imread(im_path)
#     ax.imshow(img, aspect="auto")
#     ax.axis("off")

# # Remove unused axes
# for ax in grid[len(images):]:
#     ax.remove()

# # Tight layout: kill all whitespace around and between images
# plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0, hspace=0)

# # Force axes to occupy full space (no subplot borders)
# for ax in grid:
#     pos = ax.get_position()
#     ax.set_position([pos.x0, pos.y0, pos.width * 1.01, pos.height * 1.01])

# plt.show()
