# this code is cool because it uses techniques like otsu's method
# or CCA for finding licence plates

# in other files I'll proceed to use ML to find license plates, but this one is fun start
from skimage.io import imread
from skimage.filters import threshold_otsu
from skimage import measure
from skimage.measure import regionprops
import matplotlib.pyplot as plt
import matplotlib.patches as patches


# read image in grey scale
orig_im = imread("car.jpeg")
grey_im = imread("car.jpeg", as_gray=True)

# count threshold value using otsu's method
threshold_val = threshold_otsu(grey_im)
# split image to background and foreground
bin_im = grey_im > threshold_val


fig, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=(20, 5))
ax1.imshow(orig_im)
ax2.imshow(grey_im, cmap="gray")
ax3.imshow(bin_im, cmap="gray")
ax4.imshow(orig_im);

coords = []
objects = []
# in russia license plates are usually 520x112 mm
ratio = 520 / 112
# connected component analys is to identify all the connected regions
# (labels connected regions of an integer array)
label_im = measure.label(bin_im)
# regionprops creates a list of properties of all the labelled regions
for region in regionprops(label_im):
    # skip small rects
    if region.area < 100:
        continue

    y_left, x_left, y_right, x_right = region.bbox
    region_width = x_right - x_left
    region_height = y_right - y_left
    # checks if the region looks like a license plate
    if region_width / region_height > ratio * 0.8 and region_width / region_height < ratio * 1.2:
        # draws red rectangle over current region
        coords.append((x_left, y_left, x_right, y_right))
        objects.append(bin_im[y_left:y_right, x_left:x_right])
        rectBorder = patches.Rectangle((x_left, y_left), region_width, region_height, edgecolor="red", linewidth=2, fill=False)
        ax4.add_patch(rectBorder)
    


plt.show()
