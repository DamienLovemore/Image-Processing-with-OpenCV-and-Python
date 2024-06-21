import pandas as pd
import numpy as np

from glob import glob

import cv2
import matplotlib.pyplot as plt

# Gets all the image paths
# Returns a list of paths matching a pathname pattern
dog_files = glob("./input/cat-and-dog/training_set/dogs/*.jpg")
cat_files = glob("./input/cat-and-dog/training_set/cats/*.jpg")
# print(dog_files) #See if it is working

# Gets an specific wanted image from all those set
dog_wanted = ""
for dog_name in dog_files:
    if "dog.117.jpg" in dog_name:
        dog_wanted = dog_name
        break

# How to read image files into data
img_mpl = plt.imread(dog_name)
img_cv2 = cv2.imread(dog_name)
# print(img_mpl) See if it is loaded
# print(img_mpl.shape, img_cv2.shape) #See if they are the same thing, by comparing their array sizes

# Flatten the large array into one dimensional. Them converts into a Pandas Series
data = pd.Series(img_mpl.flatten())

# Shows a histogram of color distributions
# A histogram with x going 50 by 50, with that title
data.plot(kind="hist", bins=50, title="Distribution of Pixels Values")
# Display all open figures (graphical representations)
# plt.show()

# Display an image
# Creates a figure (graphical graphic) and a set of subplots (sets the size of the image to be displayed)
fig, ax = plt.subplots(figsize=(10, 10))
# Prepare the image to be shown
ax.imshow(img_mpl)
# Remove the axis to show just the image
ax.axis("off")

# Height, width and three chanels (RGB)
# print(img_mpl.shape)

# Display RGB Channels of our image
fig2, ax2 = plt.subplots(1, 3, figsize=(15, 5))
# Display a figure (graphic) about the Red channel (cmap changes the color scale of the picture, gray, black and white, red and etc)
ax2[0].imshow(img_mpl[:, :, 0], cmap="Reds")
ax2[1].imshow(img_mpl[:, :, 1], cmap="Greens")
ax2[2].imshow(img_mpl[:, :, 2], cmap="Blues")
# Remove the axis in the figures
ax2[0].axis("off")
ax2[1].axis("off")
ax2[2].axis("off")
# Set the title of the figures
ax2[0].set_title("Red Channel")
ax2[1].set_title("Green Channel")
ax2[2].set_title("Blue Channel")

fig3, ax3 = plt.subplots(1, 2, figsize=(10, 5))
# By default OpenCV2 is in BGR not RGB
ax3[0].imshow(img_cv2)
ax3[1].imshow(img_mpl)
ax3[0].axis("off")
ax3[1].axis("off")
ax3[0].set_title("CV2 Image")
ax3[1].set_title("Matplotlib Image")

# Convert the color scheme to RGB
img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)
fig4, ax4 = plt.subplots()
ax4.imshow(img_cv2)
ax4.set_title("CV2 Image BGR to RGB")
ax4.axis("off")

# Image Manipulation

# Showing the original cat picture
img_cat = plt.imread(cat_files[4])
fig5, ax5 = plt.subplots(figsize=(8, 8))
ax5.imshow(img_cat)
ax5.set_title("Cat image")
ax5.axis("off")

# Showing the cat image in the gray scale
fig6, ax6 = plt.subplots(figsize=(8, 8))
img_gray_cat = cv2.cvtColor(img_cat, cv2.COLOR_RGB2GRAY)
ax6.imshow(img_gray_cat, cmap="Greys")
ax6.axis("off")
ax6.set_title("Grey cat image")

# Resizing and Scaling image
# Reduce the image to 1/4 of its original size
img_reduced = cv2.resize(img_cat, None, fx=0.25, fy=0.25)
fig7, ax7 = plt.subplots(figsize=(2, 2))
ax7.imshow(img_reduced)
ax7.set_title("Cat picture 1/4")
ax7.axis("off")
# Compare the sizes
print(img_cat.shape)
print(img_reduced.shape)
print()

# Reduce the image to a specific size
img_reduced2 = cv2.resize(img_cat, (100, 200))
fig8, ax8 = plt.subplots(figsize=(8, 8))
ax8.imshow(img_reduced2)
ax8.set_title("Cat picture 100 X 200")
ax8.axis("off")

# Upscale an image
# When upscaling an image you have to choose the upscale to fill the gaps/stretch the picture
img_upscaled = cv2.resize(img_cat, (5000, 5000), interpolation=cv2.INTER_CUBIC)
fig9, ax9 = plt.subplots(figsize=(8, 8))
ax9.imshow(img_upscaled)
ax9.set_title("Cat picture Upscaled")
ax9.axis("off")
# Compare the sizes... again
print(img_cat.shape)
print(img_upscaled.shape)
print()

# Sharpening and Blurring
kernel_sharpening = np.array([[-1, -1, -1],
                              [-1, 9, -1],
                              [-1, -1, -1]
                              ])

#Applies the image filter (sharpen)
sharpened_img = cv2.filter2D(img_cat, -1, kernel_sharpening)
fig10, ax10 = plt.subplots(figsize=(8, 8))
ax10.imshow(sharpened_img)
ax10.set_title("Cat picture Sharpened")
ax10.axis("off")

kernel_3X3 = np.ones((3, 3), np.float32) / 10 # A matriz 3X3 with 0.33 in it
blurred = cv2.filter2D(img_cat, -1, kernel_3X3)
fig11, ax11 = plt.subplots(figsize=(8, 8))
ax11.imshow(blurred)
ax11.set_title("Cat picture Blurred")
ax11.axis("off")


#Save an image (after you made transformations this might me useful; don`t forget the extension)
plt.imsave("image_cat_blurred.png", blurred)
plt.imsave("image_cat_sharpened.png", sharpened_img)
plt.imsave("image_cute_dog.png", img_mpl)
#With cv2 you used cv2.imwrite and the same parameters as plt.

#Shows all the figures and hold the terminal in this line while the figures are not closed
plt.show()
