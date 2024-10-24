# Implementation-of-Erosion-and-Dilation
## Aim
To implement Erosion and Dilation using Python and OpenCV.
## Software Required
1. Anaconda - Python 3.7
2. OpenCV
## Algorithm:
### Step 1:
Import the necessary libraries, such as OpenCV and NumPy.

### Step 2:
Create a black image using NumPy and add text to the image using the cv2.putText() function.

### Step 3:
Create a kernel for morphological operations using cv2.getStructuringElement() or np.ones().

### Step 4:
Perform erosion and dilation using cv2.erode() and cv2.dilate() respectively on the image.

### Step 5:
Display the original image, the eroded image, and the dilated image using matplotlib.pyplot.imshow().



 
## Program:

``` Python
import cv2
import numpy as np
from matplotlib import pyplot as plt

# Create a black image
img = np.zeros((100, 400), dtype='uint8')

# Set font and add text to the image
font = cv2.FONT_HERSHEY_SIMPLEX
cv2.putText(img, 'TheAILEarner', (5, 70), font, 2, (255), 5, cv2.LINE_AA)

# Create kernels
kernel = np.ones((5, 5), np.uint8)
kernel1 = cv2.getStructuringElement(cv2.MORPH_CROSS, (7, 7))

# Apply erosion and dilation
image_erode = cv2.erode(img, kernel1)
image_dilate = cv2.dilate(img, kernel1)

# Display the original and processed images
plt.subplot(1, 3, 1), plt.imshow(img, 'gray'), plt.title('Original')
plt.subplot(1, 3, 2), plt.imshow(image_erode, 'gray'), plt.title('Eroded')
plt.subplot(1, 3, 3), plt.imshow(image_dilate, 'gray'), plt.title('Dilated')
plt.show()

```
## Output:
![Screenshot 2024-10-24 140703](https://github.com/user-attachments/assets/ad5bd752-d673-441a-895a-a58ecbc4e8fb)

## Result
Thus the generated text image is eroded and dilated using python and OpenCV.
