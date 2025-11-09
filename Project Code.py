import cv2
import matplotlib.pyplot as plt
import numpy as np

# Show image using matplotlib
def showImg(img1, title="Image", cmap='gray'):
    plt.imshow(img1, cmap=cmap)
    plt.title(title)
    plt.axis('off')
    plt.show()

#Original image RGB Colors
original_image = cv2.imread('Fabric (4).jpg')
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

#Convert to Grayscale image
gray_image = cv2.cvtColor(original_image, cv2.COLOR_RGB2GRAY)
    
# Apply Gaussian blur
gaussian_image = cv2.GaussianBlur(gray_image, (21, 21), 0)

# Initial Otsu threshold
thresh_value1, thres1 = cv2.threshold(gaussian_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

# Slightly increase the threshold
thresh_value1 = thresh_value1 * 1.08
thresh_value1 = int(np.round(thresh_value1))

print("Adjusted Threshold:", thresh_value1)

# Apply threshold
_, thres3 = cv2.threshold(gaussian_image, thresh_value1, 255, cv2.THRESH_BINARY)

# Count pixels
count_white = np.count_nonzero(thres3 == 255)
count_black = np.count_nonzero(thres3 == 0)

print("White pixels:", count_white)
print("Black pixels:", count_black)

# If white > black, re-adjust threshold
if count_white > count_black:
    thresh_value1, thres1 = cv2.threshold(gaussian_image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    thresh_value1 = thresh_value1 * 0.95
    thresh_value1 = int(np.round(thresh_value1))
    _, thres3 = cv2.threshold(gaussian_image, thresh_value1, 255, cv2.THRESH_BINARY_INV)

# Morphological operations
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
opening_img = cv2.morphologyEx(thres3, cv2.MORPH_OPEN, kernel, iterations=2)
dilate_img = cv2.morphologyEx(opening_img, cv2.MORPH_DILATE, kernel, iterations=3)

# Contour detection function
def draw_selected_contours(thresh_img, original_img, area_factor=1.0, peri_factor=1.0):
    contours_data = cv2.findContours(thresh_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = contours_data[0] if len(contours_data) == 2 else contours_data[1]

    img_copy = original_img.copy()

    if not contours:
        return img_copy, []

    # Sort and pick largest two
    sorted_contours = sorted(contours, key=cv2.contourArea, reverse=True)
    largest = sorted_contours[:2]

    # Calculate thresholds
    areas = [cv2.contourArea(c) for c in contours]
    perimeters = [cv2.arcLength(c, True) for c in contours]

    area_thresh = np.mean(areas) + area_factor * np.std(areas)
    peri_thresh = np.mean(perimeters) + peri_factor * np.std(perimeters)

    # Filter remaining
    remaining = sorted_contours[2:]
    filtered = [
        cnt for cnt in remaining
        if cv2.contourArea(cnt) > area_thresh and cv2.arcLength(cnt, True) > peri_thresh
    ]

    final_contours = largest + filtered
    result = cv2.drawContours(img_copy, final_contours, -1, (255, 0, 0), 2)

    return result, final_contours

# Draw and label
draw_contours, final_contours = draw_selected_contours(dilate_img, original_image, area_factor=3, peri_factor=1)

if len(final_contours)>0:
    label="{Defect detection}"
else:
    label="{Not Defect detection good fabric}"

# Plot all results
plt.figure(figsize=(12, 10))

plt.subplot(2, 3, 1)
plt.title("Original Image", color='orange')
plt.imshow(original_image)
plt.axis('off')

plt.subplot(2, 3, 2)
plt.title("Gray Image", color='orange')
plt.imshow(gray_image, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 3)
plt.title("Gaussian Blur", color='orange')
plt.imshow(gaussian_image, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 4)
plt.title("Thresholded Image", color='orange')
plt.imshow(thres3, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 5)
plt.title("Dilated Image", color='orange')
plt.imshow(dilate_img, cmap='gray')
plt.axis('off')

plt.subplot(2, 3, 6)
plt.title("Final Review: " + label, color='red')
plt.imshow(draw_contours)
plt.axis('off')

plt.show()
