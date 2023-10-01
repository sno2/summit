import cv2
import numpy as np
import os

def resize_image(image, target_width):
    # Get the current dimensions
    height, width = image.shape[:2]

    # Calculate the ratio to maintain aspect ratio
    aspect_ratio = target_width / float(width)
    target_height = int(aspect_ratio * height)

    # Resize the image
    resized_image = cv2.resize(image, (target_width, target_height))

    return resized_image

def normalize_temperatures(image):
    # Split the image into its color channels
    b, g, r = cv2.split(image)

    avg_b = np.mean(b)
    avg_g = np.mean(g)
    avg_r = np.mean(r)

    print(avg_b, avg_g, avg_r)

    if avg_r - 7 > max(avg_g, avg_b):
        pass
    else:
        return image, False

    # Decrease the red and green channels to cool down the colors
    cool_r = r * 0.5
    cool_g = g * 0.7  # Adjust this value to control the amount of cooling for green
    cool_b = b * 1.5  # Increase the blue channel

    # Merge the modified channels back together
    cool_image = cv2.merge((cool_b.astype(np.uint8), cool_g.astype(np.uint8), cool_r.astype(np.uint8)))

    return cool_image, True

def find_large_and_small_boxes(contours):
    # Sort contours by area in descending order
    contours = sorted(contours, key=cv2.contourArea, reverse=True)

    # Assume the first contour is the larger box
    larger_box_contour = contours[0]

    # Initialize a list to store smaller box contours
    smaller_box_contours = []

    # Analyze the contours to identify smaller boxes
    for contour in contours[1:]:
        # Filter based on contour area and aspect ratio
        contour_area = cv2.contourArea(contour)
        _, _, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h

        if contour_area < 0.5 * cv2.contourArea(larger_box_contour) and aspect_ratio > 0.5:
            smaller_box_contours.append(contour)

    return larger_box_contour, smaller_box_contours

def detect_large_brown_object(image, min_contour_area=1000):
    # Load the image
    # image = cv2.imread(image_path)
    if image is None:
        print("Error: Could not open or find the image.")
        return

    image = resize_image(image, 500)

    image, is_converted = normalize_temperatures(image)

    # Convert the image to Lab color space
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2Lab)

    # Define the lower and upper bounds for brown color in Lab
    lower_brown = np.array([0,  127 + -40, 127 + -10])
    upper_brown = np.array([50, 127 + 40,  127 + 128])

    if is_converted:
        lower_brown = np.array([0,  127 + -60, 127 + -40])
        upper_brown = np.array([40, 127 + 20,  127 + 108])

    # Threshold the image to get only the brown regions
    brown_mask = cv2.inRange(lab_image, lower_brown, upper_brown)

    # Find contours in the thresholded image
    contours, _ = cv2.findContours(brown_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    total_area = 0.0

    center_x = image.shape[1] // 2
    center_y = image.shape[0] // 2

    # Iterate through the contours and draw a bounding box around the large brown object
    for contour in contours:
        contour_area = cv2.contourArea(contour)

        if contour_area >= min_contour_area:
            # Calculate the centroid of the contour
            M = cv2.moments(contour)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])

            # Calculate the Euclidean distance from the center
            distance_from_center = np.sqrt((center_x - cX) ** 2 + (center_y - cY) ** 2)

            # Ignore contours that are far from the center (adjust the threshold as needed)
            if distance_from_center > max(len(image), len(image[0])) * .4:       
                continue

            total_area += contour_area

            # Approximate the contour to a polygon
            epsilon = 0.02 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)

            # Draw the contour and its approximated shape
            cv2.drawContours(image, [approx], 0, (0, 255, 0), 2)

    # Display the image with the bounding boxes and sizes
    if __name__ == "__main__":
        cv2.imshow(image_path, image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    max_score = len(image) * len(image[0]) * 0.10
    return total_area < max_score, total_area / (total_area + max_score)

if __name__ == "__main__":
    for (_, _, filenames) in os.walk("trees"):
        for filename in filenames:
            image_path = 'trees/' + filename
            min_contour_area = 200  # Adjust the minimum contour area as needed
            print(filename)
            image = cv2.imread(image_path)
            print(detect_large_brown_object(image, min_contour_area))
