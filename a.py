import cv2
import numpy as np

# Function to track and trace the red color band in the frame
def track_and_trace_red_color_band(frame, trace, detected_pixels):
    # Convert the frame from BGR to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Define the new range of red color in HSV
    lower_red = np.array([150, 45, 215])
    upper_red = np.array([179, 255, 255])

    # Create a mask to extract the red color region
    mask = cv2.inRange(hsv, lower_red, upper_red)

    # Find contours in the mask
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Process each contour
    for contour in contours:
        # Calculate the center of the contour
        M = cv2.moments(contour)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])

            # Get the BGR values of the detected pixel
            bgr_value = frame[cy, cx]
            detected_pixels.append(bgr_value)

            # Determine if the marker is in the left or right half of the image
            if cx < frame.shape[1] // 2:   
                marker_color = (255, 0, 0)  # Blue color for the left half
            else:
                marker_color = (0, 255, 0)  # Green color for the right half

            # Draw a circle at the center of the contour with the determined color
            cv2.circle(frame, (cx, cy), 10, marker_color, -1)

            # Add the current position to the trace list
            trace.append((cx, cy))

    # Draw lines to trace the movement
    for i in range(1, len(trace)):
        cv2.line(frame, trace[i - 1], trace[i], (0, 0, 255), 2)

    return frame

# Open the webcam 0-webcam
cap = cv2.VideoCapture(0)

# Lists to store the trace of the red color band and detected pixel values
trace = []
detected_pixels = []

while True:
    # Capture frame-by-frame
    ret, frame = cap.read()

    # Check if the frame is valid
    if not ret:
        break

    # Flip the frame horizontally for better visualization
    frame = cv2.flip(frame, 1)

    # Track and trace the red color band in the frame
    result_frame = track_and_trace_red_color_band(frame, trace, detected_pixels)

    # Display the resulting frame
    cv2.imshow('Red Color Band Tracking with Color Change', result_frame)

    # Break the loop when 'q' key is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close all windows
cap.release()
cv2.destroyAllWindows()

# Print the detected pixel values
print("Detected Pixel Values:")
for pixel_value in detected_pixels:
    print(pixel_value)
