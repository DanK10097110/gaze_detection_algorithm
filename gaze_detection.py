import cv2
import numpy as np
from pathlib import Path

def takePic():
    """Capture a single frame from the camera."""
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Camera could not be accessed.")
        return None
    
    ret, frame = cap.read()
    cap.release()
    
    if not ret:
        print("Error: No frame captured.")
        return None
    
    return frame


def detect_faces_cascade(image):
    """
    Detect faces using Haar Cascade classifier.
    Returns list of (x, y, w, h) tuples for each face.
    """
    # Load pre-trained Haar Cascade classifier
    face_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
    )
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    return faces


def detect_eyes_in_face(image, face_roi):
    """
    Detect eyes within a face region.
    Returns list of (x, y, w, h) tuples for each eye within the face.
    """
    eye_cascade = cv2.CascadeClassifier(
        cv2.data.haarcascades + 'haarcascade_eye.xml'
    )
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    eyes = eye_cascade.detectMultiScale(gray[face_roi[1]:face_roi[1]+face_roi[3], 
                                              face_roi[0]:face_roi[0]+face_roi[2]])
    
    return eyes


def get_pupil_position(eye_region):
    """
    Detect pupil position using blob detection.
    Returns (pupil_x, pupil_y) in eye region coordinates.
    """
    gray = cv2.cvtColor(eye_region, cv2.COLOR_BGR2GRAY)
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    
    # Apply threshold to find dark regions (pupil)
    _, thresh = cv2.threshold(blurred, 50, 255, cv2.THRESH_BINARY_INV)
    
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    if not contours:
        return None
    
    # Get the largest contour (should be the pupil)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Calculate moments
    M = cv2.moments(largest_contour)
    
    if M["m00"] == 0:
        return None
    
    # Calculate centroid
    pupil_x = int(M["m10"] / M["m00"])
    pupil_y = int(M["m01"] / M["m00"])
    
    return (pupil_x, pupil_y)


def detect_gaze_direction(image):
    """
    Detect gaze direction for faces in the image.
    Returns: dict with face info and gaze estimates
    """
    if image is None:
        return {"error": "Input image is None"}
    
    # Detect faces
    faces = detect_faces_cascade(image)
    
    if len(faces) == 0:
        return {"error": "No faces detected"}
    
    img_height, img_width = image.shape[:2]
    img_center_x = img_width // 2
    img_center_y = img_height // 2
    
    results = {}
    
    for face_idx, (fx, fy, fw, fh) in enumerate(faces):
        face_id = f"face_{face_idx + 1}"
        face_left, face_top = fx, fy
        face_right, face_bottom = fx + fw, fy + fh
        face_center_x = (face_left + face_right) // 2
        face_center_y = (face_top + face_bottom) // 2
        
        # Extract face region
        face_region = image[face_top:face_bottom, face_left:face_right]
        
        if face_region.size == 0:
            continue
        
        # Estimate distance from face size
        estimated_distance = max(fh, fw) / 200.0
        
        gaze_data = {
            "face_box": (face_left, face_top, face_right, face_bottom),
            "face_center": (face_center_x, face_center_y),
            "estimated_distance": estimated_distance,
            "eyes": {}
        }
        
        # Detect eyes in face
        gray_face = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        eye_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + 'haarcascade_eye.xml'
        )
        eyes = eye_cascade.detectMultiScale(gray_face, 1.3, 5)
        
        eye_names = ['left_eye', 'right_eye']
        
        for eye_idx, (ex, ey, ew, eh) in enumerate(eyes[:2]):  # Process only first 2 eyes
            eye_name = eye_names[eye_idx]
            
            # Convert to image coordinates
            eye_left = face_left + ex
            eye_top = face_top + ey
            eye_right = eye_left + ew
            eye_bottom = ey + ey + eh
            
            # Extract eye region
            eye_region = image[eye_top:eye_bottom, eye_left:eye_right]
            
            if eye_region.size == 0:
                continue
            
            # Eye center in image coordinates
            eye_center_x = (eye_left + eye_right) // 2
            eye_center_y = (eye_top + eye_bottom) // 2
            
            # Get pupil position in eye region
            pupil_local = get_pupil_position(eye_region)
            
            if pupil_local is None:
                gaze_data["eyes"][eye_name] = {"error": "Could not detect pupil"}
                continue
            
            # Convert to image coordinates
            pupil_x = eye_left + pupil_local[0]
            pupil_y = eye_top + pupil_local[1]
            
            # Calculate gaze vector (from eye center to pupil)
            gaze_vec_x = pupil_x - eye_center_x
            gaze_vec_y = pupil_y - eye_center_y
            
            # Normalize
            magnitude = np.sqrt(gaze_vec_x**2 + gaze_vec_y**2)
            
            if magnitude > 0.1:
                gaze_vec_x /= magnitude
                gaze_vec_y /= magnitude
            else:
                gaze_vec_x = 0
                gaze_vec_y = 0
            
            # Calculate angle to image center
            vec_to_center_x = img_center_x - face_center_x
            vec_to_center_y = img_center_y - face_center_y
            
            center_magnitude = np.sqrt(vec_to_center_x**2 + vec_to_center_y**2)
            
            if center_magnitude > 0.1:
                vec_to_center_x /= center_magnitude
                vec_to_center_y /= center_magnitude
            
            # Calculate dot product (cosine of angle between vectors)
            dot_product = gaze_vec_x * vec_to_center_x + gaze_vec_y * vec_to_center_y
            
            # If dot product is high (close to 1), gaze is toward camera
            # Use threshold of 0.3 (about 73 degrees)
            looking_at_camera = dot_product > 0.3
            
            gaze_data["eyes"][eye_name] = {
                "pupil_position": (pupil_x, pupil_y),
                "gaze_vector": (float(gaze_vec_x), float(gaze_vec_y)),
                "dot_product": float(dot_product),
                "looking_at_camera": looking_at_camera
            }
        
        # Determine overall gaze
        eye_results = gaze_data["eyes"]
        if len(eye_results) >= 2:
            left_looking = eye_results.get("left_eye", {}).get("looking_at_camera", False)
            right_looking = eye_results.get("right_eye", {}).get("looking_at_camera", False)
            gaze_data["overall_looking_at_camera"] = left_looking or right_looking
        elif len(eye_results) == 1:
            eye_result = list(eye_results.values())[0]
            gaze_data["overall_looking_at_camera"] = eye_result.get("looking_at_camera", False)
        else:
            gaze_data["overall_looking_at_camera"] = False
        
        results[face_id] = gaze_data
    
    return results


def visualize_gaze(image, gaze_results):
    """
    Draw gaze vectors and information on the image.
    """
    img_copy = image.copy()
    
    for face_id, gaze_data in gaze_results.items():
        if not isinstance(gaze_data, dict) or "error" in gaze_data:
            continue
        
        # Draw face box
        if "face_box" not in gaze_data:
            continue
            
        left, top, right, bottom = gaze_data["face_box"]
        color = (0, 255, 0) if gaze_data.get("overall_looking_at_camera", False) else (0, 0, 255)
        cv2.rectangle(img_copy, (left, top), (right, bottom), color, 2)
        
        # Add text
        status = "LOOKING AT CAMERA" if gaze_data.get("overall_looking_at_camera", False) else "NOT LOOKING"
        cv2.putText(
            img_copy, status, (left, top - 10),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2
        )
        
        # Draw eye pupils
        for eye_name, eye_data in gaze_data.get("eyes", {}).items():
            if not isinstance(eye_data, dict) or "error" in eye_data:
                continue
            
            if "pupil_position" not in eye_data:
                continue
                
            pupil_x, pupil_y = eye_data["pupil_position"]
            cv2.circle(img_copy, (pupil_x, pupil_y), 3, (255, 0, 0), -1)
            
            # Draw gaze vector
            if "gaze_vector" in eye_data:
                gaze_vec_x, gaze_vec_y = eye_data["gaze_vector"]
                end_x = int(pupil_x + gaze_vec_x * 50)
                end_y = int(pupil_y + gaze_vec_y * 50)
                cv2.line(img_copy, (pupil_x, pupil_y), (end_x, end_y), (255, 255, 0), 1)
            
            # Add dot product value
            if "dot_product" in eye_data:
                dot_prod = eye_data["dot_product"]
                cv2.putText(
                    img_copy, f"{eye_name}: {dot_prod:.2f}", (pupil_x + 10, pupil_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (200, 200, 0), 1
                )
    
    return img_copy


def test_gaze_detection_from_camera():
    """Test gaze detection on a live camera frame."""
    print("Capturing frame from camera...")
    frame = takePic()
    
    if frame is None:
        print("Failed to capture frame.")
        return False
    
    print(f"Frame captured: {frame.shape}")
    
    print("Detecting gaze...")
    gaze_results = detect_gaze_direction(frame)
    
    print("\nGaze Detection Results:")
    print(gaze_results)
    
    # Visualize
    vis_frame = visualize_gaze(frame, gaze_results)
    
    # Display
    cv2.imshow('Gaze Detection', vis_frame)
    print("\nDisplaying result... Press any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return True


def test_gaze_detection_from_file(image_path):
    """Test gaze detection on an image file."""
    if not Path(image_path).exists():
        print(f"Image file not found: {image_path}")
        return False
    
    print(f"Loading image from: {image_path}")
    frame = cv2.imread(image_path)
    
    if frame is None:
        print("Failed to load image.")
        return False
    
    print(f"Image loaded: {frame.shape}")
    
    print("Detecting gaze...")
    gaze_results = detect_gaze_direction(frame)
    
    print("\nGaze Detection Results:")
    print(gaze_results)
    
    # Visualize
    vis_frame = visualize_gaze(frame, gaze_results)
    
    # Display
    cv2.imshow('Gaze Detection', vis_frame)
    print("\nDisplaying result... Press any key to close.")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    
    return True


def create_synthetic_face_image():
    """Create a synthetic test image with a face-like structure."""
    # Create a white image
    img = np.ones((480, 640, 3), dtype=np.uint8) * 255
    
    # Draw a face (circle)
    cv2.circle(img, (320, 200), 80, (200, 150, 100), -1)  # Face
    
    # Draw eyes
    cv2.circle(img, (290, 180), 15, (50, 50, 50), -1)  # Left eye white
    cv2.circle(img, (350, 180), 15, (50, 50, 50), -1)  # Right eye white
    
    # Draw pupils (looking at camera)
    cv2.circle(img, (290, 180), 8, (0, 0, 0), -1)  # Left pupil
    cv2.circle(img, (350, 180), 8, (0, 0, 0), -1)  # Right pupil
    
    # Draw nose
    points = np.array([[320, 190], [310, 210], [330, 210]], np.int32)
    cv2.polylines(img, [points], True, (150, 100, 50), 2)
    
    # Draw mouth
    cv2.ellipse(img, (320, 240), (40, 20), 0, 0, 180, (150, 100, 50), 2)
    
    return img


if __name__ == "__main__":
    print("=" * 60)
    print("Gaze Detection Test")
    print("=" * 60)
    
    # Test with synthetic image
    print("\nTesting with synthetic face image...")
    synthetic_img = create_synthetic_face_image()
    
    # Save synthetic image
    cv2.imwrite('test_face.jpg', synthetic_img)
    print("Saved synthetic test image to 'test_face.jpg'")
    
    print("Detecting gaze...")
    gaze_results = detect_gaze_direction(synthetic_img)
    
    print("\nGaze Detection Results:")
    for face_id, gaze_data in gaze_results.items():
        if isinstance(gaze_data, dict) and "error" in gaze_data:
            print(f"  {face_id}: {gaze_data['error']}")
        elif isinstance(gaze_data, dict):
            print(f"  {face_id}:")
            print(f"    Face box: {gaze_data.get('face_box', 'N/A')}")
            print(f"    Overall looking at camera: {gaze_data.get('overall_looking_at_camera', 'N/A')}")
            print(f"    Eyes detected: {len(gaze_data.get('eyes', {}))}")
            for eye_name, eye_data in gaze_data.get('eyes', {}).items():
                if isinstance(eye_data, dict) and "error" not in eye_data:
                    print(f"      {eye_name}: dot_product={eye_data.get('dot_product', 'N/A'):.2f}, "
                          f"looking={eye_data.get('looking_at_camera', 'N/A')}")
        else:
            print(f"  {face_id}: {gaze_data}")
    
    # Visualize
    vis_img = visualize_gaze(synthetic_img, gaze_results)
    cv2.imwrite('test_face_result.jpg', vis_img)
    print("\nSaved visualization to 'test_face_result.jpg'")
    
    # Try with camera if available
    print("\n" + "=" * 60)
    print("Testing with camera...")
    success = test_gaze_detection_from_camera()
    
    if success:
        print("\n✓ Gaze detection test completed successfully!")
    else:
        print("\n✗ Camera test failed (expected if no camera available).")
        print("✓ Synthetic test completed successfully!")
