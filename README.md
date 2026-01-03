# Gaze Detection

## How It Works

### Pipeline

1. **Face Detection**: Haar Cascade classifier detects face regions
2. **Eye Detection**: Haar Cascade classifier detects eyes within face regions
3. **Pupil Detection**: 
   - Extract eye region
   - Threshold to find dark areas
   - Use contour analysis to find pupil center
4. **Gaze Estimation**:
   - Calculate vector from eye center to pupil (gaze direction)
   - Normalize the vector
   - Calculate dot product with vector toward image center
   - High dot product (> 0.3) indicates looking at camera

### Key Parameters

```python
dot_product > 0.3  # Threshold for "looking at camera"
                   # 0.3 ≈ 73 degrees acceptance angle
                   # Adjust based on sensitivity needs
```

## Usage

### Basic Usage

```python
import cv2
from gaze_detection import detect_gaze_direction, visualize_gaze

# Load image (or capture from camera)
image = cv2.imread('face_image.jpg')

# Detect gaze
gaze_results = detect_gaze_direction(image)

# Visualize results
result_image = visualize_gaze(image, gaze_results)

# Display
cv2.imshow('Gaze Detection', result_image)
cv2.waitKey(0)
cv2.destroyAllWindows()
```

### Output Format

```python
{
    "face_1": {
        "face_box": (x1, y1, x2, y2),
        "face_center": (cx, cy),
        "estimated_distance": 1.5,
        "overall_looking_at_camera": True,
        "eyes": {
            "left_eye": {
                "pupil_position": (px, py),
                "gaze_vector": (vx, vy),
                "dot_product": 0.65,
                "looking_at_camera": True
            },
            "right_eye": {...}
        }
    }
}
```

## Visualization Output

The visualization shows:
- **Green boxes**: Face regions where gaze is toward camera
- **Red boxes**: Face regions where gaze is NOT toward camera
- **Blue dots**: Detected pupils
- **Yellow lines**: Gaze vectors
- **Text labels**: Dot product values (0-1 scale) and eye names

## Performance Considerations

### Speed
- Face detection: ~50-100ms per frame
- Eye detection: ~20-50ms per frame
- Pupil detection: ~10-30ms per frame
- Total per frame: ~80-180ms (5-12 FPS on average machine)

### Accuracy
- Works well with frontal faces
- Requires reasonable lighting (dark eyes needed for detection)
- Distance affects accuracy (works best 0.5-2 meters away)

## Future Improvements

1. **CNN-based Detection**: Replace Haar Cascades with deep learning models (MTCNN, RetinaFace)
2. **Gaze Tracking**: Add head pose estimation for more accurate gaze direction
3. **Performance**: Use GPU acceleration with CUDA
4. **Lighting**: Add adaptive thresholding for varying lighting conditions
5. **Calibration**: Per-user calibration to improve accuracy

## Dependencies

- `opencv-python` (>=4.5.0)
- `numpy` (>=1.19.0)
- No need for heavy dependencies like `dlib` or `face_recognition`

## Installation

```bash
pip install opencv-python numpy
```
