import numpy as np
import cv2
import math

class Metrics:
    def __init__(self):
        pass

    def calculate_entropy(self, image: np.ndarray) -> float:
        if image is None:
            return 0.0

        histogram = cv2.calcHist([image], [0], None, [256], [0, 256])
        histogram = histogram.flatten() 

        histogram_length = sum(histogram)

        samples_probability = [float(h) / histogram_length for h in histogram if h != 0]

        return -sum([p * math.log(p, 2) for p in samples_probability])

    def calculate_cii(self, original_image: np.ndarray, enhanced_image: np.ndarray) -> float:
        if original_image is None or enhanced_image is None:
            return 0.0

        std_dev_orig = np.std(original_image)
        std_dev_enhanced = np.std(enhanced_image)

        if std_dev_orig == 0:
            return float('inf')
        cii = std_dev_enhanced / std_dev_orig

        return cii

if __name__ == '__main__':
    metrics_calculator = Metrics()
    
  