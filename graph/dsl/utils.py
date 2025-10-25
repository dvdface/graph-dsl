"""
Utility classes and functions for the graph-based DSL.

This module contains helper classes for parallel processing
and other utility functions.
"""

import cv2
import numpy as np
from typing import Tuple, List


class GridWorker:
    """Worker class for parallel grid-based change detection.

    Returns per-cell change decision and a numeric score indicating magnitude
    (normalized to 0-1 where higher means more change when possible).
    """
    
    @staticmethod
    def convert_threshold(threshold: float, method: str) -> float:
        """
        Convert normalized threshold to method-specific threshold.
        
        Args:
            threshold: Normalized threshold in [0,1]. 0 = no difference, 1 = 100% difference.
            method: Detection method ("abs" or "ssim")
            
        Returns:
            Method-specific threshold value
        """
        if method == "abs":
            # For absolute difference, threshold is already normalized [0,1]
            # where 0 = no difference, 1 = 255 pixel difference
            return threshold
        elif method == "ssim":
            # For SSIM, threshold is already normalized [0,1]
            # where 0 = identical (SSIM=1), 1 = completely different (SSIM=0)
            return threshold
        else:
            # Default to normalized threshold
            return threshold
    
    @staticmethod
    def process_cell(args: Tuple):
        """
        Process a single grid cell for change detection.
        
        Args:
            args: Tuple containing (prev_frame, curr_frame, effective_mask, grid_coords, method, threshold)
            
        Returns:
            Tuple[bool, float]: (changed, score)
        """
        prev, curr, effective_mask, grid_coords, method, threshold = args
        y1, y2, x1, x2 = grid_coords
        
        try:
            # Extract cell regions
            prev_cell = prev[y1:y2, x1:x2]
            curr_cell = curr[y1:y2, x1:x2]
            mask_cell = effective_mask[y1:y2, x1:x2]
            
            # Check if cell is in valid area (has any valid pixels)
            if np.sum(mask_cell) == 0:
                # Cell is not in valid area, return -1
                return (False, -1.0)
            
            # Convert normalized threshold to method-specific threshold
            converted_threshold = GridWorker.convert_threshold(threshold, method)
            
            if method == "abs":
                return GridWorker._abs_diff_method(prev_cell, curr_cell, mask_cell, converted_threshold)
            elif method == "ssim":
                return GridWorker._ssim_method(prev_cell, curr_cell, mask_cell, converted_threshold)
            else:
                # Default to abs method
                return GridWorker._abs_diff_method(prev_cell, curr_cell, mask_cell, converted_threshold)
            
        except Exception:
            # Return no change if processing fails
            return (False, 0.0)
    
    @staticmethod
    def _abs_diff_method(prev_cell: np.ndarray, curr_cell: np.ndarray, 
                        mask_cell: np.ndarray, threshold: float):
        """
        Absolute difference method for change detection.
        
        Args:
            prev_cell: Previous frame cell
            curr_cell: Current frame cell
            mask_cell: Mask for valid areas
            threshold: Normalized threshold in [0,1]. 0 = no difference, 1 = 100% difference.
            
        Returns:
            Tuple[bool, float]: (changed, score) where score is normalized [0,1]
        """
        # Compute absolute difference
        cell_diff = cv2.absdiff(prev_cell, curr_cell)
        
        # Convert to grayscale if needed
        if len(cell_diff.shape) == 3:
            gray = cv2.cvtColor(cell_diff, cv2.COLOR_BGR2GRAY)
        else:
            gray = cell_diff
        
        # Apply effective mask to gray
        gray_masked = cv2.bitwise_and(gray, mask_cell)
        
        # Difference score in [0,1]: mean absolute difference normalized by 255
        mean_diff = float(cv2.mean(gray_masked, mask=mask_cell)[0])
        score = max(0.0, min(1.0, mean_diff / 255.0))
        
        # Unified decision: changed if score >= threshold (difference threshold)
        return (score >= threshold, score)
    
    @staticmethod
    def _ssim_method(prev_cell: np.ndarray, curr_cell: np.ndarray, 
                    mask_cell: np.ndarray, threshold: float):
        """
        Structural Similarity Index (SSIM) method for change detection.
        Uses scikit-image implementation.
        
        Args:
            prev_cell: Previous frame cell
            curr_cell: Current frame cell
            mask_cell: Mask for valid areas
            threshold: Normalized threshold in [0,1]. 0 = no difference, 1 = 100% difference.
            
        Returns:
            Tuple[bool, float]: (changed, score) where score is normalized [0,1]
        """
        try:
            from skimage.metrics import structural_similarity as ssim
            return GridWorker._ssim_skimage(prev_cell, curr_cell, mask_cell, threshold, ssim)
        except ImportError:
            # If scikit-image is not available, fallback to abs method
            print("Warning: scikit-image not available, falling back to absolute difference method")
            return GridWorker._abs_diff_method(prev_cell, curr_cell, mask_cell, threshold)
        except Exception:
            # Return False if SSIM calculation fails
            return (False, 0.0)
    
    @staticmethod
    def _ssim_skimage(prev_cell: np.ndarray, curr_cell: np.ndarray, 
                     mask_cell: np.ndarray, threshold: float, ssim_func):
        """SSIM using scikit-image (more accurate)"""
        # Convert to grayscale if needed
        if len(prev_cell.shape) == 3:
            prev_gray = cv2.cvtColor(prev_cell, cv2.COLOR_BGR2GRAY)
            curr_gray = cv2.cvtColor(curr_cell, cv2.COLOR_BGR2GRAY)
        else:
            prev_gray = prev_cell
            curr_gray = curr_cell
        
        # Apply mask to both images
        prev_masked = cv2.bitwise_and(prev_gray, mask_cell)
        curr_masked = cv2.bitwise_and(curr_gray, mask_cell)
        
        # Calculate SSIM
        ssim_value = float(ssim_func(prev_masked, curr_masked, data_range=255))
        # Difference score in [0,1]: 1 - SSIM (where SSIM=1 means identical, SSIM=0 means completely different)
        score = max(0.0, min(1.0, 1.0 - ssim_value))
        # Unified decision: changed if score >= threshold
        changed = score >= threshold
        return (changed, score)


class ImageUtils:
    """Utility functions for image processing operations."""
    
    @staticmethod
    def resize_with_aspect_ratio(image: np.ndarray, target_size: Tuple[int, int], 
                                interpolation: int = cv2.INTER_LINEAR) -> np.ndarray:
        """
        Resize image while maintaining aspect ratio.
        
        Args:
            image: Input image
            target_size: Target (width, height)
            interpolation: Interpolation method
            
        Returns:
            Resized image
        """
        h, w = image.shape[:2]
        target_w, target_h = target_size
        
        # Calculate scaling factor
        scale = min(target_w / w, target_h / h)
        
        # Calculate new dimensions
        new_w = int(w * scale)
        new_h = int(h * scale)
        
        # Resize image
        resized = cv2.resize(image, (new_w, new_h), interpolation=interpolation)
        
        # Create canvas with target size
        canvas = np.zeros((target_h, target_w, image.shape[2] if len(image.shape) == 3 else 1), 
                         dtype=image.dtype)
        
        # Center the resized image on canvas
        y_offset = (target_h - new_h) // 2
        x_offset = (target_w - new_w) // 2
        
        if len(image.shape) == 3:
            canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized
        else:
            canvas[y_offset:y_offset+new_h, x_offset:x_offset+new_w, 0] = resized
            
        return canvas

    @staticmethod
    def create_grid_overlay(image: np.ndarray, grid_size: Tuple[int, int], 
                          color: Tuple[int, int, int] = (0, 255, 0), 
                          thickness: int = 1) -> np.ndarray:
        """
        Create a grid overlay on the image.
        
        Args:
            image: Input image
            grid_size: Grid dimensions (rows, cols)
            color: Grid line color (BGR)
            thickness: Grid line thickness
            
        Returns:
            Image with grid overlay
        """
        h, w = image.shape[:2]
        m, n = grid_size
        
        overlay = image.copy()
        
        # Draw vertical lines
        for i in range(1, n):
            x = int(i * w / n)
            cv2.line(overlay, (x, 0), (x, h), color, thickness)
        
        # Draw horizontal lines
        for i in range(1, m):
            y = int(i * h / m)
            cv2.line(overlay, (0, y), (w, y), color, thickness)
        
        return overlay

    @staticmethod
    def draw_roi_boxes(image: np.ndarray, roi_coords: List[Tuple[int, int, int, int]], 
                      color: Tuple[int, int, int] = (255, 0, 0), 
                      thickness: int = 2) -> np.ndarray:
        """
        Draw ROI boxes on the image.
        
        Args:
            image: Input image
            roi_coords: List of (x1, y1, x2, y2) coordinates
            color: Box color (BGR)
            thickness: Box line thickness
            
        Returns:
            Image with ROI boxes drawn
        """
        overlay = image.copy()
        
        for i, (x1, y1, x2, y2) in enumerate(roi_coords):
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)
            # Add label
            cv2.putText(overlay, f"ROI {i+1}", (x1, y1-10), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, thickness)
        
        return overlay

    @staticmethod
    def draw_detection_boxes(image: np.ndarray, detections: List[dict], 
                           color: Tuple[int, int, int] = (0, 255, 0), 
                           thickness: int = 2) -> np.ndarray:
        """
        Draw object detection boxes on the image.
        
        Args:
            image: Input image
            detections: List of detection dictionaries
            color: Default box color (BGR)
            thickness: Box line thickness
            
        Returns:
            Image with detection boxes drawn
        """
        overlay = image.copy()
        
        for detection in detections:
            x1, y1, x2, y2 = detection['bbox']
            confidence = detection['confidence']
            class_name = detection['class_name']
            
            # Draw bounding box
            cv2.rectangle(overlay, (x1, y1), (x2, y2), color, thickness)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, thickness)[0]
            
            # Draw label background
            cv2.rectangle(overlay, (x1, y1 - label_size[1] - 10), 
                         (x1 + label_size[0], y1), color, -1)
            
            # Draw label text
            cv2.putText(overlay, label, (x1, y1 - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), thickness)
        
        return overlay


class DebugUtils:
    """Utility functions for debug visualization."""
    
    @staticmethod
    def save_debug_frame(image: np.ndarray, filename: str, debug_dir: str = "debug_videos"):
        """
        Save a frame for debugging purposes.
        
        Args:
            image: Image to save
            filename: Filename for the saved image
            debug_dir: Directory to save debug images
        """
        import os
        os.makedirs(debug_dir, exist_ok=True)
        filepath = os.path.join(debug_dir, filename)
        cv2.imwrite(filepath, image)

    @staticmethod
    def create_change_visualization(image: np.ndarray, changes: List[bool], 
                                  grid_size: Tuple[int, int], 
                                  change_color: Tuple[int, int, int] = (0, 0, 255)) -> np.ndarray:
        """
        Create a visualization of detected changes.
        
        Args:
            image: Input image
            changes: List of boolean values indicating changes
            grid_size: Grid dimensions (rows, cols)
            change_color: Color for changed cells (BGR)
            
        Returns:
            Image with change visualization overlay
        """
        h, w = image.shape[:2]
        m, n = grid_size
        cell_h = h // m
        cell_w = w // n
        
        overlay = image.copy()
        
        for i, changed in enumerate(changes):
            if changed:
                row = i // n
                col = i % n
                
                y1 = row * cell_h
                y2 = (row + 1) * cell_h
                x1 = col * cell_w
                x2 = (col + 1) * cell_w
                
                # Draw semi-transparent overlay
                cv2.rectangle(overlay, (x1, y1), (x2, y2), change_color, -1)
        
        # Blend with original image
        result = cv2.addWeighted(image, 0.7, overlay, 0.3, 0)
        return result
