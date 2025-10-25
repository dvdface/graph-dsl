"""
Core classes for the graph-based DSL.

This module contains the main Data class and AST node definitions
for building image processing pipelines.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Union, Dict, Any
import os
import threading
from queue import Queue
from .source import Source
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


class Node:
    """AST node representing an operation in the processing pipeline."""
    
    def __init__(self, op_name: str, params: Dict[str, Any]):
        """
        Initialize an AST node.
        
        Args:
            op_name: Name of the operation
            params: Parameters for the operation
        """
        self.op_name = op_name
        self.params = params
    
    def __repr__(self):
        return f"Node({self.op_name}, {self.params})"


class Data:
    """
    Main class for the graph-based DSL.
    
    Provides a fluent interface for building image processing pipelines
    with support for various operations like ROI, filtering, object detection,
    and change detection.
    """
    
    def __init__(self, source: Union[str, int]):
        """
        Initialize the Data object with a source.
        
        Args:
            source: Image directory path / video file path / camera index (int)
        """
        self.ast: List[Node] = []
        self.frames: List[np.ndarray] = []
        self.frame_paths: List[str] = []
        self.frame_offsets: List[float] = []

        # DSL related default parameters (will be set by Config node)

        # Debug settings
        self.debug_flag = False
        # Default debug output directory: current directory
        self.debug_dir = "."
        os.makedirs(self.debug_dir, exist_ok=True)
        
        # Verbose settings
        self.verbose_flag = False
        # Default verbose output directory: current directory
        self.verbose_dir = "."
        os.makedirs(self.verbose_dir, exist_ok=True)

        # Create Source object internally
        self.source = self._create_source(source)
        
        # Determine source type for compatibility
        self.is_video = hasattr(self.source, 'video_path')
        self.is_camera = hasattr(self.source, 'camera_index')

    def _create_source(self, source: Union[str, int]) -> Source:
        """Create appropriate Source object based on input type."""
        if isinstance(source, int):
            # Camera mode
            return Source.create('camera', source)
        else:
            source_path = Path(source)
            if source_path.is_dir():
                # Image directory mode
                return Source.create('dir', str(source_path))
            elif source_path.is_file():
                # Video file mode
                return Source.create('file', str(source_path))
            else:
                raise ValueError(f"Invalid source: {source}")

    # -------------------- DSL Methods --------------------
    
    def roi(self, crops: List[Tuple[int, int, int, int]]):
        """
        Define regions of interest to extract from images.
        
        Args:
            crops: List of (x1, y1, x2, y2) coordinates for ROI extraction
            
        Returns:
            Self for method chaining
        """
        self.ast.append(Node("ROI", {"crops": crops}))
        return self

    def filter(self, method: str = "gaussian", ksize: int = 5):
        """
        Apply filtering operations to images.
        
        Args:
            method: Filter method ("gaussian", "median")
            ksize: Kernel size for the filter
            
        Returns:
            Self for method chaining
        """
        self.ast.append(Node("Filter", {"method": method.lower(), "ksize": ksize}))
        return self


    def config(self, method: str = "ssim", threshold: Optional[float] = None, 
               grid_size: Union[int, Tuple[int, int], None] = None):
        """
        Configure change detection parameters.
        
        Args:
            method: Comparison method ("abs" or "ssim"). Default: "ssim"
            threshold: Normalized difference threshold in [0,1]. Default: 0.01 for "abs", 0.1 for "ssim"
            grid_size: Grid size as int (square grid) or tuple (rows, cols). Default: (32, 16)
            
        Returns:
            Self for method chaining
        """
        # Set method-specific default threshold if not provided
        if threshold is None:
            if method == "ssim":
                threshold = 0.1
            elif method == "abs":
                threshold = 0.01
            else:
                raise ValueError(f"Unsupported change detection method: {method}. Supported methods: 'ssim', 'abs'")
        
        # Set default grid size if not provided
        if grid_size is None:
            grid_size = (32, 16)
        elif isinstance(grid_size, int):
            grid_size = (grid_size, grid_size)
        elif isinstance(grid_size, (tuple, list)) and len(grid_size) == 2:
            grid_size = tuple(grid_size)
        else:
            raise ValueError("grid_size must be an int or a tuple/list of 2 integers")
        
        # Create Config AST node
        self.ast.append(Node("Config", {
            "method": method,
            "threshold": threshold,
            "grid_size": grid_size
        }))
        
        return self

    def detect(self, type: str = "both", 
               include_area: Optional[List[Tuple[float, float, float, float]]] = [],
               exclude_area: Optional[List[Tuple[float, float, float, float]]] = [],
               start_include_area: Optional[List[Tuple[float, float, float, float]]] = [],
               start_exclude_area: Optional[List[Tuple[float, float, float, float]]] = [],
               stop_include_area: Optional[List[Tuple[float, float, float, float]]] = [],
               stop_exclude_area: Optional[List[Tuple[float, float, float, float]]] = [],
               stable_frames: int = 60, stable_duration_ms: Optional[int] = 500):
        """
        Unified change detection method.
        
        Args:
            type: Detection type ("start", "stop", "both"). Default: "start"
            include_area: Common include areas for all detection types. Default: []
            exclude_area: Common exclude areas for all detection types. Default: []
            start_include_area: Additional include areas for start detection. Default: []
            start_exclude_area: Additional exclude areas for start detection. Default: []
            stop_include_area: Additional include areas for stop detection. Default: []
            stop_exclude_area: Additional exclude areas for stop detection. Default: []
            stable_frames: Number of consecutive frames without changes to consider stable. Default: 60
            stable_duration_ms: Duration in milliseconds without changes to consider stable. Default: 500ms
            
        Returns:
            Self for method chaining
        """
        # Get method and threshold from Config node
        method = 'ssim'
        threshold = 0.1
        for node in self.ast:
            if node.op_name == "Config":
                method = node.params.get("method", "ssim")
                threshold = node.params.get("threshold", 0.1 if method == "ssim" else 0.01)
                break
        
        # Create a unified detect node with all parameters
        self.ast.append(Node("detect", {
            "type": type,
            "include_area": include_area, 
            "exclude_area": exclude_area,
            "start_include_area": start_include_area,
            "start_exclude_area": start_exclude_area,
            "stop_include_area": stop_include_area,
            "stop_exclude_area": stop_exclude_area,
            "method": method,
            "threshold": threshold,
            "stable_frames": stable_frames,
            "stable_duration_ms": stable_duration_ms
        }))
        
        return self


    def debug(self, output_dir: Optional[str] = ".", filename: str = "debug.mp4", delay_stop_frames: int = 10):
        """
        Enable debug mode for visualization.
        
        Args:
            output_dir: Directory to write debug outputs. Defaults to current directory.
            filename: Debug video filename. Defaults to "debug.mp4".
            delay_stop_frames: Number of frames to delay before stopping detection. Default: 10
        
        Returns:
            Self for method chaining
        """
        self.debug_flag = True
        self.debug_dir = str(output_dir)
        self.debug_filename = filename
        self.delay_stop_frames = delay_stop_frames
        
        # Create debug AST node
        self.ast.append(Node("Debug", {
            "enabled": True,
            "output_dir": self.debug_dir,
            "filename": self.debug_filename,
            "delay_stop_frames": delay_stop_frames
        }))
        
        os.makedirs(self.debug_dir, exist_ok=True)
        return self

    def verbose(self, output_dir: Optional[str] = None, filename: str = "verbose.txt"):
        """
        Enable verbose mode for detailed grid comparison output.
        
        Args:
            output_dir: Directory to write verbose outputs. Defaults to current directory.
            filename: Verbose output filename. Defaults to "verbose.txt".
        
        Returns:
            Self for method chaining
        """
        self.verbose_flag = True
        if output_dir:
            self.verbose_dir = str(output_dir)
        else:
            self.verbose_dir = "."  # Current directory
        self.verbose_filename = filename
        
        # Create verbose AST node
        self.ast.append(Node("Verbose", {
            "enabled": True,
            "output_dir": self.verbose_dir,
            "filename": self.verbose_filename
        }))
        
        os.makedirs(self.verbose_dir, exist_ok=True)
        return self

    # -------------------- Internal Utility Methods --------------------
    

    def _normalize_or_pixel_coords(self, coords: Optional[List[Tuple[float, float, float, float]]], 
                                 w: int, h: int) -> List[Tuple[int, int, int, int]]:
        """
        Convert normalized coordinates to pixel coordinates.
        
        Args:
            coords: List of coordinate tuples
            w: Image width
            h: Image height
            
        Returns:
            List of pixel coordinate tuples
        """
        if not coords:
            return []
            
        result = []
        for x1, y1, x2, y2 in coords:
            if max(x1, y1, x2, y2) <= 1.0:
                # Normalized coordinates
                result.append((int(x1 * w), int(y1 * h), int(x2 * w), int(y2 * h)))
            else:
                # Pixel coordinates
                result.append((int(x1), int(y1), int(x2), int(y2)))
        return result

    def run(self):
        """
        Execute the processing pipeline defined by the AST.
        
        Returns:
            List of processing results
        """
        from .processing import ProcessingEngine
        engine = ProcessingEngine(self)
        return engine.execute()
