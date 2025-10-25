""" 
    Graph-based DSL for image and video processing with change detection.
    This module provides a fluent interface for building image processing pipelines with support for ROI extraction, filtering, object detection, and change detection.
"""

from .core import Data, Node
from .processing import ProcessingEngine
from .utils import GridWorker
from .source import Source, DirSource, FileSource, CameraSource

__version__ = "0.1.0"


__all__ = ["Data", "Node", "ProcessingEngine", "GridWorker", "Source", "DirSource", "FileSource", "CameraSource"]