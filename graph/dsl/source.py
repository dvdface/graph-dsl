"""
Source base class for frame data providers.

This module contains the Source base class that provides a common interface
for different types of frame sources (video files, cameras, etc.).
"""

import numpy as np
import os
import re
import cv2
import glob
import platform
import time
import threading
import queue
import tempfile
from abc import ABC, abstractmethod
from typing import Tuple, Optional, List


class Source(ABC):
    """
    Abstract base class for frame sources.
    
    Provides a common interface for accessing frame data with index,
    frame data, and timestamp information.
    """
    
    @abstractmethod
    def next_frame(self) -> Tuple[int, np.ndarray, float]:
        """
        Return the next frame from the source.
        
        Returns:
            Tuple containing:
            - frame_index (int): Index of the frame (0-based)
            - frame_data (np.ndarray): Frame data as numpy array
            - timestamp_ms (float): Frame timestamp in milliseconds (supports decimal precision)
            
        Raises:
            StopIteration: When no more frames are available
        """
        pass
    
    @abstractmethod
    def count(self) -> int:
        """
        Return the total number of frames in the source.
        
        Returns:
            int: Total frame count
        """
        pass
    
    def __iter__(self):
        """Make the source iterable."""
        return self
    
    def __next__(self):
        """Support for iteration protocol."""
        try:
            return self.next_frame()
        except StopIteration:
            raise
    
    @staticmethod
    def extract_ts(filename: str) -> float:
        """
        Extract timestamp from filename in format: xxxxxxxxxx_<integer_part>_<decimal_part>.ext
        
        Args:
            filename: The filename to extract timestamp from
            
        Returns:
            Timestamp in milliseconds (with decimal precision)
            
        Raises:
            ValueError: If timestamp cannot be extracted from filename
        """
        # Pattern to match: xxxxxxxxxx_<timestamp>.ext
        pattern = r'^[^_]+_(\d+_\d+)\.'
        match = re.match(pattern, os.path.basename(filename))
        
        if not match:
            raise ValueError(f"Could not extract timestamp from filename: {filename}")
        
        # Replace underscore with dot and convert to milliseconds
        ts_str = match.group(1).replace('_', '.')
        return float(ts_str) * 1000
    
    @staticmethod
    def create(source_type: str, *args, **kwargs) -> 'Source':
        """
        Factory method to create Source instances based on type.
        
        Args:
            source_type: Type of source to create. Supported types:
                - 'camera': Create CameraSource
                - 'file': Create FileSource  
                - 'dir': Create DirSource
            *args: Positional arguments passed to the source constructor
            **kwargs: Keyword arguments passed to the source constructor
            
        Returns:
            Source instance of the specified type
            
        Raises:
            ValueError: If source_type is not supported
            
        Examples:
            # Create camera source
            camera = Source.create('camera', 0, "640x480@30fps mjpg -6")
            
            # Create file source
            video = Source.create('file', "path/to/video.mp4")
            
            # Create directory source
            images = Source.create('dir', "path/to/images", ['.jpg', '.png'])
        """
        source_type = source_type.lower()
        
        if source_type == 'camera':
            return CameraSource(*args, **kwargs)
        elif source_type == 'file':
            return FileSource(*args, **kwargs)
        elif source_type == 'dir':
            return DirSource(*args, **kwargs)
        else:
            raise ValueError(f"Unsupported source type: {source_type}. "
                           f"Supported types are: 'camera', 'file', 'dir'")


class DirSource(Source):
    """ 
    Source for reading frames from a directory of image files.
    
    Supports common image formats and extracts timestamps from filenames.
    Uses yield-based loading for efficient memory usage.
    """
    
    def __init__(self, directory_path: str, extensions: List[str] = None):
        """
        Initialize directory source.
        
        Args:
            directory_path: Path to directory containing image files
            extensions: List of file extensions to include (default: common image formats)
        """
        self.directory_path = directory_path
        self.extensions = extensions or ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif']
        
        # Get all image files and sort them
        self.image_files = []
        for ext in self.extensions:
            pattern = os.path.join(directory_path, f"*{ext}")
            self.image_files.extend(glob.glob(pattern))
            pattern = os.path.join(directory_path, f"*{ext.upper()}")
            self.image_files.extend(glob.glob(pattern))
        
        # Remove duplicates (in case of case-insensitive filesystem)
        self.image_files = list(set(self.image_files))
        
        # Sort files by name (assuming they have timestamp in filename)
        self.image_files.sort()
        self.current_index = 0

        self.warning = False
    
    def next_frame(self) -> Tuple[int, np.ndarray, float]:
        """
        Return the next frame from the directory using yield-based loading.
        
        Returns:
            Tuple containing frame index, frame data, and timestamp in ms
            
        Raises:
            StopIteration: When no more frames are available
        """
        if self.current_index >= len(self.image_files):
            raise StopIteration
        
        file_path = self.image_files[self.current_index]
        
        # Read image only when requested (yield-based loading)
        frame = cv2.imread(file_path)
        if frame is None:
            raise ValueError(f"Could not read image: {file_path}")
        
        # Extract timestamp from filename
        try:
            timestamp_ms = self.extract_ts(file_path)
        except ValueError:
            # If no timestamp found, use index as fallback
            if(self.warning == False):
                print("图像文件名中不包含时间戳, 无法输出计算时间偏移量和时间戳")
                self.warning = True
            timestamp_ms = None
        
        frame_index = self.current_index
        self.current_index += 1
        
        return frame_index, frame, timestamp_ms
    
    def count(self) -> int:
        """Return the total number of frames in the directory."""
        return len(self.image_files)


class FileSource(Source):
    """
    Source for reading frames from a video file.
    
    Extracts base timestamp from filename, then reads frame timestamps
    directly from video metadata and adds the base timestamp.
    Uses yield-based loading for efficient memory usage.
    """
    
    def __init__(self, video_path: str):
        """
        Initialize video file source.
        
        Args:
            video_path: Path to video file
        """
        self.video_path = video_path
        self.cap = None
        self.current_index = 0
        self.frame_count = 0
        self.base_timestamp_ms = None
        
        # Extract base timestamp from filename
        try:
            self.base_timestamp_ms = self.extract_ts(video_path)
        except ValueError:
            # If no timestamp found, use 0 as fallback
            print("视频文件中不带时间戳, 使用默认时间戳0")
            self.base_timestamp_ms = 0.0
    
    def _ensure_cap_open(self):
        """Ensure video capture is open."""
        if self.cap is None:
            self.cap = cv2.VideoCapture(self.video_path)
            if not self.cap.isOpened():
                raise ValueError(f"Could not open video file: {self.video_path}")
            
            self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    def next_frame(self) -> Tuple[int, np.ndarray, float]:
        """
        Return the next frame from the video using yield-based loading.
        
        Returns:
            Tuple containing frame index, frame data, and timestamp in ms
            
        Raises:
            StopIteration: When no more frames are available
        """
        self._ensure_cap_open()
        
        if self.current_index >= self.frame_count:
            raise StopIteration
        
        # Read frame only when requested (yield-based loading)
        ret, frame = self.cap.read()
        if not ret or frame is None:
            raise StopIteration
        
        # Get frame timestamp from video metadata (in milliseconds)
        frame_timestamp_ms = self.cap.get(cv2.CAP_PROP_POS_MSEC)
        
        # Add base timestamp from filename to get absolute timestamp
        absolute_timestamp_ms = self.base_timestamp_ms + frame_timestamp_ms
        
        frame_index = self.current_index
        self.current_index += 1
        
        return frame_index, frame, absolute_timestamp_ms
    
    def count(self) -> int:
        """Return the total number of frames in the video."""
        self._ensure_cap_open()
        return self.frame_count
    
    def __del__(self):
        """Clean up video capture when object is destroyed."""
        if self.cap is not None:
            self.cap.release()


class CameraSource(Source):
    """
    High-performance camera source with threaded frame capture and caching.
    
    Uses a high-priority thread to continuously capture frames and cache them,
    while an async thread saves frames to disk with timestamps.
    """
    
    def __init__(self, camera_index: int, caps: str = None, save_directory: str = None):
        """
        Initialize camera source.
        
        Args:
            camera_index: Index of the camera device (0, 1, 2, etc.)
            caps: Configuration string in format "WIDTHxHEIGHT@FPS ENCODING EXPOSURE"
                  Example: "640x480@120fps jmpg -100"
            save_directory: Directory to save frames (if None, uses temp directory)
        """
        self.camera_index = camera_index
        self.caps = caps
        self.save_directory = save_directory
        self.cap = None
        self.current_index = 0
        
        # Threading components
        self.frame_queue = queue.Queue()  # No size limit
        self.capture_thread = None
        self.save_thread = None
        self.running = False
        self.stop_event = threading.Event()
        self.frame_counter = 0  # Global frame counter
        
        # Save thread state management
        self.save_position = 0  # Last saved position
        self.save_condition = threading.Condition()  # For wake/sleep control
        self.new_frame_event = threading.Event()  # Signal new frame arrival
        
        # Parse configuration if provided
        self._parse_config()
        
        # Setup save directory
        self._setup_save_directory()
        
        # Start threads
        self._start_threads()
    
    def _parse_config(self):
        """Parse configuration string and set camera parameters."""
        if not self.caps:
            return
        
        try:
            # Parse format: "WIDTHxHEIGHT@FPS ENCODING EXPOSURE"
            parts = self.caps.strip().split()
            if len(parts) < 1:
                return
            
            # Parse resolution and fps: "640x480@120fps"
            res_fps = parts[0]
            if '@' in res_fps:
                resolution, fps_part = res_fps.split('@')
                width, height = map(int, resolution.split('x'))
                fps = int(fps_part.replace('fps', ''))
            else:
                # Default resolution if no fps specified
                width, height = map(int, res_fps.split('x'))
                fps = 30  # Default fps
            
            # Parse encoding: "jmpg", "mjpeg", etc.
            encoding = parts[1] if len(parts) > 1 else None
            
            # Parse exposure: any integer value
            exposure = int(parts[2]) if len(parts) > 2 else None
            
            self._camera_params = {
                'width': width,
                'height': height,
                'fps': fps,
                'encoding': encoding,
                'exposure': exposure
            }
            
        except (ValueError, IndexError) as e:
            raise ValueError(f"Failed to parse camera config '{self.caps}': {e}")
    
    def _setup_save_directory(self):
        """Setup directory for saving frames."""
        if self.save_directory is None:
            # Create temporary directory
            self.save_directory = tempfile.mkdtemp(prefix="camera_frames_")
        else:
            # Create specified directory if it doesn't exist
            os.makedirs(self.save_directory, exist_ok=True)
    
    def _start_threads(self):
        """Start capture and save threads."""
        self.running = True
        self.stop_event.clear()
        
        # Start high-priority capture thread
        self.capture_thread = threading.Thread(target=self._capture_frames, daemon=True)
        self.capture_thread.start()
        
        # Start save thread
        self.save_thread = threading.Thread(target=self._save_frames, daemon=True)
        self.save_thread.start()
    
    def _capture_frames(self):
        """High-priority thread to continuously capture frames."""
        # Set thread priority to highest (Windows)
        try:
            import ctypes
            from ctypes import wintypes
            handle = ctypes.windll.kernel32.GetCurrentThread()
            ctypes.windll.kernel32.SetThreadPriority(handle, 2)  # THREAD_PRIORITY_HIGHEST
        except:
            pass  # Ignore if not on Windows or if setting priority fails
        
        self._ensure_cap_open()
        
        while self.running and not self.stop_event.is_set():
            try:
                # Read frame
                ret, frame = self.cap.read()
                if not ret or frame is None:
                    continue
                
                # Get timestamp
                timestamp_ms = time.time() * 1000.0
                
                # Create frame data structure
                frame_data = {
                    'sequence': self.frame_counter,  # 序号，递增
                    'frame_index': self.current_index,  # 帧的序号
                    'frame_data': frame,  # 帧的数据
                    'timestamp_ms': timestamp_ms,  # 时间戳
                    'file_path': None  # 落盘文件的路径，没有落盘前为None
                }
                
                # Put frame in queue (no size limit)
                self.frame_queue.put_nowait(frame_data)
                self.current_index += 1
                self.frame_counter += 1
                
                # Notify save thread of new frame
                self.new_frame_event.set()
                        
            except Exception as e:
                raise ValueError(f"Error in capture thread: {e}")
    
    def _save_frames(self):
        """Smart save thread with position tracking and wake/sleep control."""
        
        while self.running and not self.stop_event.is_set():
            try:
                # Check if there are frames to save from current position
                frames_to_save = []
                current_queue_size = self.frame_queue.qsize()
                
                # Collect frames from current position to end of queue
                temp_frames = []
                while not self.frame_queue.empty():
                    try:
                        frame_data = self.frame_queue.get_nowait()
                        temp_frames.append(frame_data)
                    except queue.Empty:
                        break
                
                # Filter frames that need to be saved (from save_position onwards)
                frames_to_save = [f for f in temp_frames if f['sequence'] >= self.save_position]
                
                # Put back frames that don't need saving yet
                for frame_data in temp_frames:
                    if frame_data['sequence'] < self.save_position:
                        self.frame_queue.put_nowait(frame_data)
                
                if frames_to_save:
                    
                    # Save frames
                    for frame_data in frames_to_save:
                        if not self.running or self.stop_event.is_set():
                            break
                            
                        sequence = frame_data['sequence']
                        frame_index = frame_data['frame_index']
                        frame = frame_data['frame_data']
                        timestamp_ms = frame_data['timestamp_ms']
                        
                        # Convert timestamp to filename format
                        timestamp_seconds = int(timestamp_ms // 1000)
                        timestamp_decimal = int((timestamp_ms % 1000) * 1000)
                        
                        # Create filename with timestamp
                        filename = f"frame_{timestamp_seconds}_{timestamp_decimal:06d}.jpg"
                        filepath = os.path.join(self.save_directory, filename)
                        
                        # Save frame
                        cv2.imwrite(filepath, frame)
                        
                        # Update frame data with file path
                        frame_data['file_path'] = filepath
                        
                        # Update save position
                        self.save_position = sequence + 1
                        
                        # Mark task as done
                        self.frame_queue.task_done()
                else:
                    time.sleep(0.5)
                
            except Exception as e:
                raise ValueError((f"Error in save thread: {e}"))
    
    def _ensure_cap_open(self):
        """Ensure camera capture is open with configured parameters."""
        if self.cap is None:
            # Choose backend based on operating system
            system = platform.system().lower()
            if system == "windows":
                # Windows: Use DirectShow backend
                self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
            elif system == "linux":
                # Linux: Use V4L2 backend
                self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_V4L2)
            else:
                # Default backend for other systems
                self.cap = cv2.VideoCapture(self.camera_index)
            
            if not self.cap.isOpened():
                raise ValueError(f"Could not open camera {self.camera_index} using {system} backend")
            
            # Apply configuration if available
            if hasattr(self, '_camera_params') and self._camera_params:
                params = self._camera_params
                
                # Set resolution
                if 'width' in params and 'height' in params:
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, params['width'])
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, params['height'])
                    
                    # Verify resolution setting
                    actual_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    actual_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    if actual_width != params['width'] or actual_height != params['height']:
                        raise ValueError(f"Failed to set resolution to {params['width']}x{params['height']}, got {actual_width}x{actual_height}")
                
                # Set fps
                if 'fps' in params:
                    self.cap.set(cv2.CAP_PROP_FPS, params['fps'])
                    
                    # Verify fps setting (round to nearest integer for comparison)
                    actual_fps = round(self.cap.get(cv2.CAP_PROP_FPS))
                    if actual_fps != params['fps']:
                        raise ValueError(f"Failed to set FPS to {params['fps']}, got {actual_fps}")
                
                # Set encoding format
                if 'encoding' in params and params['encoding']:
                    encoding_map = {
                        'mjpeg': cv2.VideoWriter_fourcc(*'MJPG'),
                        'jmpg': cv2.VideoWriter_fourcc(*'MJPG'),
                        'h264': cv2.VideoWriter_fourcc(*'H264'),
                        'yuyv': cv2.VideoWriter_fourcc(*'YUYV'),
                    }
                    if params['encoding'].lower() in encoding_map:
                        expected_fourcc = encoding_map[params['encoding'].lower()]
                        self.cap.set(cv2.CAP_PROP_FOURCC, expected_fourcc)
                        
                        # Verify encoding setting - throw exception if failed
                        actual_fourcc_num = int(self.cap.get(cv2.CAP_PROP_FOURCC))
                        if actual_fourcc_num != expected_fourcc:
                            # Convert to string for better error message
                            actual_fourcc_str = "".join([chr((actual_fourcc_num >> 8 * i) & 0xFF) for i in range(4)])
                            expected_fourcc_str = "".join([chr((expected_fourcc >> 8 * i) & 0xFF) for i in range(4)])
                            
                            # Check if it's a similar format (like YUYV vs YUY2)
                            if actual_fourcc_num == 0:
                                raise ValueError(f"Encoding {params['encoding']} not supported by camera")
                            elif actual_fourcc_str.strip() == "":
                                raise ValueError(f"Encoding {params['encoding']} not supported by camera")
                            else:
                                raise ValueError(f"Failed to set encoding to {params['encoding']} ({expected_fourcc_str}), camera returned {actual_fourcc_str}")
                    else:
                        raise ValueError(f"Unsupported encoding format: {params['encoding']}. Supported formats: {list(encoding_map.keys())}")
                
                # Set exposure
                if 'exposure' in params and params['exposure'] is not None:
                    self.cap.set(cv2.CAP_PROP_EXPOSURE, params['exposure'])
                    
                    # Verify exposure setting - throw exception if failed (allow 10 unit tolerance)
                    actual_exposure = self.cap.get(cv2.CAP_PROP_EXPOSURE)
                    if (actual_exposure != params['exposure']):  # Allow 10 unit tolerance
                        raise ValueError(f"Failed to set exposure to {params['exposure']}, camera returned {actual_exposure}")
                    
    
    def next_frame(self) -> Tuple[int, np.ndarray, float]:
        """
        Return the next frame from the cached queue.
        
        Returns:
            Tuple containing frame index, frame data, and timestamp in ms
            
        Raises:
            StopIteration: When no frames are available in queue
        """
        try:
            # Get frame from queue (blocking with timeout)
            frame_data = self.frame_queue.get(timeout=5.0)
            
            # Extract data from frame structure
            frame_index = frame_data['frame_index']
            frame = frame_data['frame_data']
            timestamp_ms = frame_data['timestamp_ms']
            
            # Mark task as done
            self.frame_queue.task_done()
            
            return frame_index, frame, timestamp_ms
            
        except queue.Empty:
            raise StopIteration("No frames available in queue") 
    
    def count(self) -> int:
        """
        Return current frame count.
        """
        return self.current_index
    
    def get_queue_size(self) -> int:
        """
        Get current queue size.
        
        Returns:
            Number of frames currently in queue
        """
        return self.frame_queue.qsize()
    
    def get_save_directory(self) -> str:
        """
        Get the directory where frames are being saved.
        
        Returns:
            Path to save directory
        """
        return self.save_directory
    
    def get_save_position(self) -> int:
        """
        Get the current save position.
        
        Returns:
            Current save position (sequence number)
        """
        return self.save_position
    
    def next_frame_data(self) -> dict:
        """
        Return the next frame with complete data structure.
        
        Returns:
            Dictionary containing complete frame data:
            {
                'sequence': int,  # 序号，递增
                'frame_index': int,  # 帧的序号
                'frame_data': np.ndarray,  # 帧的数据
                'timestamp_ms': float,  # 时间戳
                'file_path': str or None  # 落盘文件的路径，没有落盘前为None
            }
            
        Raises:
            StopIteration: When no frames are available in queue
        """
        try:
            # Get frame from queue (blocking with timeout)
            frame_data = self.frame_queue.get(timeout=5.0)
            
            # Mark task as done
            self.frame_queue.task_done()
            
            return frame_data
            
        except queue.Empty:
            raise StopIteration("No frames available in queue")
    
    def get_camera_info(self) -> dict:
        """
        Get current camera information and settings.
        
        Returns:
            Dictionary containing camera properties
        """
        self._ensure_cap_open()
        
        return {
            'width': int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            'height': int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            'fps': self.cap.get(cv2.CAP_PROP_FPS),
            'fourcc': int(self.cap.get(cv2.CAP_PROP_FOURCC)),
            'exposure': self.cap.get(cv2.CAP_PROP_EXPOSURE),
            'auto_exposure': self.cap.get(cv2.CAP_PROP_AUTO_EXPOSURE),
            'brightness': self.cap.get(cv2.CAP_PROP_BRIGHTNESS),
            'contrast': self.cap.get(cv2.CAP_PROP_CONTRAST),
        }
    
    def stop(self):
        """Stop capture and save threads."""
        self.running = False
        self.stop_event.set()
        
        # Wait for threads to finish
        if self.capture_thread and self.capture_thread.is_alive():
            self.capture_thread.join(timeout=2.0)
        if self.save_thread and self.save_thread.is_alive():
            self.save_thread.join(timeout=2.0)
    
    def __del__(self):
        """Clean up camera capture and threads when object is destroyed."""
        self.stop()
        if self.cap is not None:
            self.cap.release()
