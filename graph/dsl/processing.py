"""
Processing engine for executing the AST pipeline.

This module contains the ProcessingEngine class that executes
the operations defined in the AST nodes.
"""

import cv2
import numpy as np
import os
import threading
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Any, Optional, Tuple
from .core import Data, Node
from .utils import GridWorker, ImageUtils
try:
    from tqdm import tqdm
    TQDM_AVAILABLE = True
except ImportError:
    TQDM_AVAILABLE = False


class ProcessingEngine:
    """Engine for executing the processing pipeline defined by AST nodes."""
    
    def __init__(self, data: Data):
        """
        Initialize the processing engine.
        
        Args:
            data: Data object containing frames and AST
        """
        self.data = data
        self.processed_frames: List[np.ndarray] = []
        self.outputs: List[Dict[str, Any]] = []
        self.change_detection_state = {
            'active': False,
            'started': False,
            'include_area': None,
            'exclude_area': None,
            'grid': None,
            'effective_mask': None,
            'previous_frame': None,
            'ready': False,  # skip detection on the same frame as start
            'stopped': False,
            'method': 'ssim',
            'threshold': 0.1,
            'stable_frames': 5,
            'stable_duration_ms': None,
            'phase': 'waiting',
            'change_start_frame': None,
            'stable_count': 0,
            'last_change_time': None
        }
        # Change stats for progress and summary
        self.change_stats = {
            'first_change_frame': None,   # 第一次检测到变化的帧
            'last_change_frame': None,    # 最近一次检测到变化的帧
            'num_change_frames': 0        # 开始到结束之间累计变化帧数量
        }
        self.debug_writer = None
        self.debug_frame_size: Optional[Tuple[int, int]] = None
        self.verbose_writer = None
        
        # Thread pool for parallel grid computation (created once, reused for all frames)
        max_workers = max(1, os.cpu_count() or 1)
        self.thread_pool = ThreadPoolExecutor(max_workers=max_workers)
        
        # State management
        self.state = 'init'  # init, start, stop, stable
        self.change_frames = []  # List of frames with changes
        self.frames_after_detection = 0      # Counter for frames after detection
        self.detect_type = None  # Will be set from AST
        self.delay_stop_frames = getattr(data, 'delay_stop_frames', 10)  # Default frames to delay before stopping
        self.stable_frames = None  # Default stable frames, will be set from detect node
        self.mask_cache = {}  # Cache for masks
        self.previous_frame = None  # Previous frame for comparison

    def __del__(self):
        """Cleanup resources when ProcessingEngine is destroyed."""
        # Shutdown thread pool if it exists
        if hasattr(self, 'thread_pool') and self.thread_pool is not None:
            self.thread_pool.shutdown(wait=True)
        
        # Close debug writer if it exists
        if hasattr(self, 'debug_writer') and self.debug_writer is not None:
            self.debug_writer.release()
        
        # Close verbose writer if it exists
        if hasattr(self, 'verbose_writer') and self.verbose_writer is not None:
            self.verbose_writer.close()

    def execute(self) -> List[Dict[str, Any]]:
        """
        Execute the complete processing pipeline.
        
        Returns:
            List of processing results for each frame
        """
        # Setup debug writer if enabled
        if self.data.debug_flag:
            # Get frame dimensions from first frame of source
            try:
                # Get first frame to determine dimensions
                first_frame_idx, first_frame, _ = next(iter(self.data.source))
                h0, w0 = first_frame.shape[:2]
                self.debug_frame_size = (w0, h0)
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                # Ensure directory exists
                try:
                    import os
                    os.makedirs(self.data.debug_dir, exist_ok=True)
                except Exception:
                    pass
                debug_path = f"{self.data.debug_dir}/{self.data.debug_filename}"
                self.debug_writer = cv2.VideoWriter(debug_path, fourcc, 4.0, (w0, h0))
            except (StopIteration, Exception) as e:
                print(f"Warning: Could not setup debug writer: {e}")
                self.debug_writer = None

        # Setup verbose writer if enabled
        if self.data.verbose_flag:
            try:
                import os
                os.makedirs(self.data.verbose_dir, exist_ok=True)
            except Exception:
                pass
            verbose_path = f"{self.data.verbose_dir}/{self.data.verbose_filename}"
            self.verbose_writer = open(verbose_path, 'w', encoding='utf-8')

        # Initialize state management
        self._initialize_state_management()

        # Prepare frame iterator: use Source interface
        total_frames = self.data.source.count()

        # Process each frame through the pipeline with progress bar
        
        if TQDM_AVAILABLE:
            progress_bar = tqdm(
                total=total_frames,
                desc="处理帧",
                unit="帧",
                bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]"
            )
        else:
            progress_bar = None
            if total_frames:
                print(f"开始处理 {total_frames} 帧...")
            else:
                print("开始处理帧...")
        
        # Main processing loop using Source interface
        try:
            for frame_idx, frame, timestamp_ms in self.data.source:
                # Store timestamp for later use
                self.data.frame_offsets.append(timestamp_ms)
                
                # Process frame and get detection result
                detection_result = self._process_frame_detection(frame_idx, frame)
                
                # Call debug function
                self._debug_callback(frame, frame_idx, detection_result)
                
                # Call verbose function
                self._verbose_callback(frame_idx, detection_result)
                
                # State management processing
                should_exit = self._state_management(detection_result)
                
                # Store result
                result = {
                    'frame_idx': frame_idx,
                    'timestamp': self.data.frame_offsets[frame_idx] if frame_idx < len(self.data.frame_offsets) else 0,
                    'change_detection': detection_result
                }
                self.outputs.append(result)
                
                # Check if we should exit early
                if should_exit:
                    print(f"检测完成，在第 {frame_idx} 帧退出")
                    break
                
                # Update progress
                if progress_bar:
                    # Update internal change stats
                    if detection_result:
                        self._update_change_stats(frame_idx, detection_result)
                    
                    progress_bar.set_postfix({
                        '状态': '处理中',
                        '起始帧': self.change_stats['first_change_frame'] if self.change_stats['first_change_frame'] is not None else '-',
                        '结束帧': self.change_stats['last_change_frame'] if self.change_stats['last_change_frame'] is not None else '-',
                        '变化帧数': self.change_stats['num_change_frames']
                    })
                    progress_bar.update(1)
                else:
                    # Simple progress without tqdm
                    if (frame_idx + 1) % max(1, total_frames // 10) == 0 or frame_idx == total_frames - 1:
                        progress = (frame_idx + 1) / total_frames * 100
                        if detection_result:
                            self._update_change_stats(frame_idx, detection_result)
                        print(f"进度: {progress:.1f}% ({frame_idx + 1}/{total_frames})")
        except StopIteration:
            # Source has no more frames
            pass
        
        if progress_bar:
            progress_bar.close()
        
        # Release debug writer
        if self.debug_writer is not None:
            self.debug_writer.release()
            self.debug_writer = None

        # Close verbose writer
        if self.verbose_writer is not None:
            self.verbose_writer.close()
            self.verbose_writer = None

        # Cleanup is handled by Source objects automatically

        # Convert outputs to the new JSON format
        return self._format_results()

    def _format_results(self) -> Dict[str, Any]:
        """
        Format the processing results into the specified JSON structure.
        
        Returns:
            Dictionary with 'params' and 'result' keys
        """
        # Collect parameters from AST
        params = {}
        
        # Collect all parameters from AST nodes
        for node in self.data.ast:
            if node.op_name == "Config":
                params["change_method"] = node.params.get("method", "ssim")
                params["change_threshold"] = node.params.get("threshold", 0.1)
                params["grid_size"] = node.params.get("grid_size", (32, 16))
            elif node.op_name == "ROI":
                params["roi_crops"] = node.params.get("crops", [])
            elif node.op_name == "Filter":
                params["filter_method"] = node.params.get("method", "gaussian")
                params["filter_ksize"] = node.params.get("ksize", 5)
            elif node.op_name == "detect":
                # Detect type parameter
                params["detect_type"] = node.params.get("type", "both")
                
                # Unconditionally save all area parameters (unconditional save, conditional use)
                params["include_area"] = node.params.get("include_area")
                params["exclude_area"] = node.params.get("exclude_area")
                params["start_include_area"] = node.params.get("start_include_area")
                params["start_exclude_area"] = node.params.get("start_exclude_area")
                params["stop_include_area"] = node.params.get("stop_include_area")
                params["stop_exclude_area"] = node.params.get("stop_exclude_area")
                params["stable_frames"] = node.params.get("stable_frames", 5)
                params["stable_duration_ms"] = node.params.get("stable_duration_ms")
                
                # Method and threshold are now in Config node
            elif node.op_name == "Debug":
                # Debug configuration parameters
                params["debug_enabled"] = node.params.get("enabled", True)
                import os
                debug_dir = node.params.get("output_dir", ".")
                params["debug_dir"] = os.path.abspath(debug_dir)
                params["debug_filename"] = node.params.get("filename", "debug.mp4")
                params["delay_stop_frames"] = node.params.get("delay_stop_frames", 10)
            elif node.op_name == "Verbose":
                # Verbose configuration parameters
                params["verbose_enabled"] = node.params.get("enabled", True)
                import os
                verbose_dir = node.params.get("output_dir", ".")
                params["verbose_dir"] = os.path.abspath(verbose_dir)
                params["verbose_filename"] = node.params.get("filename", "verbose.txt")
        
        # Format results based on detect_type and change_frames
        result = {}
        detect_type = params.get("detect_type", "both")
        threshold = params.get("change_threshold", 0.1)

        # Process change frames based on type
        if detect_type == "start":
            # For start detection, include all change frames
            frames_to_process = self.change_frames[:1] 
        elif detect_type == "stop":
            # For stop detection, include all change frames
            frames_to_process = self.change_frames[-1:]
        elif detect_type == "both":
            # For both detection, include all change frames
            frames_to_process = self.change_frames
        else:
            frames_to_process = []
        
        # Get first frame timestamp for calculating relative offset
        first_frame_timestamp = None
        if self.data.frame_offsets:
            first_frame_timestamp = self.data.frame_offsets[0]
        
        # Process each change frame
        for frame_idx in frames_to_process:
            # Get original timestamp for this frame
            original_timestamp_ms = self.data.frame_offsets[frame_idx] if frame_idx < len(self.data.frame_offsets) else None
            
            # Calculate relative offset from first frame
            if original_timestamp_ms is not None and first_frame_timestamp is not None:
                relative_offset_ms = original_timestamp_ms - first_frame_timestamp
            else:
                relative_offset_ms = None
            
            # Find the corresponding output data
            output_data = None
            for output in self.outputs:
                if output['frame_idx'] == frame_idx:
                    output_data = output
                    break
            
            if not output_data:
                continue
                
            change_detection = output_data.get('change_detection')
            if not change_detection:
                continue
            
            # Skip initialization frames
            if change_detection.get('action') == 'initialized':
                continue
            
            # Only include frames with changes
            changed_cells_count = change_detection.get('changed_cells_count', 0)
            if changed_cells_count > 0:
                # Get grid matrices
                grid_changes_matrix = change_detection.get('grid_changes_matrix', [])
                grid_scores_matrix = change_detection.get('grid_scores_matrix', [])
                
                if not grid_changes_matrix or not grid_scores_matrix:
                    continue
                
                m, n = len(grid_changes_matrix), len(grid_changes_matrix[0])
                
                # Create list of over-threshold grid cells with normalized coordinates
                over_threshold_cells = []
                for i in range(m):
                    for j in range(n):
                        if i < len(grid_changes_matrix) and j < len(grid_changes_matrix[i]):
                            if grid_changes_matrix[i][j] and grid_scores_matrix[i][j] >= threshold:
                                # Calculate normalized coordinates
                                # (x, y) is top-left, (x1, y1) is bottom-right
                                x = j / n  # left edge
                                y = i / m  # top edge
                                x1 = (j + 1) / n  # right edge
                                y1 = (i + 1) / m  # bottom edge
                                
                                over_threshold_cells.append([
                                    round(x, 4), round(y, 4), 
                                    round(x1, 4), round(y1, 4), 
                                    round(grid_scores_matrix[i][j], 4)
                                ])
                
                if over_threshold_cells:  # Only add if there are over-threshold cells
                    result[str(frame_idx)] = {
                        "offset": round(relative_offset_ms, 2) if relative_offset_ms is not None else None,  # Relative offset from first frame (ms)
                        "ts": round(original_timestamp_ms, 2) if original_timestamp_ms is not None else None,   # Original timestamp (ms)
                        "change": over_threshold_cells
                    }
        
        return {
            "params": params,
            "result": result
        }

    def _update_change_stats(self, frame_idx: int, change_detection: Dict[str, Any]) -> None:
        """
        Update stats about first/last change frames and count of change frames.
        判定规则：当 changed_cells > 0 视为该帧发生变化。
        """
        changed_cells = change_detection.get('changed_cells_count', change_detection.get('changed_cells'))
        total_cells = change_detection.get('total_cells')
        if changed_cells is None or total_cells is None:
            return
        if changed_cells > 0:
            # First change frame
            if self.change_stats['first_change_frame'] is None:
                self.change_stats['first_change_frame'] = frame_idx
            # Last change frame always updates
            self.change_stats['last_change_frame'] = frame_idx
            # Count change frames between start and end (inclusive as it accrues)
            self.change_stats['num_change_frames'] += 1



    def _apply_roi(self, img: np.ndarray, params: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply ROI extraction.
        
        Args:
            img: Input image
            params: ROI parameters
            
        Returns:
            Tuple of (processed_image, operation_result)
        """
        crops = params.get("crops", [])
        if not crops:
            return img, {"operation": "ROI", "applied": False, "reason": "No crops specified"}
        
        try:
            cropped_imgs = []
            for x1, y1, x2, y2 in crops:
                # Ensure coordinates are within image bounds
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(img.shape[1], x2), min(img.shape[0], y2)
                if x2 > x1 and y2 > y1:
                    cropped_imgs.append(img[y1:y2, x1:x2])
            
            if cropped_imgs:
                img = cv2.hconcat(cropped_imgs)
                return img, {
                    "operation": "ROI", 
                    "applied": True, 
                    "crops_count": len(cropped_imgs),
                    "output_shape": img.shape
                }
            else:
                return img, {"operation": "ROI", "applied": False, "reason": "No valid crops"}
        except Exception as e:
            return img, {"operation": "ROI", "applied": False, "error": str(e)}

    def _apply_filter(self, img: np.ndarray, params: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Apply filtering operations.
        
        Args:
            img: Input image
            params: Filter parameters
            
        Returns:
            Tuple of (processed_image, operation_result)
        """
        method = params.get("method", "gaussian")
        ksize = params.get("ksize", 5)
        
        if method == "gaussian":
            # Ensure ksize is odd
            ksize = ksize if ksize % 2 == 1 else ksize + 1
            img = cv2.GaussianBlur(img, (ksize, ksize), 0)
        elif method == "median":
            # Ensure ksize is odd
            ksize = ksize if ksize % 2 == 1 else ksize + 1
            img = cv2.medianBlur(img, ksize)
        else:
            raise ValueError(f"Unsupported filter method: {method}. Supported methods: 'gaussian', 'median'")
        
        return img, {
            "operation": "Filter", 
            "applied": True, 
            "method": method, 
            "ksize": ksize
        }



    def _compute_grid_change_parallel(self, prev: np.ndarray, curr: np.ndarray, 
                                    effective_mask: np.ndarray, grid: Tuple[int, int],
                                    method: str = "ssim", threshold: Optional[float] = None):
        """
        Compute grid-based changes using parallel processing.
        
        Args:
            prev: Previous frame
            curr: Current frame
            effective_mask: Mask for valid areas
            grid: Grid dimensions (rows, cols)
            method: Comparison method ("abs" or "ssim")
            threshold: Normalized threshold in [0,1]. 0 = no difference, 1 = 100% difference.
                Default: 0.01 for "abs" method, 0.1 for "ssim" method.
            
        Returns:
            List of tuples (changed: bool, score: float) for each grid cell
        """
        # Set method-specific default threshold
        if threshold is None:
            if method == "ssim":
                threshold = 0.1
            elif method == "abs":
                threshold = 0.01
            else:
                raise ValueError(f"Unsupported change detection method: {method}. Supported methods: 'ssim', 'abs'")
                
        h, w = prev.shape[:2]
        m, n = grid
        cell_h = h // m
        cell_w = w // n
        
        # Prepare tasks for parallel processing
        tasks = []
        for i in range(m):
            for j in range(n):
                y1, y2 = i * cell_h, (i + 1) * cell_h
                x1, x2 = j * cell_w, (j + 1) * cell_w
                tasks.append((prev, curr, effective_mask, (y1, y2, x1, x2), method, threshold))
        
        # Use instance thread pool for parallel computation (reused across all frames)
        try:
            if TQDM_AVAILABLE:
                results = list(tqdm(
                    self.thread_pool.map(GridWorker.process_cell, tasks),
                    total=len(tasks),
                    desc="网格检测",
                    unit="格",
                    leave=False
                ))
                return results
            else:
                results = list(self.thread_pool.map(GridWorker.process_cell, tasks))
                return results

        except Exception as e:
            print(f"Warning: Threaded processing failed, falling back to sequential: {e}")
            # Fallback to sequential processing
            if TQDM_AVAILABLE:
                return list(tqdm(
                    (GridWorker.process_cell(task) for task in tasks),
                    total=len(tasks),
                    desc="网格检测(顺序)",
                    unit="格",
                    leave=False
                ))
            else:
                return [GridWorker.process_cell(task) for task in tasks]

    def _initialize_state_management(self):
        """Initialize state management based on AST nodes."""
        # Get detection type and parameters from AST
        for node in self.data.ast:
            if node.op_name == "detect":
                # Get stable frames parameter
                self.stable_frames = node.params.get("stable_frames", 60)
                self.detect_type = node.params.get("type", "both")
                
            elif node.op_name == "Debug":
                # Get delay_stop_frames parameter from debug node
                self.delay_stop_frames = node.params.get("delay_stop_frames", 10)

    def _process_frame_detection(self, frame_idx: int, frame: np.ndarray) -> Optional[Dict[str, Any]]:
        """
        Process frame for change detection.
        
        Args:
            frame_idx: Frame index
            frame: Current frame
            
        Returns:
            Detection result with frame_idx, changed_cells_count, grid_changes_matrix, grid_scores_matrix
        """
        # Apply preprocessing operations (ROI, Filter)
        processed_frame = frame.copy()
        
        for node in self.data.ast:
            if node.op_name == "ROI":
                processed_frame, _ = self._apply_roi(processed_frame, node.params)
            elif node.op_name == "Filter":
                processed_frame, _ = self._apply_filter(processed_frame, node.params)
        
        # Get change detection parameters
        change_detection_node = None
        for node in self.data.ast:
            if node.op_name == "detect":
                change_detection_node = node
                break
        
        if not change_detection_node:
            return None
        
        # Get mask based on current state
        effective_mask = self._get_effective_mask(processed_frame, change_detection_node)
        
        # Get configuration from Config node
        method = "ssim"
        threshold = 0.1
        grid_size = (32, 16)
        
        for node in self.data.ast:
            if node.op_name == "Config":
                method = node.params.get("method", "ssim")
                threshold = node.params.get("threshold", 0.1)
                grid_size = node.params.get("grid_size", (32, 16))
                break
        
        m, n = grid_size
        
        # Initialize previous frame if first frame
        if frame_idx == 0 or self.previous_frame is None:
            self.previous_frame = processed_frame.copy()
            return {
                "frame_idx": frame_idx,
                "action": "initialized",
                "changed_cells_count": 0,
                "grid_changes_matrix": [[False] * n for _ in range(m)],
                "grid_scores_matrix": [[0.0] * n for _ in range(m)]
            }
        
        # Perform grid-based change detection
        changed_and_scores = self._compute_grid_change_parallel(
            self.previous_frame,
            processed_frame,
            effective_mask,
            (m, n),
            method,
            threshold
        )
        
        # Convert to matrices
        grid_changes_matrix = []
        grid_scores_matrix = []
        changed_cells_count = 0
        
        for i in range(m):
            changes_row = []
            scores_row = []
            for j in range(n):
                idx = i * n + j
                if idx < len(changed_and_scores):
                    changed, score = changed_and_scores[idx]
                    changes_row.append(changed)
                    scores_row.append(score)
                    if changed:
                        changed_cells_count += 1
                else:
                    changes_row.append(False)
                    scores_row.append(0.0)
            grid_changes_matrix.append(changes_row)
            grid_scores_matrix.append(scores_row)
        
        # Update previous frame
        self.previous_frame = processed_frame.copy()
        
        return {
            "frame_idx": frame_idx,
            "action": "detected",
            "changed_cells_count": changed_cells_count,
            "grid_changes_matrix": grid_changes_matrix,
            "grid_scores_matrix": grid_scores_matrix
        }

    def _get_effective_mask(self, frame: np.ndarray, change_detection_node: Node) -> np.ndarray:
        """
        Get effective mask based on current state.
        Before first change: use start areas + common areas
        After first change: use stop areas + common areas
        """
        h, w = frame.shape[:2]
        
        # Get common areas
        common_include = change_detection_node.params.get("include_area", [])
        common_exclude = change_detection_node.params.get("exclude_area", [])
        
        # Determine which specific areas to use based on state
        if self.state in ['init', 'start']:
            # Use start-specific areas if available
            start_include = change_detection_node.params.get("start_include_area", [])
            start_exclude = change_detection_node.params.get("start_exclude_area", [])
            # Handle None values
            if start_include is None:
                start_include = []
            if start_exclude is None:
                start_exclude = []
            include_areas = common_include + start_include
            exclude_areas = common_exclude + start_exclude
        else:
            # Use stop-specific areas if available
            stop_include = change_detection_node.params.get("stop_include_area", [])
            stop_exclude = change_detection_node.params.get("stop_exclude_area", [])
            # Handle None values
            if stop_include is None:
                stop_include = []
            if stop_exclude is None:
                stop_exclude = []
            include_areas = common_include + stop_include
            exclude_areas = common_exclude + stop_exclude
        
        # Create cache key
        cache_key = f"{w}x{h}_{hash(str(include_areas))}_{hash(str(exclude_areas))}"
        
        # Check cache
        if cache_key in self.mask_cache:
            return self.mask_cache[cache_key]
        
        # Convert to pixel coordinates
        include_area_pixels = self.data._normalize_or_pixel_coords(include_areas, w, h)
        exclude_area_pixels = self.data._normalize_or_pixel_coords(exclude_areas, w, h)
        
        # Create masks
        include_mask = np.zeros((h, w), np.uint8)
        for x1, y1, x2, y2 in include_area_pixels:
            include_mask[y1:y2, x1:x2] = 255
        
        exclude_mask = np.zeros((h, w), np.uint8)
        for x1, y1, x2, y2 in exclude_area_pixels:
            exclude_mask[y1:y2, x1:x2] = 255
        
        # If no include areas are specified, use the full frame as valid area
        if not include_area_pixels:
            include_mask = np.full((h, w), 255, np.uint8)
        
        effective_mask = cv2.bitwise_and(include_mask, cv2.bitwise_not(exclude_mask))
        
        # Cache the result
        self.mask_cache[cache_key] = effective_mask
        
        return effective_mask

    def _debug_callback(self, frame: np.ndarray, frame_idx: int, detection_result: Optional[Dict[str, Any]]):
        """Debug callback function."""
        if self.debug_writer is None or detection_result is None:
            return
        
        # Create debug overlay
        overlay = frame.copy()
        
        # Draw include and exclude areas first
        self._draw_include_exclude_areas(overlay, frame)
        
        # Draw grid
        if 'grid_changes_matrix' in detection_result:
            grid_changes = detection_result['grid_changes_matrix']
            grid_scores = detection_result['grid_scores_matrix']
            m, n = len(grid_changes), len(grid_changes[0])
            
            h, w = frame.shape[:2]
            cell_h = h // m
            cell_w = w // n
            
            # Draw grid lines with semi-transparent effect
            grid_overlay = ImageUtils.create_grid_overlay(overlay, (m, n), color=(128, 128, 128), thickness=1)
            overlay = cv2.addWeighted(overlay, 0.8, grid_overlay, 0.2, 0)
            
            # Highlight changed cells and show scores
            for i in range(m):
                for j in range(n):
                    y1, y2 = i * cell_h, (i + 1) * cell_h
                    x1, x2 = j * cell_w, (j + 1) * cell_w
                    
                    if grid_changes[i][j]:
                        # Draw thick yellow border for changed cells
                        # Clamp coordinates to ensure border stays within image bounds
                        x1_clamped, y1_clamped, x2_clamped, y2_clamped = self._clamp_rectangle_to_bounds(
                            x1, y1, x2, y2, 4, overlay.shape[1], overlay.shape[0])
                        cv2.rectangle(overlay, (x1_clamped, y1_clamped), (x2_clamped, y2_clamped), (0, 255, 255), 4)  # Yellow border, thick
                    
                    # Show score
                    score = grid_scores[i][j]
                    if score > 0.0:
                        # Check if score is less than 0.001
                        if score < 0.001:
                            text = "..."
                        else:
                            # Format to 3 decimal places, remove leading zero if < 1.0
                            if score < 1.0:
                                text = f"{score:.3f}"[1:]  # Remove leading "0"
                            else:
                                text = f"{score:.3f}"
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.4
                        thickness = 1
                        (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)
                        center_x = x1 + (cell_w // 2)
                        center_y = y1 + (cell_h // 2)
                        text_x = int(center_x - text_w / 2)
                        # putText uses baseline at y; align vertically by adding half height
                        text_y = int(center_y + text_h / 2)
                        
                        # Add white semi-transparent background for better visibility
                        padding = 3
                        bg_x1 = text_x - padding
                        bg_y1 = text_y - text_h - padding
                        bg_x2 = text_x + text_w + padding
                        bg_y2 = text_y + padding
                        
                        # Draw semi-transparent white background
                        bg_color = (255, 255, 255)  # White background
                        cv2.rectangle(overlay, (bg_x1, bg_y1), (bg_x2, bg_y2), bg_color, -1)
                        
                        # Choose text color - black text for all cells
                        text_color = (0, 0, 0)  # Black text
                        cv2.putText(overlay, text, (text_x, text_y), font, font_scale, text_color, thickness)
        
        # Use pure overlay without blending to avoid color mixing
        debug_frame = overlay
        
        # Draw frame index and state
        cv2.putText(debug_frame, f"Frame {frame_idx} - State: {self.state}", (8, 22), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        self.debug_writer.write(debug_frame)

    def _verbose_callback(self, frame_idx: int, detection_result: Optional[Dict[str, Any]]):
        """Verbose callback function."""
        if self.verbose_writer is None or detection_result is None:
            return
        
        self.verbose_writer.write(f"Frame {frame_idx} - State: {self.state}\n")
        
        if 'grid_scores_matrix' in detection_result:
            grid_scores = detection_result['grid_scores_matrix']
            m, n = len(grid_scores), len(grid_scores[0])
            
            self.verbose_writer.write(f"Grid {m}x{n} scores:\n")
            for i in range(m):
                row_scores = []
                for j in range(n):
                    score = grid_scores[i][j]
                    if score == -1.0:
                        row_scores.append("-1.000")
                    else:
                        row_scores.append(f"{score:.4f}")
                self.verbose_writer.write("  " + " ".join(row_scores) + "\n")
        
        self.verbose_writer.write(f"Changed cells: {detection_result.get('changed_cells_count', 0)}\n")
        self.verbose_writer.write("\n")
        self.verbose_writer.flush()

    def _state_management(self, detection_result: Optional[Dict[str, Any]]) -> bool:
        """
        State management processing based on detect type.
        
        Returns:
            True if should exit, False if should continue
        """
        if detection_result is None:
            return False
        
        frame_idx = detection_result.get('frame_idx', 0)
        changed_cells_count = detection_result.get('changed_cells_count', 0)
        has_changes = changed_cells_count > 0
        
        # Record change frames if there are changes
        if has_changes:
            self.change_frames.append(frame_idx)
        
        # Different state machine logic based on detect_type
        if self.detect_type == "start":
            return self._state_machine_start(frame_idx, has_changes)
        elif self.detect_type in ["stop", "both"]:
            # stop and both have identical logic
            return self._state_machine_stop(frame_idx, has_changes)
        else:
            # raise error
            raise ValueError(f"Invalid detect type: {self.detect_type}")

    
    def _state_machine_start(self, frame_idx: int, has_changes: bool) -> bool:
        """State machine for start detection only."""
        if self.state == 'init':
            if frame_idx == 0:
                return False
            else:
                # Start detection from frame 1
                self.state = 'start'
                return False
        
        elif self.state == 'start':
            if len(self.change_frames) > 0:
                # if has detected first change frame, switch to stop state
                self.state = 'delay_stop'# if has changes, record the frame index
        
        elif self.state == 'delay_stop' and self.delay_stop_frames is not None:
            
            self.frames_after_detection += 1
            if self.frames_after_detection >= self.delay_stop_frames:
                return True  # Exit
        
        return False
        
    
    def _state_machine_stop(self, frame_idx: int, has_changes: bool) -> bool:
        """State machine for stop detection only."""
        if self.state == 'init':
            if frame_idx == 0:
                return False
            else:
                # Start detection from frame 1
                self.state = 'start'
                return False
        
        elif self.state == 'start':
            if len(self.change_frames) > 0:
                # if has detected first change frame, switch to stop state
                self.state = 'stop'
        
        elif self.state == 'stop':
            if has_changes:
                self.frames_after_detection = 0
            else:
                # No changes, count stable frames
                self.frames_after_detection += 1
                if self.frames_after_detection >= self.stable_frames:
                    self.state = 'delay_stop'
                    return True  # Exit after stable period
        
        elif self.state == 'delay_stop':
            self.frames_after_detection += 1
            if self.frames_after_detection >= self.delay_stop_frames:
                return True  # Exit
        
        return False
    
    

    def _draw_include_exclude_areas(self, overlay: np.ndarray, frame: np.ndarray):
        """
        Draw include and exclude area boxes on the overlay.
        
        Args:
            overlay: The overlay image to draw on
            frame: The original frame for dimensions
        """
        h, w = frame.shape[:2]
        
        # Get change detection parameters from AST
        change_detection_node = None
        for node in self.data.ast:
            if node.op_name == "detect":
                change_detection_node = node
                break
        
        if not change_detection_node:
            return
        
        # Get common include and exclude areas (always drawn)
        common_include = change_detection_node.params.get("include_area", [(0.0, 0.0, 1.0, 1.0)])
        common_exclude = change_detection_node.params.get("exclude_area", [(0.0, 0.0, 0.0, 0.0)])
        
        # Convert common areas to pixel coordinates and draw them
        common_include_pixels = self.data._normalize_or_pixel_coords(common_include, w, h)
        common_exclude_pixels = self.data._normalize_or_pixel_coords(common_exclude, w, h)
        
        # Collect all areas to draw with their colors
        areas_to_draw = []
        
        # Add common areas (always drawn)
        for x1, y1, x2, y2 in common_include_pixels:
            if x2 > x1 and y2 > y1:
                areas_to_draw.append(((x1, y1, x2, y2), (0, 255, 0), "include"))  # Green for include
        
        for x1, y1, x2, y2 in common_exclude_pixels:
            if x2 > x1 and y2 > y1:
                areas_to_draw.append(((x1, y1, x2, y2), (0, 0, 255), "exclude"))  # Red for exclude
        
        # Draw areas based on current state
        if self.state in ['init', 'start']:
            # In init or start state: draw start-specific areas
            start_include = change_detection_node.params.get("start_include_area", [])
            start_exclude = change_detection_node.params.get("start_exclude_area", [])
            
            if start_include or start_exclude:
                start_include_pixels = self.data._normalize_or_pixel_coords(start_include, w, h)
                start_exclude_pixels = self.data._normalize_or_pixel_coords(start_exclude, w, h)
                
                for x1, y1, x2, y2 in start_include_pixels:
                    if x2 > x1 and y2 > y1:
                        areas_to_draw.append(((x1, y1, x2, y2), (0, 255, 0), "start_include"))  # Green for start include
                
                for x1, y1, x2, y2 in start_exclude_pixels:
                    if x2 > x1 and y2 > y1:
                        areas_to_draw.append(((x1, y1, x2, y2), (0, 0, 255), "start_exclude"))  # Red for start exclude
        else:
            # In other states (stop, stable): draw stop-specific areas
            stop_include = change_detection_node.params.get("stop_include_area", [])
            stop_exclude = change_detection_node.params.get("stop_exclude_area", [])
            
            if stop_include or stop_exclude:
                stop_include_pixels = self.data._normalize_or_pixel_coords(stop_include, w, h)
                stop_exclude_pixels = self.data._normalize_or_pixel_coords(stop_exclude, w, h)
                
                for x1, y1, x2, y2 in stop_include_pixels:
                    if x2 > x1 and y2 > y1:
                        areas_to_draw.append(((x1, y1, x2, y2), (0, 255, 0), "stop_include"))  # Green for stop include
                
                for x1, y1, x2, y2 in stop_exclude_pixels:
                    if x2 > x1 and y2 > y1:
                        areas_to_draw.append(((x1, y1, x2, y2), (0, 0, 255), "stop_exclude"))  # Red for stop exclude
        
        # Draw all areas with mixed color support
        self._draw_areas_with_mixed_colors(overlay, areas_to_draw)
    
    def _clamp_rectangle_to_bounds(self, x1: int, y1: int, x2: int, y2: int, 
                                   thickness: int, img_width: int, img_height: int) -> Tuple[int, int, int, int]:
        """
        Clamp rectangle coordinates to ensure the border stays within image bounds.
        
        Args:
            x1, y1, x2, y2: Rectangle coordinates
            thickness: Border thickness
            img_width, img_height: Image dimensions
            
        Returns:
            Clamped coordinates (x1, y1, x2, y2)
        """
        # Calculate border extension (thickness extends outward from the rectangle)
        border_extension = thickness // 2
        
        # Clamp coordinates to ensure border stays within bounds
        # Left border: x1 - border_extension >= 0
        x1_clamped = max(border_extension, x1)
        # Top border: y1 - border_extension >= 0  
        y1_clamped = max(border_extension, y1)
        # Right border: x2 + border_extension < img_width
        x2_clamped = min(img_width - border_extension, x2)
        # Bottom border: y2 + border_extension < img_height
        y2_clamped = min(img_height - border_extension, y2)
        
        return x1_clamped, y1_clamped, x2_clamped, y2_clamped

    def _draw_areas_with_mixed_colors(self, overlay: np.ndarray, areas_to_draw: List[Tuple]):
        """
        Draw areas with mixed color support for overlapping regions.
        
        Args:
            overlay: The overlay image to draw on
            areas_to_draw: List of tuples containing (bbox, color, area_type)
        """
        if not areas_to_draw:
            return
        
        # Create a color accumulation buffer
        h, w = overlay.shape[:2]
        color_buffer = np.zeros((h, w, 3), dtype=np.float32)
        count_buffer = np.zeros((h, w), dtype=np.float32)
        
        # Draw each area and accumulate colors
        for (x1, y1, x2, y2), color, area_type in areas_to_draw:
            # Convert color to float
            color_float = np.array(color, dtype=np.float32)
            
            # Draw thick border by drawing multiple lines
            thickness = 4
            # Clamp coordinates to ensure border stays within image bounds
            x1_clamped, y1_clamped, x2_clamped, y2_clamped = self._clamp_rectangle_to_bounds(
                x1, y1, x2, y2, thickness, w, h)
            
            for t in range(thickness):
                # Top and bottom borders
                for x in range(max(0, x1_clamped-t), min(w, x2_clamped+t+1)):
                    if 0 <= y1_clamped-t < h:
                        color_buffer[y1_clamped-t, x] += color_float
                        count_buffer[y1_clamped-t, x] += 1
                    if 0 <= y2_clamped+t < h:
                        color_buffer[y2_clamped+t, x] += color_float
                        count_buffer[y2_clamped+t, x] += 1
                
                # Left and right borders
                for y in range(max(0, y1_clamped-t), min(h, y2_clamped+t+1)):
                    if 0 <= x1_clamped-t < w:
                        color_buffer[y, x1_clamped-t] += color_float
                        count_buffer[y, x1_clamped-t] += 1
                    if 0 <= x2_clamped+t < w:
                        color_buffer[y, x2_clamped+t] += color_float
                        count_buffer[y, x2_clamped+t] += 1
        
        # Normalize colors and apply to overlay
        for y in range(h):
            for x in range(w):
                if count_buffer[y, x] > 0:
                    # Average the colors
                    avg_color = color_buffer[y, x] / count_buffer[y, x]
                    # Apply with some transparency to show the mixed effect
                    alpha = 0.8
                    overlay[y, x] = (1 - alpha) * overlay[y, x] + alpha * avg_color