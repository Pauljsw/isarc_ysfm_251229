# """
# COLMAP SFM Wrapper
# Automates Structure from Motion using COLMAP.
# """
# import subprocess
# import json
# import numpy as np
# from pathlib import Path
# from typing import Dict, Optional, Tuple
# import logging
# import shutil

# logger = logging.getLogger(__name__)


# class COLMAPRunner:
#     """Wrapper for COLMAP SFM pipeline"""

#     def __init__(self, colmap_executable: str = 'colmap'):
#         """
#         Initialize COLMAP runner.

#         Args:
#             colmap_executable: Path to COLMAP executable
#         """
#         self.colmap_exe = colmap_executable
#         self._check_colmap_installed()
#         self._cuda_available = self._detect_cuda_available()

#     def _check_colmap_installed(self):
#         """Check if COLMAP is installed"""
#         try:
#             result = subprocess.run(
#                 [self.colmap_exe, '--version'],
#                 capture_output=True,
#                 text=True,
#                 timeout=5
#             )
#             if result.returncode == 0:
#                 logger.info(f"COLMAP found: {result.stdout.strip()}")
#             else:
#                 logger.warning("COLMAP executable found but version check failed")
#         except FileNotFoundError:
#             logger.error(f"COLMAP not found at: {self.colmap_exe}")
#             logger.error("Install COLMAP: https://colmap.github.io/install.html")
#             raise RuntimeError("COLMAP not installed")
#         except Exception as e:
#             logger.warning(f"Could not verify COLMAP installation: {e}")

#     def _detect_cuda_available(self) -> bool:
#         """Detect if CUDA is available on the system"""
#         try:
#             # Check if nvidia-smi command works (indicates NVIDIA GPU present)
#             result = subprocess.run(
#                 ['nvidia-smi'],
#                 capture_output=True,
#                 text=True,
#                 timeout=5
#             )
#             if result.returncode == 0:
#                 logger.info("CUDA detected: NVIDIA GPU available")
#                 return True
#             else:
#                 logger.info("CUDA not available: nvidia-smi failed")
#                 return False
#         except FileNotFoundError:
#             logger.info("CUDA not available: nvidia-smi not found")
#             return False
#         except Exception as e:
#             logger.debug(f"CUDA detection failed: {e}")
#             return False
    
#     def run_sfm_pipeline(
#         self,
#         image_dir: str,
#         output_dir: str,
#         camera_model: str = 'OPENCV',
#         use_cuda: str = 'auto',
#         quality: str = 'high'
#     ) -> str:
#         """
#         Run complete COLMAP SFM pipeline.

#         Args:
#             image_dir: Directory containing RGB images
#             output_dir: Output directory for COLMAP results
#             camera_model: Camera model (OPENCV, PINHOLE, RADIAL, etc.)
#             use_cuda: 'auto' (detect), 'true', 'false', or boolean
#             quality: 'low', 'medium', 'high', 'extreme'

#         Returns:
#             Path to sparse reconstruction directory
#         """
#         # Handle use_cuda parameter (auto, true, false, or bool)
#         if isinstance(use_cuda, str):
#             if use_cuda.lower() == 'auto':
#                 use_cuda_bool = self._cuda_available
#                 logger.info(f"CUDA mode: auto (detected={'available' if use_cuda_bool else 'unavailable'})")
#             elif use_cuda.lower() in ('true', '1', 'yes'):
#                 use_cuda_bool = True
#                 logger.info("CUDA mode: enabled (forced)")
#             else:
#                 use_cuda_bool = False
#                 logger.info("CUDA mode: disabled (forced)")
#         else:
#             use_cuda_bool = bool(use_cuda)
#             logger.info(f"CUDA mode: {'enabled' if use_cuda_bool else 'disabled'}")
#         output_path = Path(output_dir)
#         output_path.mkdir(parents=True, exist_ok=True)
        
#         database_path = output_path / 'database.db'
#         sparse_dir = output_path / 'sparse'
#         sparse_dir.mkdir(exist_ok=True)
        
#         logger.info("=" * 80)
#         logger.info("Starting COLMAP SFM Pipeline")
#         logger.info("=" * 80)

#         # Step 1: Feature Extraction
#         logger.info("Step 1/4: Feature Extraction")
#         self._run_feature_extraction(
#             database_path, image_dir, camera_model, use_cuda_bool, quality
#         )

#         # Step 2: Feature Matching
#         logger.info("Step 2/4: Feature Matching")
#         self._run_feature_matching(database_path, use_cuda_bool, quality)
        
#         # Step 3: Sparse Reconstruction
#         logger.info("Step 3/4: Sparse Reconstruction")
#         self._run_mapper(database_path, image_dir, sparse_dir)
        
#         # Step 4: Model Conversion
#         logger.info("Step 4/4: Converting to standard format")
#         model_dir = sparse_dir / '0'  # COLMAP creates numbered models
        
#         if not model_dir.exists():
#             raise RuntimeError(f"Reconstruction failed: {model_dir} not found")
        
#         logger.info(f"SFM reconstruction complete: {model_dir}")
        
#         return str(model_dir)
    
#     def _run_feature_extraction(
#         self,
#         database_path: Path,
#         image_dir: str,
#         camera_model: str,
#         use_gpu: bool,
#         quality: str
#     ):
#         """Extract features from images"""
#         quality_settings = {
#             'low': {'max_image_size': 1600, 'max_num_features': 4096},
#             'medium': {'max_image_size': 2400, 'max_num_features': 8192},
#             'high': {'max_image_size': 3200, 'max_num_features': 16384},
#             'extreme': {'max_image_size': 4800, 'max_num_features': 32768}
#         }

#         settings = quality_settings.get(quality, quality_settings['high'])

#         cmd = [
#             self.colmap_exe, 'feature_extractor',
#             '--database_path', str(database_path),
#             '--image_path', image_dir,
#             '--ImageReader.camera_model', camera_model,
#             '--ImageReader.single_camera', '1',  # Assume same camera for all images
#             '--SiftExtraction.max_image_size', str(settings['max_image_size']),
#             '--SiftExtraction.max_num_features', str(settings['max_num_features']),
#         ]

#         if use_gpu:
#             cmd.extend(['--SiftExtraction.use_gpu', '1'])

#         # Try with GPU flag, fallback to CPU if GPU not supported
#         try:
#             self._run_command(cmd, "Feature extraction failed")
#         except RuntimeError as e:
#             if use_gpu and 'use_gpu' in str(e):
#                 logger.warning("GPU flag not supported by this COLMAP version, retrying with CPU")
#                 # Retry without GPU flag
#                 cmd = [
#                     self.colmap_exe, 'feature_extractor',
#                     '--database_path', str(database_path),
#                     '--image_path', image_dir,
#                     '--ImageReader.camera_model', camera_model,
#                     '--ImageReader.single_camera', '1',
#                     '--SiftExtraction.max_image_size', str(settings['max_image_size']),
#                     '--SiftExtraction.max_num_features', str(settings['max_num_features']),
#                 ]
#                 self._run_command(cmd, "Feature extraction failed")
#             else:
#                 raise
    
#     def _run_feature_matching(
#         self,
#         database_path: Path,
#         use_gpu: bool,
#         quality: str
#     ):
#         """Match features between images"""
#         # Use exhaustive matching for small datasets, sequential for large
#         cmd = [
#             self.colmap_exe, 'exhaustive_matcher',
#             '--database_path', str(database_path),
#         ]

#         if use_gpu:
#             cmd.extend(['--SiftMatching.use_gpu', '1'])

#         # For larger datasets, consider sequential or spatial matching
#         # cmd = [self.colmap_exe, 'sequential_matcher', ...]

#         # Try with GPU flag, fallback to CPU if GPU not supported
#         try:
#             self._run_command(cmd, "Feature matching failed")
#         except RuntimeError as e:
#             if use_gpu and 'use_gpu' in str(e):
#                 logger.warning("GPU flag not supported by this COLMAP version, retrying with CPU")
#                 # Retry without GPU flag
#                 cmd = [
#                     self.colmap_exe, 'exhaustive_matcher',
#                     '--database_path', str(database_path),
#                 ]
#                 self._run_command(cmd, "Feature matching failed")
#             else:
#                 raise
    
#     def _run_mapper(
#         self,
#         database_path: Path,
#         image_dir: str,
#         output_dir: Path
#     ):
#         """Run sparse reconstruction"""
#         cmd = [
#             self.colmap_exe, 'mapper',
#             '--database_path', str(database_path),
#             '--image_path', image_dir,
#             '--output_path', str(output_dir),
#             '--Mapper.ba_refine_focal_length', '1',
#             '--Mapper.ba_refine_extra_params', '1',
#         ]
        
#         self._run_command(cmd, "Sparse reconstruction failed")
    
#     def _run_command(self, cmd: list, error_msg: str):
#         """Run COLMAP command"""
#         logger.debug(f"Running: {' '.join(cmd)}")
        
#         try:
#             result = subprocess.run(
#                 cmd,
#                 capture_output=True,
#                 text=True,
#                 timeout=3600  # 1 hour timeout
#             )
            
#             if result.returncode != 0:
#                 logger.error(f"STDOUT: {result.stdout}")
#                 logger.error(f"STDERR: {result.stderr}")
#                 raise RuntimeError(f"{error_msg}: {result.stderr}")
            
#             logger.debug(f"Command successful")
            
#         except subprocess.TimeoutExpired:
#             raise RuntimeError(f"{error_msg}: Command timeout")
#         except Exception as e:
#             raise RuntimeError(f"{error_msg}: {e}")
    
#     def convert_to_poses_json(
#         self,
#         sparse_model_dir: str,
#         output_json: str,
#         image_names: Optional[list] = None
#     ) -> Dict:
#         """
#         Convert COLMAP sparse model to poses.json format.
        
#         Args:
#             sparse_model_dir: Directory containing COLMAP sparse model
#             output_json: Output JSON file path
#             image_names: Optional list of image names to include
            
#         Returns:
#             Dictionary of poses
#         """
#         logger.info(f"Converting COLMAP model to poses.json: {sparse_model_dir}")
        
#         model_path = Path(sparse_model_dir)
        
#         # Read COLMAP files
#         cameras = self._read_cameras_txt(model_path / 'cameras.txt')
#         images = self._read_images_txt(model_path / 'images.txt')
        
#         # Convert to our format
#         poses = {}
        
#         for img_id, img_data in images.items():
#             img_name = img_data['name']
            
#             # Skip if not in requested list
#             if image_names is not None and img_name not in image_names:
#                 continue
            
#             # Get camera
#             cam_id = img_data['camera_id']
#             camera = cameras.get(cam_id)
            
#             if camera is None:
#                 logger.warning(f"Camera {cam_id} not found for image {img_name}")
#                 continue
            
#             # COLMAP uses quaternion (qw, qx, qy, qz) and translation
#             qvec = img_data['qvec']  # [qw, qx, qy, qz]
#             tvec = img_data['tvec']  # [tx, ty, tz]
            
#             # Convert quaternion to rotation matrix
#             R = self._qvec_to_rotmat(qvec)
            
#             # COLMAP convention: world-to-camera
#             # We need camera-to-world for our pipeline
#             # R_c2w = R_w2c^T, t_c2w = -R_c2w @ t_w2c
#             R_c2w = R.T
#             t_c2w = -R_c2w @ tvec
            
#             # Build intrinsic matrix
#             K = self._build_intrinsic_matrix(camera)
            
#             poses[img_name] = {
#                 'filename': img_name,
#                 'R': R_c2w.tolist(),
#                 't': t_c2w.reshape(3, 1).tolist(),
#                 'K': K.tolist()
#             }
        
#         logger.info(f"Converted {len(poses)} image poses")
        
#         # Save to JSON
#         output_path = Path(output_json)
#         output_path.parent.mkdir(parents=True, exist_ok=True)
        
#         with open(output_path, 'w') as f:
#             json.dump(poses, f, indent=2)
        
#         logger.info(f"Poses saved to: {output_json}")
        
#         return poses
    
#     def _read_cameras_txt(self, cameras_file: Path) -> Dict:
#         """
#         Read COLMAP cameras file (supports both .txt and .bin formats).

#         Args:
#             cameras_file: Path to cameras.txt (will also try cameras.bin)

#         Returns:
#             Dictionary of camera data
#         """
#         import struct

#         # Try binary format first
#         cameras_bin = cameras_file.parent / 'cameras.bin'
#         cameras_txt = cameras_file

#         if cameras_bin.exists():
#             logger.debug(f"Reading COLMAP cameras (binary): {cameras_bin}")
#             cameras = {}

#             # COLMAP camera model ID to name mapping
#             CAMERA_MODEL_NAMES = {
#                 0: 'SIMPLE_PINHOLE',
#                 1: 'PINHOLE',
#                 2: 'SIMPLE_RADIAL',
#                 3: 'RADIAL',
#                 4: 'OPENCV',
#                 5: 'OPENCV_FISHEYE',
#                 6: 'FULL_OPENCV',
#                 7: 'FOV',
#                 8: 'SIMPLE_RADIAL_FISHEYE',
#                 9: 'RADIAL_FISHEYE',
#                 10: 'THIN_PRISM_FISHEYE'
#             }

#             with open(cameras_bin, 'rb') as f:
#                 num_cameras = struct.unpack('Q', f.read(8))[0]

#                 for _ in range(num_cameras):
#                     cam_id = struct.unpack('I', f.read(4))[0]
#                     model_id = struct.unpack('i', f.read(4))[0]
#                     width = struct.unpack('Q', f.read(8))[0]
#                     height = struct.unpack('Q', f.read(8))[0]

#                     model_name = CAMERA_MODEL_NAMES.get(model_id, f'MODEL_{model_id}')

#                     # Number of params depends on model
#                     num_params_map = {
#                         'SIMPLE_PINHOLE': 3,
#                         'PINHOLE': 4,
#                         'SIMPLE_RADIAL': 4,
#                         'RADIAL': 5,
#                         'OPENCV': 8,
#                         'OPENCV_FISHEYE': 8,
#                         'FULL_OPENCV': 12,
#                         'FOV': 5,
#                         'SIMPLE_RADIAL_FISHEYE': 4,
#                         'RADIAL_FISHEYE': 5,
#                         'THIN_PRISM_FISHEYE': 12
#                     }

#                     num_params = num_params_map.get(model_name, 8)  # default to 8
#                     params = struct.unpack(f'{num_params}d', f.read(8 * num_params))

#                     cameras[cam_id] = {
#                         'model': model_name,
#                         'width': int(width),
#                         'height': int(height),
#                         'params': list(params)
#                     }

#             logger.debug(f"Read {len(cameras)} cameras from binary")
#             return cameras

#         # Fallback to text format
#         elif cameras_txt.exists():
#             logger.debug(f"Reading COLMAP cameras (text): {cameras_txt}")
#             cameras = {}

#             with open(cameras_txt, 'r') as f:
#                 for line in f:
#                     line = line.strip()
#                     if not line or line.startswith('#'):
#                         continue

#                     parts = line.split()
#                     cam_id = int(parts[0])
#                     model = parts[1]
#                     width = int(parts[2])
#                     height = int(parts[3])
#                     params = [float(p) for p in parts[4:]]

#                     cameras[cam_id] = {
#                         'model': model,
#                         'width': width,
#                         'height': height,
#                         'params': params
#                     }

#             logger.debug(f"Read {len(cameras)} cameras from text")
#             return cameras

#         else:
#             raise FileNotFoundError(
#                 f"COLMAP cameras file not found. Tried: {cameras_bin}, {cameras_txt}"
#             )
    
#     def _read_images_txt(self, images_file: Path) -> Dict:
#         """
#         Read COLMAP images file (supports both .txt and .bin formats).

#         Args:
#             images_file: Path to images.txt (will also try images.bin)

#         Returns:
#             Dictionary of image data
#         """
#         import struct

#         # Try binary format first
#         images_bin = images_file.parent / 'images.bin'
#         images_txt = images_file

#         if images_bin.exists():
#             logger.debug(f"Reading COLMAP images (binary): {images_bin}")
#             images = {}

#             with open(images_bin, 'rb') as f:
#                 num_images = struct.unpack('Q', f.read(8))[0]

#                 for _ in range(num_images):
#                     img_id = struct.unpack('I', f.read(4))[0]

#                     # Quaternion (qw, qx, qy, qz)
#                     qvec = struct.unpack('dddd', f.read(32))

#                     # Translation (tx, ty, tz)
#                     tvec = struct.unpack('ddd', f.read(24))

#                     camera_id = struct.unpack('I', f.read(4))[0]

#                     # Image name (null-terminated string)
#                     name_chars = []
#                     while True:
#                         char = f.read(1)
#                         if char == b'\x00':
#                             break
#                         name_chars.append(char)
#                     name = b''.join(name_chars).decode('utf-8')

#                     # Skip points2D data
#                     num_points2D = struct.unpack('Q', f.read(8))[0]
#                     # Each point2D: x, y (2 doubles) + point3D_id (uint64) = 24 bytes
#                     f.read(num_points2D * 24)

#                     images[img_id] = {
#                         'name': name,
#                         'camera_id': camera_id,
#                         'qvec': np.array(qvec),
#                         'tvec': np.array(tvec)
#                     }

#             logger.debug(f"Read {len(images)} images from binary")
#             return images

#         # Fallback to text format
#         elif images_txt.exists():
#             logger.debug(f"Reading COLMAP images (text): {images_txt}")
#             images = {}

#             with open(images_txt, 'r') as f:
#                 lines = f.readlines()

#             i = 0
#             while i < len(lines):
#                 line = lines[i].strip()
#                 i += 1

#                 if not line or line.startswith('#'):
#                     continue

#                 # Image line
#                 parts = line.split()
#                 img_id = int(parts[0])
#                 qw, qx, qy, qz = map(float, parts[1:5])
#                 tx, ty, tz = map(float, parts[5:8])
#                 camera_id = int(parts[8])
#                 name = parts[9]

#                 # Skip points2D line
#                 if i < len(lines):
#                     i += 1

#                 images[img_id] = {
#                     'name': name,
#                     'camera_id': camera_id,
#                     'qvec': np.array([qw, qx, qy, qz]),
#                     'tvec': np.array([tx, ty, tz])
#                 }

#             logger.debug(f"Read {len(images)} images from text")
#             return images

#         else:
#             raise FileNotFoundError(
#                 f"COLMAP images file not found. Tried: {images_bin}, {images_txt}"
#             )
    
#     def _qvec_to_rotmat(self, qvec: np.ndarray) -> np.ndarray:
#         """Convert quaternion to rotation matrix"""
#         qw, qx, qy, qz = qvec
        
#         R = np.array([
#             [1 - 2*qy**2 - 2*qz**2, 2*qx*qy - 2*qz*qw, 2*qx*qz + 2*qy*qw],
#             [2*qx*qy + 2*qz*qw, 1 - 2*qx**2 - 2*qz**2, 2*qy*qz - 2*qx*qw],
#             [2*qx*qz - 2*qy*qw, 2*qy*qz + 2*qx*qw, 1 - 2*qx**2 - 2*qy**2]
#         ])
        
#         return R
    
#     def _build_intrinsic_matrix(self, camera: Dict) -> np.ndarray:
#         """Build intrinsic matrix from COLMAP camera parameters"""
#         model = camera['model']
#         params = camera['params']
#         width = camera['width']
#         height = camera['height']
        
#         if model == 'PINHOLE':
#             # params: fx, fy, cx, cy
#             fx, fy, cx, cy = params
#         elif model == 'SIMPLE_PINHOLE':
#             # params: f, cx, cy
#             f, cx, cy = params
#             fx = fy = f
#         elif model == 'OPENCV' or model == 'RADIAL':
#             # params: fx, fy, cx, cy, k1, k2, p1, p2 [, k3, k4, k5, k6]
#             fx, fy, cx, cy = params[:4]
#         elif model == 'SIMPLE_RADIAL':
#             # params: f, cx, cy, k
#             f, cx, cy, k = params
#             fx = fy = f
#         else:
#             logger.warning(f"Unknown camera model: {model}, using default")
#             fx = fy = max(width, height)
#             cx, cy = width / 2, height / 2
        
#         K = np.array([
#             [fx, 0, cx],
#             [0, fy, cy],
#             [0, 0, 1]
#         ])
        
#         return K
    
#     def extract_dense_point_cloud(
#         self,
#         sparse_model_dir: str,
#         image_dir: str,
#         output_dir: str,
#         max_image_size: int = 2000,
#         dense_params: Optional[Dict] = None
#     ) -> str:
#         """
#         Run dense reconstruction (optional, for reference point cloud).

#         Args:
#             sparse_model_dir: Sparse model directory
#             image_dir: Image directory
#             output_dir: Output directory for dense reconstruction
#             max_image_size: Maximum image size for dense reconstruction
#             dense_params: Dense reconstruction parameters (geom_consistency, input_type, etc.)

#         Returns:
#             Path to dense point cloud
#         """
#         logger.info("Running dense reconstruction (this may take a while)...")

#         # Parse dense parameters with defaults
#         if dense_params is None:
#             dense_params = {}

#         geom_consistency = dense_params.get('geom_consistency', False)
#         input_type = dense_params.get('input_type', 'photometric')
#         max_img_size = dense_params.get('max_image_size', max_image_size)

#         # Stereo fusion parameters (optional)
#         min_num_pixels = dense_params.get('min_num_pixels', 3)
#         max_reproj_error = dense_params.get('max_reproj_error', 3.0)
#         max_depth_error = dense_params.get('max_depth_error', 0.01)
#         max_normal_error = dense_params.get('max_normal_error', 25.0)

#         output_path = Path(output_dir)
#         output_path.mkdir(parents=True, exist_ok=True)

#         dense_dir = output_path / 'dense'
#         dense_dir.mkdir(exist_ok=True)

#         logger.info(f"Dense params: input_type={input_type}, geom_consistency={geom_consistency}")

#         # Step 1: Undistort images
#         logger.info("Step 1/3: Image undistortion")
#         cmd = [
#             self.colmap_exe, 'image_undistorter',
#             '--image_path', image_dir,
#             '--input_path', sparse_model_dir,
#             '--output_path', str(dense_dir),
#             '--output_type', 'COLMAP',
#             '--max_image_size', str(max_img_size)
#         ]
#         self._run_command(cmd, "Image undistortion failed")

#         # Step 2: Patch match stereo
#         logger.info("Step 2/3: Patch match stereo")
#         cmd = [
#             self.colmap_exe, 'patch_match_stereo',
#             '--workspace_path', str(dense_dir),
#             '--workspace_format', 'COLMAP',
#             '--PatchMatchStereo.geom_consistency', str(geom_consistency).lower()
#         ]
#         self._run_command(cmd, "Patch match stereo failed")

#         # Step 3: Stereo fusion
#         logger.info("Step 3/3: Stereo fusion")

#         # Output filename based on input_type
#         output_filename = f'fused_{input_type}.ply' if input_type != 'geometric' else 'fused.ply'

#         cmd = [
#             self.colmap_exe, 'stereo_fusion',
#             '--workspace_path', str(dense_dir),
#             '--workspace_format', 'COLMAP',
#             '--input_type', input_type,
#             '--output_path', str(dense_dir / output_filename),
#             '--StereoFusion.min_num_pixels', str(min_num_pixels),
#             '--StereoFusion.max_reproj_error', str(max_reproj_error),
#             '--StereoFusion.max_depth_error', str(max_depth_error),
#             '--StereoFusion.max_normal_error', str(max_normal_error)
#         ]
#         self._run_command(cmd, "Stereo fusion failed")

#         output_ply = dense_dir / output_filename
#         logger.info(f"Dense point cloud saved: {output_ply}")

#         return str(output_ply)


# def run_colmap_sfm_auto(
#     image_dir: str,
#     output_dir: str,
#     poses_json_output: str,
#     camera_model: str = 'OPENCV',
#     quality: str = 'high',
#     dense: bool = False,
#     dense_params: Optional[Dict] = None,
#     colmap_exe: str = 'colmap',
#     use_cuda: str = 'auto'
# ) -> Dict:
#     """
#     Automatic COLMAP SFM pipeline.

#     Args:
#         image_dir: Directory with RGB images
#         output_dir: Output directory for COLMAP
#         poses_json_output: Output path for poses.json
#         camera_model: COLMAP camera model
#         quality: Quality setting ('low', 'medium', 'high', 'extreme')
#         dense: Whether to run dense reconstruction
#         dense_params: Dense reconstruction parameters (geom_consistency, input_type, etc.)
#         colmap_exe: Path to COLMAP executable (default: 'colmap')
#         use_cuda: 'auto' (detect), 'true', 'false', or boolean (default: 'auto')

#     Returns:
#         Dictionary of poses
#     """
#     runner = COLMAPRunner(colmap_executable=colmap_exe)

#     # Run sparse reconstruction
#     sparse_model_dir = runner.run_sfm_pipeline(
#         image_dir=image_dir,
#         output_dir=output_dir,
#         camera_model=camera_model,
#         use_cuda=use_cuda,
#         quality=quality
#     )

#     # Convert to poses.json
#     poses = runner.convert_to_poses_json(
#         sparse_model_dir=sparse_model_dir,
#         output_json=poses_json_output
#     )

#     # Optional: Dense reconstruction
#     if dense:
#         runner.extract_dense_point_cloud(
#             sparse_model_dir=sparse_model_dir,
#             image_dir=image_dir,
#             output_dir=output_dir,
#             dense_params=dense_params
#         )

#     return poses


# if __name__ == '__main__':
#     # Test
#     import argparse
#     import sys
    
#     parser = argparse.ArgumentParser(description='Run COLMAP SFM')
#     parser.add_argument('--image-dir', required=True, help='Directory with images')
#     parser.add_argument('--output-dir', required=True, help='Output directory')
#     parser.add_argument('--poses-json', default='poses.json', help='Output poses.json')
#     parser.add_argument('--camera-model', default='OPENCV', help='Camera model')
#     parser.add_argument('--quality', default='high', choices=['low', 'medium', 'high', 'extreme'])
#     parser.add_argument('--dense', action='store_true', help='Run dense reconstruction')
    
#     args = parser.parse_args()
    
#     logging.basicConfig(level=logging.INFO)
    
#     try:
#         poses = run_colmap_sfm_auto(
#             image_dir=args.image_dir,
#             output_dir=args.output_dir,
#             poses_json_output=args.poses_json,
#             camera_model=args.camera_model,
#             quality=args.quality,
#             dense=args.dense
#         )
        
#         print(f"\n✅ SFM complete!")
#         print(f"   Poses: {args.poses_json}")
#         print(f"   Reconstructed {len(poses)} images")
        
#     except Exception as e:
#         print(f"\n❌ SFM failed: {e}")
#         sys.exit(1)




# -*- coding: utf-8 -*-
"""
colmap_sfm.py
Safe COLMAP wrapper with:
- Robust detection of COLMAP binary and version (uses `colmap help`).
- Auto‑detection of SIFT GPU option name (`--SiftExtraction.use_gpu` vs `--SiftExtraction.use_cuda`).
- Auto‑detection of Matching GPU option name (`--SiftMatching.use_*`).
- Removal of stale read‑only COLMAP database before feature extraction.
- Dense reconstruction prechecks (CUDA linkage + GPU visibility).
- Geometric consistency retry logic for PatchMatchStereo.
"""

import subprocess
import json
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, List
import logging
import shutil
import os

logger = logging.getLogger(__name__)


class COLMAPRunner:
    """Wrapper for COLMAP SFM + MVS pipeline"""

    def __init__(self, colmap_executable: str = 'colmap'):
        self.colmap_exe = colmap_executable
        self._check_colmap_installed()

    # ----------------------------
    # Environment / capability checks
    # ----------------------------
    def _check_colmap_installed(self) -> None:
        """Check COLMAP binary exists and is runnable; log banner line."""
        exe = shutil.which(self.colmap_exe) or self.colmap_exe
        try:
            r = subprocess.run([exe, 'help'], capture_output=True, text=True, timeout=8)
            if r.returncode != 0:
                logger.warning("COLMAP at %s but `help` returned non-zero: %s", exe, (r.stderr or '').strip())
            else:
                head = (r.stdout or '').splitlines()[0:1]
                logger.info("COLMAP found at %s | %s", exe, (head[0] if head else '').strip())
        except FileNotFoundError:
            raise RuntimeError(f"COLMAP not found at: {exe}")
        except Exception as e:
            logger.warning("Could not verify COLMAP executable: %s", e)

    def _supports_option(self, subcmd: str, opt_name: str) -> bool:
        """Return True if `colmap <subcmd> --help` lists `--<opt_name>`."""
        exe = shutil.which(self.colmap_exe) or self.colmap_exe
        try:
            r = subprocess.run([exe, subcmd, '--help'], capture_output=True, text=True, timeout=8)
            txt = (r.stdout or '') + (r.stderr or '')
            return f'--{opt_name}' in txt
        except Exception:
            return False

    def _colmap_has_cuda(self) -> bool:
        exe = shutil.which(self.colmap_exe) or self.colmap_exe
        try:
            r = subprocess.run(['ldd', exe], capture_output=True, text=True)
            return 'libcudart' in (r.stdout or '')
        except Exception:
            return False

    def _gpu_visible(self) -> bool:
        try:
            r = subprocess.run(['nvidia-smi', '-L'], capture_output=True, text=True, timeout=5)
            return r.returncode == 0 and (r.stdout or '').strip() != ''
        except FileNotFoundError:
            return False

    # ----------------------------
    # SFM stages
    # ----------------------------
    def run_sfm_pipeline(
        self,
        image_dir: str,
        output_dir: str,
        camera_model: str = 'OPENCV',
        use_gpu: bool = True,
        quality: str = 'high',
        matcher: str = 'exhaustive',  # 'exhaustive' | 'sequential' | 'spatial'
    ) -> str:
        out_dir = Path(output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        database_path = out_dir / 'database.db'
        # Clear stale/readonly DB left by previous runs (often appears if a prior run died under sudo)
        if database_path.exists():
            try:
                database_path.unlink()
                logger.info("Removed stale database %s", database_path)
            except Exception as e:
                raise RuntimeError(f"Cannot remove stale {database_path}: {e}")

        sparse_dir = out_dir / 'sparse'
        sparse_dir.mkdir(exist_ok=True)

        logger.info("=" * 80)
        logger.info("Starting COLMAP SFM Pipeline")
        logger.info("=" * 80)

        # 1) Feature Extraction
        logger.info("Step 1/4: Feature Extraction")
        self._run_feature_extraction(database_path, image_dir, camera_model, use_gpu, quality)

        # 2) Feature Matching
        logger.info("Step 2/4: Feature Matching")
        self._run_feature_matching(database_path, use_gpu, matcher)

        # 3) Sparse Reconstruction (Mapper)
        logger.info("Step 3/4: Sparse Reconstruction")
        self._run_mapper(database_path, image_dir, sparse_dir)

        # 4) Convert to standard format (poses)
        logger.info("Step 4/4: Converting to standard format")
        model_dir = sparse_dir / '0'
        if not model_dir.exists():
            raise RuntimeError(f"Reconstruction failed: {model_dir} not found")
        logger.info("SFM reconstruction complete: %s", model_dir)

        return str(model_dir)

    def _run_feature_extraction(
        self,
        database_path: Path,
        image_dir: str,
        camera_model: str,
        use_gpu: bool,
        quality: str
    ) -> None:
        quality_settings = {
            'low':     {'max_image_size': 1600, 'max_num_features':  4096},
            'medium':  {'max_image_size': 2400, 'max_num_features':  8192},
            'high':    {'max_image_size': 3200, 'max_num_features': 16384},
            'extreme': {'max_image_size': 4800, 'max_num_features': 32768},
        }
        s = quality_settings.get(quality, quality_settings['high'])

        cmd = [
            self.colmap_exe, 'feature_extractor',
            '--database_path', str(database_path),
            '--image_path', image_dir,
            '--ImageReader.camera_model', camera_model,
            '--ImageReader.single_camera', '1',
            '--SiftExtraction.max_image_size', str(s['max_image_size']),
            '--SiftExtraction.max_num_features', str(s['max_num_features']),
        ]

        if use_gpu:
            # COLMAP 3.8+ uses SiftExtraction.use_gpu; some dev builds use SiftExtraction.use_cuda; some auto-detect GPU
            if self._supports_option('feature_extractor', 'SiftExtraction.use_gpu'):
                cmd += ['--SiftExtraction.use_gpu', '1']
            elif self._supports_option('feature_extractor', 'SiftExtraction.use_cuda'):
                cmd += ['--SiftExtraction.use_cuda', '1']
            else:
                logger.warning("SIFT GPU flag not supported in this COLMAP build; proceeding without explicit GPU flag.")

        self._run_command(cmd, "Feature extraction failed")

    def _run_feature_matching(self, database_path: Path, use_gpu: bool, matcher: str) -> None:
        if matcher == 'sequential':
            cmd = [self.colmap_exe, 'sequential_matcher', '--database_path', str(database_path)]
        elif matcher == 'spatial':
            cmd = [self.colmap_exe, 'spatial_matcher', '--database_path', str(database_path)]
        else:
            cmd = [self.colmap_exe, 'exhaustive_matcher', '--database_path', str(database_path)]

        if use_gpu:
            # COLMAP 3.8+ uses SiftMatching.use_gpu; some dev builds use SiftMatching.use_cuda
            if self._supports_option('exhaustive_matcher', 'SiftMatching.use_gpu'):
                cmd += ['--SiftMatching.use_gpu', '1']
            elif self._supports_option('exhaustive_matcher', 'SiftMatching.use_cuda'):
                cmd += ['--SiftMatching.use_cuda', '1']
            else:
                logger.warning("SIFT Matching GPU flag not supported in this COLMAP build; proceeding without explicit GPU flag.")

        self._run_command(cmd, "Feature matching failed")

    def _run_mapper(self, database_path: Path, image_dir: str, output_dir: Path) -> None:
        cmd = [
            self.colmap_exe, 'mapper',
            '--database_path', str(database_path),
            '--image_path', image_dir,
            '--output_path', str(output_dir),
            '--Mapper.ba_refine_focal_length', '1',
            '--Mapper.ba_refine_extra_params', '1',
        ]
        self._run_command(cmd, "Sparse reconstruction failed")

    # ----------------------------
    # Dense MVS
    # ----------------------------
    def extract_dense_point_cloud(
        self,
        sparse_model_dir: str,
        image_dir: str,
        output_dir: str,
        max_image_size: int = 2000,
        geom_consistency: bool = False,
        gpu_index: Optional[int] = None,
    ) -> str:
        """Run dense reconstruction (PatchMatch + Fusion)."""

        # Precheck: CUDA linkage & GPU visibility
        if not self._colmap_has_cuda() or not self._gpu_visible():
            raise RuntimeError(
                "Dense reconstruction requires CUDA-enabled COLMAP and a visible NVIDIA GPU.\n"
                f"- COLMAP CUDA link: {'OK' if self._colmap_has_cuda() else 'MISSING libcudart'}\n"
                f"- GPU visible: {'OK' if self._gpu_visible() else 'NO (check nvidia-smi / CUDA_VISIBLE_DEVICES)'}"
            )

        logger.info("Running dense reconstruction (this may take a while)...")
        out_dir = Path(output_dir); out_dir.mkdir(parents=True, exist_ok=True)
        dense_dir = (out_dir / 'dense'); dense_dir.mkdir(exist_ok=True)

        # 1) Undistort
        logger.info("Step 1/3: Image undistortion")
        self._run_command([
            self.colmap_exe, 'image_undistorter',
            '--image_path', image_dir,
            '--input_path', str(sparse_model_dir),
            '--output_path', str(dense_dir),
            '--output_type', 'COLMAP',
            '--max_image_size', str(max_image_size),
        ], "Image undistortion failed")

        # 2) PatchMatch Stereo
        logger.info("Step 2/3: Patch match stereo")
        pm = [
            self.colmap_exe, 'patch_match_stereo',
            '--workspace_path', str(dense_dir),
            '--workspace_format', 'COLMAP',
            '--PatchMatchStereo.geom_consistency', '1' if (geom_consistency) else '0',
            '--PatchMatchStereo.max_image_size', str(max_image_size),
        ]
        if gpu_index is not None:
            pm += ['--PatchMatchStereo.gpu_index', str(gpu_index)]
        try:
            self._run_command(pm, "Patch match stereo failed")
        except RuntimeError as e:
            if geom_consistency:
                logger.warning("Geom consistency failed; retrying with geom_consistency=0. Error: %s", e)
                # flip the flag in-place and retry once without geom consistency
                gi = pm.index('--PatchMatchStereo.geom_consistency') + 1
                pm[gi] = '0'
                self._run_command(pm, "Patch match stereo failed (retry without geom consistency)")
            else:
                raise

        # 3) Fusion
        logger.info("Step 3/3: Stereo fusion")
        self._run_command([
            self.colmap_exe, 'stereo_fusion',
            '--workspace_path', str(dense_dir),
            '--workspace_format', 'COLMAP',
            '--input_type', 'geometric',
            '--output_path', str(dense_dir / 'fused.ply'),
        ], "Stereo fusion failed")

        ply = dense_dir / 'fused.ply'
        logger.info("Dense point cloud saved: %s", ply)
        return str(ply)

    # ----------------------------
    # Utilities
    # ----------------------------
    def _run_command(self, cmd: List[str], error_msg: str) -> None:
        logger.debug("Running: %s", " ".join(cmd))
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=360000)
            if result.returncode != 0:
                logger.error("STDOUT: %s", result.stdout)
                logger.error("STDERR: %s", result.stderr)
                raise RuntimeError(f"{error_msg}: {result.stderr}")
        except subprocess.TimeoutExpired:
            raise RuntimeError(f"{error_msg}: Command timeout")
        except Exception as e:
            raise RuntimeError(f"{error_msg}: {e}")

    # ----------------------------
    # Model conversion helpers
    # ----------------------------
    def convert_to_poses_json(
        self,
        sparse_model_dir: str,
        output_json: str,
        image_names: Optional[list] = None
    ) -> Dict:
        """
        Convert COLMAP sparse model to poses.json format.
        """
        logger.info("Converting COLMAP model to poses.json: %s", sparse_model_dir)
        model_path = Path(sparse_model_dir)

        cameras = self._read_cameras_txt(model_path / 'cameras.txt')
        images  = self._read_images_txt (model_path / 'images.txt')

        poses: Dict[str, Dict] = {}

        for img_id, img_data in images.items():
            name = img_data['name']
            if image_names is not None and name not in image_names:
                continue

            cam_id = img_data['camera_id']
            camera = cameras.get(cam_id)
            if camera is None:
                logger.warning("Camera %s not found for image %s", cam_id, name)
                continue

            qvec = img_data['qvec']  # [qw, qx, qy, qz]
            tvec = img_data['tvec']  # [tx, ty, tz]

            R_w2c = self._qvec_to_rotmat(qvec)
            R_c2w = R_w2c.T
            t_c2w = -R_c2w @ tvec

            K = self._build_intrinsic_matrix(camera)

            poses[name] = {
                'filename': name,
                'R': R_c2w.tolist(),
                't': t_c2w.reshape(3, 1).tolist(),
                'K': K.tolist(),
            }

        logger.info("Converted %d image poses", len(poses))

        out = Path(output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, 'w') as f:
            json.dump(poses, f, indent=2)

        logger.info("Poses saved to: %s", output_json)
        return poses

    def _read_cameras_txt(self, cameras_file: Path) -> Dict:
        """Read COLMAP cameras file (.txt or .bin)."""
        import struct
        cameras_bin = cameras_file.parent / 'cameras.bin'
        if cameras_bin.exists():
            logger.debug("Reading COLMAP cameras (binary): %s", cameras_bin)
            cameras: Dict[int, Dict] = {}
            CAMERA_MODEL_NAMES = {
                0: 'SIMPLE_PINHOLE',
                1: 'PINHOLE',
                2: 'SIMPLE_RADIAL',
                3: 'RADIAL',
                4: 'OPENCV',
                5: 'OPENCV_FISHEYE',
                6: 'FULL_OPENCV',
                7: 'FOV',
                8: 'SIMPLE_RADIAL_FISHEYE',
                9: 'RADIAL_FISHEYE',
                10: 'THIN_PRISM_FISHEYE',
            }
            with open(cameras_bin, 'rb') as f:
                n = struct.unpack('Q', f.read(8))[0]
                for _ in range(n):
                    cam_id = struct.unpack('I', f.read(4))[0]
                    model_id = struct.unpack('i', f.read(4))[0]
                    width  = struct.unpack('Q', f.read(8))[0]
                    height = struct.unpack('Q', f.read(8))[0]
                    model  = CAMERA_MODEL_NAMES.get(model_id, f"MODEL_{model_id}")
                    npar = {
                        'SIMPLE_PINHOLE': 3, 'PINHOLE': 4, 'SIMPLE_RADIAL': 4, 'RADIAL': 5,
                        'OPENCV': 8, 'OPENCV_FISHEYE': 8, 'FULL_OPENCV': 12, 'FOV': 5,
                        'SIMPLE_RADIAL_FISHEYE': 4, 'RADIAL_FISHEYE': 5, 'THIN_PRISM_FISHEYE': 12
                    }.get(model, 8)
                    params = struct.unpack(f'{npar}d', f.read(8 * npar))
                    cameras[int(cam_id)] = {'model': model, 'width': int(width), 'height': int(height), 'params': list(params)}
            return cameras

        # Fallback to text
        if not cameras_file.exists():
            raise FileNotFoundError(f"COLMAP cameras file not found: {cameras_file} (and {cameras_bin})")
        logger.debug("Reading COLMAP cameras (text): %s", cameras_file)
        cameras: Dict[int, Dict] = {}
        with open(cameras_file, 'r') as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                parts = line.split()
                cam_id  = int(parts[0]); model = parts[1]
                width   = int(parts[2]); height = int(parts[3])
                params  = [float(p) for p in parts[4:]]
                cameras[cam_id] = {'model': model, 'width': width, 'height': height, 'params': params}
        return cameras

    def _read_images_txt(self, images_file: Path) -> Dict:
        """Read COLMAP images file (.txt or .bin)."""
        import struct
        images_bin = images_file.parent / 'images.bin'
        if images_bin.exists():
            logger.debug("Reading COLMAP images (binary): %s", images_bin)
            images: Dict[int, Dict] = {}
            with open(images_bin, 'rb') as f:
                n = struct.unpack('Q', f.read(8))[0]
                for _ in range(n):
                    img_id = struct.unpack('I', f.read(4))[0]
                    qvec = np.array(struct.unpack('dddd', f.read(32)))
                    tvec = np.array(struct.unpack('ddd',  f.read(24)))
                    cam_id = struct.unpack('I', f.read(4))[0]
                    # name (null-terminated)
                    chars = []
                    while True:
                        ch = f.read(1)
                        if ch == b'\x00': break
                        chars.append(ch)
                    name = b''.join(chars).decode('utf-8')
                    # skip points2D
                    n2 = struct.unpack('Q', f.read(8))[0]
                    f.seek(n2 * 24, 1)
                    images[int(img_id)] = {'name': name, 'camera_id': int(cam_id), 'qvec': qvec, 'tvec': tvec}
            return images

        if not images_file.exists():
            raise FileNotFoundError(f"COLMAP images file not found: {images_file} (and {images_bin})")
        logger.debug("Reading COLMAP images (text): %s", images_file)
        images: Dict[int, Dict] = {}
        with open(images_file, 'r') as f:
            lines = f.readlines()
        i = 0
        while i < len(lines):
            line = lines[i].strip(); i += 1
            if not line or line.startswith('#'):
                continue
            parts = line.split()
            img_id = int(parts[0])
            qw, qx, qy, qz = map(float, parts[1:5])
            tx, ty, tz     = map(float, parts[5:8])
            cam_id         = int(parts[8])
            name           = parts[9]
            # skip points2D line
            if i < len(lines): i += 1
            images[int(img_id)] = {
                'name': name,
                'camera_id': cam_id,
                'qvec': np.array([qw, qx, qy, qz]),
                'tvec': np.array([tx, ty, tz]),
            }
        return images

    @staticmethod
    def _qvec_to_rotmat(qvec: np.ndarray) -> np.ndarray:
        """Convert quaternion [qw, qx, qy, qz] to 3x3 rotation matrix."""
        qw, qx, qy, qz = qvec
        R = np.array([
            [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
            [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
            [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
        ])
        return R

    @staticmethod
    def _build_intrinsic_matrix(camera: Dict) -> np.ndarray:
        """Build 3×3 K from COLMAP camera record."""
        model  = camera['model']
        params = camera['params']
        width  = camera['width']
        height = camera['height']

        if model == 'PINHOLE':
            fx, fy, cx, cy = params[:4]
        elif model == 'SIMPLE_PINHOLE':
            f, cx, cy = params[:3]
            fx = fy = f
        elif model in ('OPENCV', 'RADIAL', 'OPENCV_FISHEYE', 'FULL_OPENCV'):
            fx, fy, cx, cy = params[:4]
        elif model in ('SIMPLE_RADIAL', 'SIMPLE_RADIAL_FISHEYE'):
            f, cx, cy = params[:3]
            fx = fy = f
        else:
            logger.warning("Unknown camera model %s, falling back to center/size defaults", model)
            fx = fy = float(max(width, height)); cx = width / 2.0; cy = height / 2.0

        K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
        return K


def run_colmap_sfm_auto(
    image_dir: str,
    output_dir: str,
    poses_json_output: str,
    camera_model: str = 'OPENCV',
    quality: str = 'high',
    dense: bool = False,
    colmap_exe: str = 'colmap',
    use_gpu: bool = True,
    matcher: str = 'exhaustive',
    dense_params: Optional[Dict] = None,
) -> Dict:
    """High-level orchestration of the COLMAP SFM → poses.json → (optional) Dense MVS."""

    runner = COLMAPRunner(colmap_executable=colmap_exe)

    # --- Sparse reconstruction
    sparse_model_dir = runner.run_sfm_pipeline(
        image_dir=image_dir,
        output_dir=output_dir,
        camera_model=camera_model,
        quality=quality,
        use_gpu=use_gpu,
        matcher=matcher,
    )

    # --- poses.json export
    poses = runner.convert_to_poses_json(
        sparse_model_dir=sparse_model_dir,
        output_json=poses_json_output,
    )

    # --- Dense reconstruction (optional)
    if dense:
        dense_params = dense_params or {}
        runner.extract_dense_point_cloud(
            sparse_model_dir=sparse_model_dir,
            image_dir=image_dir,
            output_dir=output_dir,
            max_image_size=int(dense_params.get('max_image_size', 2000)),
            geom_consistency=bool(dense_params.get('geom_consistency', False)),
            gpu_index=dense_params.get('gpu_index', None),
        )

    return poses


if __name__ == '__main__':
    # Minimal CLI for standalone testing
    import argparse
    import sys
    logging.basicConfig(level=logging.INFO)

    p = argparse.ArgumentParser()
    p.add_argument('--image-dir', required=True)
    p.add_argument('--output-dir', required=True)
    p.add_argument('--poses-json', default='poses.json')
    p.add_argument('--camera-model', default='OPENCV')
    p.add_argument('--quality', choices=['low', 'medium', 'high', 'extreme'], default='high')
    p.add_argument('--dense', action='store_true')
    p.add_argument('--colmap-exe', default='colmap')
    p.add_argument('--use-gpu', action='store_true')
    p.add_argument('--matcher', choices=['exhaustive', 'sequential', 'spatial'], default='exhaustive')
    p.add_argument('--max-image-size', type=int, default=2000)
    p.add_argument('--geom-consistency', action='store_true')
    p.add_argument('--gpu-index', type=int, default=None)

    a = p.parse_args()
    try:
        poses = run_colmap_sfm_auto(
            image_dir=a.image_dir,
            output_dir=a.output_dir,
            poses_json_output=a.poses_json,
            camera_model=a.camera_model,
            quality=a.quality,
            dense=a.dense,
            colmap_exe=a.colmap_exe,
            use_gpu=a.use_gpu,
            matcher=a.matcher,
            dense_params={'max_image_size': a.max_image_size, 'geom_consistency': a.geom_consistency, 'gpu_index': a.gpu_index},
        )
        print("✅ SFM complete. Poses:", a.poses_json, "Images:", len(poses))
    except Exception as e:
        print("❌ SFM failed:", e)
        sys.exit(1)