"""
Core metrics for KinshipForge validation.
Geometry, Identity, Image Quality, Performance.
"""
import numpy as np
import torch
import cv2
from typing import Dict, Any, Optional, Tuple
from pathlib import Path


# ============================================================================
# Geometry Metrics (using MediaPipe Face Mesh)
# ============================================================================

try:
    import mediapipe as mp
    MP_AVAILABLE = True
    mp_face_mesh = mp.solutions.face_mesh
except ImportError:
    MP_AVAILABLE = False


LANDMARK_INDICES = {
    # Key facial landmarks (MediaPipe 468-point model)
    "left_eye_outer": 33,
    "left_eye_inner": 133,
    "right_eye_inner": 362,
    "right_eye_outer": 263,
    "nose_tip": 1,
    "mouth_left": 61,
    "mouth_right": 291,
    "left_cheek": 234,
    "right_cheek": 454,
    "left_jaw": 172,
    "right_jaw": 397,
    "chin": 152,
    "forehead": 10,
    "left_temple": 127,
    "right_temple": 356,
    # For width/height
    "left_eye_center": 468,   # iris center (approximate)
    "right_eye_center": 473,
}


def get_landmarks(image: np.ndarray) -> Optional[np.ndarray]:
    """Extract 468 facial landmarks using MediaPipe."""
    if not MP_AVAILABLE:
        return None
    
    with mp_face_mesh.FaceMesh(
        static_image_mode=True,
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5
    ) as face_mesh:
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) if len(image.shape) == 3 else image
        results = face_mesh.process(rgb)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            h, w = image.shape[:2]
            points = np.array([(lm.x * w, lm.y * h) for lm in landmarks.landmark])
            return points
    return None


def compute_face_width(landmarks: np.ndarray) -> float:
    """Bizygomatic width: distance between left and right cheekbones."""
    left = landmarks[LANDMARK_INDICES["left_cheek"]]
    right = landmarks[LANDMARK_INDICES["right_cheek"]]
    return float(np.linalg.norm(right - left))


def compute_jaw_width(landmarks: np.ndarray) -> float:
    """Bigonial width: distance between left and right jaw angles."""
    left = landmarks[LANDMARK_INDICES["left_jaw"]]
    right = landmarks[LANDMARK_INDICES["right_jaw"]]
    return float(np.linalg.norm(right - left))


def compute_face_height(landmarks: np.ndarray) -> float:
    """Face height: forehead to chin."""
    top = landmarks[LANDMARK_INDICES["forehead"]]
    bottom = landmarks[LANDMARK_INDICES["chin"]]
    return float(np.linalg.norm(bottom - top))


def compute_interocular_distance(landmarks: np.ndarray) -> float:
    """Distance between eye centers (for normalization)."""
    left = landmarks[LANDMARK_INDICES["left_eye_center"]]
    right = landmarks[LANDMARK_INDICES["right_eye_center"]]
    return float(np.linalg.norm(right - left))


def compute_geometry_metrics(image: np.ndarray) -> Dict[str, float]:
    """Compute all geometry metrics for a face image."""
    landmarks = get_landmarks(image)
    
    if landmarks is None:
        return {
            "width": -1, "height": -1, "wh_ratio": -1,
            "jaw_width": -1, "cheek_width": -1,
            "interocular": -1
        }
    
    width = compute_face_width(landmarks)
    height = compute_face_height(landmarks)
    jaw = compute_jaw_width(landmarks)
    cheek = compute_face_width(landmarks)  # same as width
    interocular = compute_interocular_distance(landmarks)
    
    return {
        "width": float(width),
        "height": float(height),
        "wh_ratio": float(width / height) if height > 0 else -1,
        "jaw_width": float(jaw),
        "cheek_width": float(cheek),
        "interocular": float(interocular),
        # Normalized
        "norm_jaw_width": float(jaw / interocular) if interocular > 0 else -1,
        "norm_width": float(width / interocular) if interocular > 0 else -1,
    }


# ============================================================================
# Identity Metrics (ArcFace)
# ============================================================================

try:
    import insightface
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False


class ArcFaceEvaluator:
    """ArcFace identity similarity evaluator."""
    
    def __init__(self, device: torch.device = None):
        self.device = device or (torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
        self.model = None
        self._load_model()
    
    def _load_model(self):
        if not INSIGHTFACE_AVAILABLE:
            print("Warning: insightface not available")
            return
        try:
            self.model = insightface.app.FaceAnalysis(name="buffalo_l", providers=['CUDAExecutionProvider'])
            self.model.prepare(ctx_id=0 if torch.cuda.is_available() else -1)
        except Exception as e:
            print(f"Warning: Could not load ArcFace: {e}")
    
    def get_embedding(self, image: np.ndarray) -> Optional[np.ndarray]:
        """Get 512-dim ArcFace embedding."""
        if self.model is None:
            return None
        faces = self.model.get(image)
        if faces:
            return faces[0].embedding
        return None
    
    def cosine_similarity(self, emb1: np.ndarray, emb2: np.ndarray) -> float:
        """Cosine similarity between two embeddings."""
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))
    
    def identity_metrics(self, child: np.ndarray, 
                         father: np.ndarray, 
                         mother: np.ndarray) -> Dict[str, float]:
        """Compute identity similarity metrics."""
        child_emb = self.get_embedding(child)
        father_emb = self.get_embedding(father)
        mother_emb = self.get_embedding(mother)
        
        results = {}
        if child_emb is not None and father_emb is not None:
            results["arcface_father"] = self.cosine_similarity(child_emb, father_emb)
        if child_emb is not None and mother_emb is not None:
            results["arcface_mother"] = self.cosine_similarity(child_emb, mother_emb)
        if "arcface_father" in results and "arcface_mother" in results:
            results["arcface_combined"] = (results["arcface_father"] + results["arcface_mother"]) / 2
        return results


# ============================================================================
# Image Quality Metrics
# ============================================================================

def compute_ssim(img1: np.ndarray, img2: np.ndarray) -> float:
    """Structural Similarity Index."""
    from skimage.metrics import structural_similarity as ssim
    if len(img1.shape) == 3:
        return float(ssim(img1, img2, channel_axis=2, data_range=255))
    return float(ssim(img1, img2, data_range=255))


def compute_lpips(img1: np.ndarray, img2: np.ndarray, net: str = 'alex') -> float:
    """LPIPS perceptual distance."""
    try:
        import lpips
        loss_fn = lpips.LPIPS(net=net)
        t1 = torch.from_numpy(img1.transpose(2, 0, 1)).float().unsqueeze(0) / 127.5 - 1
        t2 = torch.from_numpy(img2.transpose(2, 0, 1)).float().unsqueeze(0) / 127.5 - 1
        with torch.no_grad():
            return float(loss_fn(t1, t2).item())
    except ImportError:
        return -1.0


def compute_psnr(img1: np.ndarray, img2: np.ndarray) -> float:
    """Peak Signal-to-Noise Ratio."""
    mse = np.mean((img1.astype(np.float32) - img2.astype(np.float32)) ** 2)
    if mse == 0:
        return float('inf')
    return float(20 * np.log10(255.0 / np.sqrt(mse)))


def compute_mae(img1: np.ndarray, img2: np.ndarray) -> float:
    """Mean Absolute Error."""
    return float(np.mean(np.abs(img1.astype(np.float32) - img2.astype(np.float32))))


def compute_image_quality(child: np.ndarray, 
                          target: Optional[np.ndarray] = None) -> Dict[str, float]:
    """Compute image quality metrics. Full-reference if target provided."""
    if target is None:
        return {"ssim": -1, "lpips_alex": -1, "lpips_vgg": -1, "psnr": -1, "mae": -1}
    
    return {
        "ssim": compute_ssim(child, target),
        "lpips_alex": compute_lpips(child, target, 'alex'),
        "lpips_vgg": compute_lpips(child, target, 'vgg'),
        "psnr": compute_psnr(child, target),
        "mae": compute_mae(child, target),
    }


# ============================================================================
# Performance Metrics
# ============================================================================

def measure_runtime(func, *args, **kwargs) -> Tuple[Any, float]:
    """Measure execution time in seconds."""
    torch.cuda.synchronize() if torch.cuda.is_available() else None
    start = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    end = torch.cuda.Event(enable_timing=True) if torch.cuda.is_available() else None
    
    if start:
        start.record()
    else:
        import time
        start_time = time.perf_counter()
    
    result = func(*args, **kwargs)
    
    if end:
        end.record()
        torch.cuda.synchronize()
        elapsed = start.elapsed_time(end) / 1000.0  # seconds
    else:
        elapsed = time.perf_counter() - start_time
    
    return result, elapsed


def get_gpu_memory() -> Dict[str, float]:
    """Get GPU memory usage in GB."""
    if not torch.cuda.is_available():
        return {"allocated": 0, "reserved": 0, "free": 0}
    return {
        "allocated": torch.cuda.memory_allocated() / 1e9,
        "reserved": torch.cuda.memory_reserved() / 1e9,
        "free": (torch.cuda.get_device_properties(0).total_memory - torch.cuda.memory_allocated()) / 1e9
    }


# ============================================================================
# Master Metrics Function
# ============================================================================

def compute_all_metrics(child: np.ndarray,
                        father: np.ndarray,
                        mother: np.ndarray,
                        real_child: Optional[np.ndarray] = None,
                        arcface: Optional['ArcFaceEvaluator'] = None) -> Dict[str, Any]:
    """
    Compute ALL metrics for a single child generation.
    Returns comprehensive dict.
    """
    results = {}
    
    # Geometry (no reference needed)
    results["geometry"] = compute_geometry_metrics(child)
    
    # Image quality (full-reference if real_child provided)
    results["image_quality"] = compute_image_quality(child, real_child)
    
    # Identity (if ArcFace available)
    if arcface:
        results["identity"] = arcface.identity_metrics(child, father, mother)
    
    # Kinship verification
    if arcface:
        child_emb = arcface.get_embedding(child)
        father_emb = arcface.get_embedding(father)
        mother_emb = arcface.get_embedding(mother)
        if all(e is not None for e in [child_emb, father_emb, mother_emb]):
            results["kinship"] = {
                "father": float(np.dot(child_emb, father_emb) / 
                               (np.linalg.norm(child_emb) * np.linalg.norm(father_emb))),
                "mother": float(np.dot(child_emb, mother_emb) / 
                               (np.linalg.norm(child_emb) * np.linalg.norm(mother_emb))),
            }
    
    return results


# ============================================================================
# Type hints
# ============================================================================

from typing import Optional, Tuple, Any
import time