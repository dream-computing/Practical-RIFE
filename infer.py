import numpy as np
import torch
from torch.nn import functional as F

from rife.RIFE_HDv3 import Model

# Constants
MODEL_DIR = None
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class RIFE:
    def __init__(self):
        if torch.cuda.is_available():
            torch.backends.cudnn.enabled = True
            torch.backends.cudnn.benchmark = True
        torch.set_grad_enabled(False)
        self.model = None

    def setup(self):
        """Initialize the RIFE model"""
        self.model = Model()
        self.model.load_model(MODEL_DIR, -1)
        self.model.eval()
        self.model.device()

    def inference(self, frame1, frame2):
        """
        Interpolate between two frames
        frame1, frame2: numpy arrays (RGB format)
        returns: numpy array (RGB format)
        """
        # Convert frames to tensors
        img0 = torch.from_numpy(frame1).permute(2, 0, 1).float().to(DEVICE) / 255.
        img1 = torch.from_numpy(frame2).permute(2, 0, 1).float().to(DEVICE) / 255.
        
        img0 = img0.unsqueeze(0)
        img1 = img1.unsqueeze(0)

        # Pad images
        n, c, h, w = img0.shape
        ph = ((h - 1) // 64 + 1) * 64
        pw = ((w - 1) // 64 + 1) * 64
        padding = (0, pw - w, 0, ph - h)
        img0 = F.pad(img0, padding)
        img1 = F.pad(img1, padding)

        # Generate middle frame
        middle = self.model.inference(img0, img1, 0.5)
        
        # Convert back to numpy array
        frame = (middle[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w]
        return np.ascontiguousarray(frame)  # Ensure memory is contiguous