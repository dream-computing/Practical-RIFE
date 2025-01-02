import torch
from torch.nn import functional as F
from RIFE_HDv3 import Model

# Constants
MODEL_DIR = 'train_log'
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
MODEL_VERSION = 3.9
WIDTH = 448
HEIGHT = 256

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

    def inference(self, frames, exp=4):
        """
        Interpolate between frames
        frames: list of numpy arrays (RGB format)
        exp: number of times to exponentially increase frame count
        returns: list of numpy arrays (RGB format)
        """
        if len(frames) < 2:
            return frames

        # Convert frames to tensors
        tensor_frames = []
        for frame in frames:
            # Assuming frame is numpy array in RGB format
            tensor = torch.from_numpy(frame).permute(2, 0, 1).float().to(DEVICE) / 255.
            tensor = tensor.unsqueeze(0)
            tensor_frames.append(tensor)

        interpolated_frames = []
        for i in range(len(tensor_frames) - 1):
            img0 = tensor_frames[i]
            img1 = tensor_frames[i + 1]

            # Pad images
            n, c, h, w = img0.shape
            ph = ((h - 1) // 64 + 1) * 64
            pw = ((w - 1) // 64 + 1) * 64
            padding = (0, pw - w, 0, ph - h)
            img0 = F.pad(img0, padding)
            img1 = F.pad(img1, padding)

            # Generate intermediate frames
            if MODEL_VERSION >= 3.9:
                n = 2 ** exp
                if i == 0:
                    interpolated_frames.append((img0[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w])
                for j in range(n-1):
                    middle = self.model.inference(img0, img1, (j+1) * 1. / n)
                    frame = (middle[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w]
                    interpolated_frames.append(frame)
                if i == len(tensor_frames) - 2:
                    interpolated_frames.append((img1[0] * 255).byte().cpu().numpy().transpose(1, 2, 0)[:h, :w])

        return interpolated_frames