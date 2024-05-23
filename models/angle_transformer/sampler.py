import numpy as np
import torch
import pyrosetta
from pyrosetta import rosetta

# Initialize PyRosetta
pyrosetta.init()

class Sampler:
    def __init__(self):
        self.pose = Sampler.create_inital_rosetta() 

    
    
# Idealized bond lengths (in Angstroms)
def unquantize_phi_psi_omega(data, n_bins=64):
  phi_psi_omega = data["quantized_phi_psi_omega"] - n_bins
  phi_psi_omega = phi_psi_omega / n_bins
  return phi_psi_omega

def sin_cos_angle_to_degrees(data):
    """Convert sin/cos representation to degrees."""
    return torch.atan2(data[..., 0], data[..., 1]) * 180 / np.pi

# unquantized = unquantize_phi_psi_omega(data)
# sin_cos = sin_cos_angle_to_degrees(unquantized)
# sin_cos[:,:,0] = sin_cos[:,:,0].roll(dims=-1,shifts=-1)

