import numpy as np
import torch
import pyrosetta
from pyrosetta import rosetta

from D3Fold.rosetta.adapters import create_from_torsion, poses_to_dataset
from dataclasses import dataclass

# Initialize PyRosetta
pyrosetta.init()

@dataclass
class SeedStructure:
    sequence: str
    phis: np.array
    psis: np.array
    omegas: np.array

class Sampler:
    def __init__(self, seed_structure: SeedStructure):
        self.pose = create_from_torsion(
            seed_structure.sequence,
            seed_structure.phis,
            seed_structure.psis,
            seed_structure.omegas
        )

        self.dataset = poses_to_dataset(
            [self.pose],
            type_dict=None,
            pdb_path="./denovo_pdbs",
            processed_path="./processed_pdbs",
            output_names=["seed_structure"],
        )

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

