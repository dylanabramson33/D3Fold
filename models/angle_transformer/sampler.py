import numpy as np
import torch
from D3Fold.rosetta.adapters import create_from_torsion, poses_to_dataset, append_torsion

# from D3Fold.rosetta.adapters import create_from_torsion, poses_to_dataset
from D3Fold.data.torch_data import Collator
from torch.utils.data import DataLoader
from dataclasses import dataclass
from D3Fold.data.openfold.rigid_utils import Rotation, Rigid

# Initialize PyRosetta

@dataclass
class SeedStructure:
    sequence: str
    phis: np.array
    psis: np.array
    omegas: np.array
# Initialize PyRosetta

@dataclass
class Sampler:
    def __init__(
        self,
        seed_structure: SeedStructure,
        type_dict: dict,
        model: torch.nn.Module,
        filtern_fns_w_fields,
        num_samples: int = 1,
        device: str = "cuda",
        sampling_fn: callable = None,
    ):
        self.pose = create_from_torsion(
            seed_structure.sequence,
            seed_structure.phis,
            seed_structure.psis,
            seed_structure.omegas
        )

        self.dataset = poses_to_dataset(
            [self.pose],
            type_dict=type_dict,
            pdb_path="./denovo_pdbs",
            processed_path="./processed_pdbs",
            output_names=["seed_structure"],
            filtern_fns_w_fields=filtern_fns_w_fields
        )
        self.device = device

        collator = Collator(type_dict)
        self.data_loader = DataLoader(
            self.dataset,
            batch_size=1,
            collate_fn=collator,
            pin_memory=True,
            num_workers=4,
            shuffle=False
        )
        self.data = next(iter(self.data_loader))
        self.model = model.to(device)
        self.sampling_fn = sampling_fn

    def unquantize_phi_psi_omega(self, new_phi, new_psi, new_omega, n_bins=64):
        new_phi = (new_phi - n_bins) / n_bins * 180.0
        new_psi = (new_psi - n_bins) / n_bins * 180.0
        new_omega = (new_omega - n_bins) / n_bins * 180.0
        return new_phi, new_psi, new_omega

    def get_new_frames(self):
        newest_ind = len(self.pose.residues)
        ca_ind = self.pose.residue(newest_ind).atom_index("CA")
        (x_CA,y_CA,z_CA) = self.pose.residue(newest_ind).atom(ca_ind).xyz()
        c_ind = self.pose.residue(newest_ind).atom_index("C")
        (x_C,y_C,z_C) = self.pose.residue(newest_ind).atom(c_ind).xyz()
        n_ind = self.pose.residue(newest_ind).atom_index("N")
        (x_N,y_N,z_N) = self.pose.residue(newest_ind).atom(n_ind).xyz()

        rigid = Rigid.from_3_points(
            p_neg_x_axis=torch.tensor([x_N,y_N,z_N]),
            origin=torch.tensor([x_CA,y_CA,z_CA]),
            p_xy_plane=torch.tensor([x_C,y_C,z_C]),
            eps=1e-8
        )

        rots = torch.eye(3)
        rots[0, 0] = -1
        rots[0, 2] = -1
        # not sure if first frame is rotated
        rots = Rotation(rot_mats=rots)
        rotated = rigid.compose(Rigid(rots, None))

        return rotated.to_tensor_4x4().unsqueeze(0).unsqueeze(0)

    def concat_new_prediction(self, new_frame):
        self.data["backbone_rigid_tensor"] = torch.cat([self.data["backbone_rigid_tensor"], new_frame.to(self.device)], dim=1)

    def sample_step(self):
        phi_pred,psi_pred,omega_pred,aa_pred = self.model(self.data.to(self.device))
        new_phi = phi_pred[:,-1]
        new_psi = psi_pred[:,-1]
        new_omega = omega_pred[:,-1]
        new_aa = aa_pred[:,-1]
        new_phi, new_psi, new_omega, new_aa = self.sampling_fn(new_phi, new_psi, new_omega, new_aa)
        new_phi_deg, new_psi_deg, new_omega_deg = self.unquantize_phi_psi_omega(new_phi, new_psi, new_omega)
        append_torsion(self.pose, "G", new_phi_deg, new_psi_deg,new_omega_deg)
        new_frame = self.get_new_frames()
        self.concat_new_prediction(new_frame)
        return new_phi, new_psi, new_omega, new_aa