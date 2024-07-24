import numpy as np
import torch
from d3fold.rosetta.adapters import create_from_torsion, poses_to_dataset, append_torsion

# from D3Fold.rosetta.adapters import create_from_torsion, poses_to_dataset
from d3fold.data.torch_data import Collator
from torch.utils.data import DataLoader
from dataclasses import dataclass
from d3fold.data.openfold.rigid_utils import Rotation, Rigid

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
        device: str = "cpu",
        sampling_fn: callable = None,
    ):
        self.seed_structure = seed_structure
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

    def unqauantize_phi_psi_omega(self, new_phi, new_psi, new_omega, n_bins=64):
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

    def concat_new_prediction(self, new_frame, new_aa):
        new_aa = new_aa.unsqueeze(0).to(self.device)
        new_residue_index = self.data["residue_index"].max().unsqueeze(0).unsqueeze(0).to(self.device)
        new_mask = torch.ones(1,1).to(self.device)

        self.data["aatype"] = torch.cat([self.data["aatype"], new_aa], dim=1)
        self.data["backbone_rigid_tensor"] = torch.cat([self.data["backbone_rigid_tensor"], new_frame.to(self.device)], dim=1)
        self.data["residue_index"] = torch.cat([self.data["residue_index"], new_residue_index], dim=1)
        self.data["backbone_rigid_mask"] = torch.cat([self.data["backbone_rigid_mask"], new_mask], dim=1)

    def sample_step(self):
        phi_pred,psi_pred,omega_pred,aa_pred = self.model(self.data.to(self.device))
        new_phi = phi_pred[:,-1]
        new_psi = psi_pred[:,-1]
        new_omega = omega_pred[:,-1]
        new_aa = aa_pred[:,-1]
        new_phi, new_psi, new_omega, new_aa = self.sampling_fn(new_phi, new_psi, new_omega, new_aa)
        new_phi_deg, new_psi_deg, new_omega_deg = self.unquantize_phi_psi_omega(new_phi, new_psi, new_omega)
        aa_code = ind_to_aa[new_aa.item()]
        self.seed_structure.sequence += aa_code
        self.seed_structure.phis = np.append(self.seed_structure.phis, new_phi_deg)
        self.seed_structure.psis = np.append(self.seed_structure.psis, new_psi_deg)
        self.seed_structure.omegas = np.append(self.seed_structure.omegas, new_omega_deg)
        self.pose = create_from_torsion(
            self.seed_structure.sequence,
            self.seed_structure.phis,
            self.seed_structure.psis,
            self.seed_structure.omegas
        )
        new_frame = self.get_new_frames()
        self.concat_new_prediction(new_frame, new_aa)
        return new_phi, new_psi, new_omega, new_aa


def greedy_sampler(pred_phi, pred_psi, pred_omega, pred_aa):
    max_phi = torch.argmax(pred_phi, dim=-1)
    max_psi = torch.argmax(pred_psi, dim=-1)
    max_omega = torch.argmax(pred_omega, dim=-1)
    max_aa = torch.argmax(pred_aa, dim=-1)
    return max_phi, max_psi, max_omega, max_aa

def multinomial(pred_phi, pred_psi, pred_omega, pred_aa):
    pred_phi = torch.softmax(pred_phi, dim=-1)
    pred_psi = torch.softmax(pred_psi, dim=-1)
    pred_omega = torch.softmax(pred_omega, dim=-1)
    pred_aa = torch.softmax(pred_aa, dim=-1)

    samp_phi = torch.multinomial(pred_phi, 1)
    samp_psi = torch.multinomial(pred_psi, 1)
    samp_omega = torch.multinomial(pred_omega, 1)
    samp_aa = torch.multinomial(pred_aa, 1)
    return samp_phi[0], samp_psi[0], samp_omega[0], samp_aa[0]

def select_residues(data, index):
    data = data.clone()
    for key in data.keys():
      try:
        data[key] = data[key][:,index]
      except Exception as e:
        print(e)
        pass
    return data