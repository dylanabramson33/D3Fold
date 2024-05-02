import torch
from torch import nn
from mamba_ssm import Mamba
from typing import Dict, Union

import esm
from torch_geometric.nn import radius_graph
import lightning as L
import numpy as np
from D3Fold.data.openfold.residue_constants import (
    restype_rigid_group_default_frame,
    restype_atom14_to_rigid_group,
    restype_atom14_mask,
    restype_atom14_rigid_group_positions,
)
from D3Fold.data.openfold import rigid_matrix_vector, rotation_matrix, vector
from D3Fold.data.openfold import Rotation, Rigid

from model.losses import sequence_loss
from invariant_point_attention import InvariantPointAttention

def torsion_angles_to_frames(
    r: Union[Rigid, rigid_matrix_vector.Rigid3Array],
    alpha: torch.Tensor,
    aatype: torch.Tensor,
    rrgdf: torch.Tensor,
):

    rigid_type = type(r)

    # [*, N, 8, 4, 4]
    default_4x4 = rrgdf[aatype, ...]

    # [*, N, 8] transformations, i.e.
    #   One [*, N, 8, 3, 3] rotation matrix and
    #   One [*, N, 8, 3]    translation matrix
    default_r = rigid_type.from_tensor_4x4(default_4x4)

    bb_rot = alpha.new_zeros((*((1,) * len(alpha.shape[:-1])), 2))
    bb_rot[..., 1] = 1

    # [*, N, 8, 2]
    alpha = torch.cat(
        [bb_rot.expand(*alpha.shape[:-2], -1, -1), alpha], dim=-2
    )

    # [*, N, 8, 3, 3]
    # Produces rotation matrices of the form:
    # [
    #   [1, 0  , 0  ],
    #   [0, a_2,-a_1],
    #   [0, a_1, a_2]
    # ]
    # This follows the original code rather than the supplement, which uses
    # different indices.

    all_rots = alpha.new_zeros(default_r.shape + (4, 4))
    all_rots[..., 0, 0] = 1
    all_rots[..., 1, 1] = alpha[..., 1]
    all_rots[..., 1, 2] = -alpha[..., 0]
    all_rots[..., 2, 1:3] = alpha

    all_rots = rigid_type.from_tensor_4x4(all_rots)
    all_frames = default_r.compose(all_rots)

    chi2_frame_to_frame = all_frames[..., 5]
    chi3_frame_to_frame = all_frames[..., 6]
    chi4_frame_to_frame = all_frames[..., 7]

    chi1_frame_to_bb = all_frames[..., 4]
    chi2_frame_to_bb = chi1_frame_to_bb.compose(chi2_frame_to_frame)
    chi3_frame_to_bb = chi2_frame_to_bb.compose(chi3_frame_to_frame)
    chi4_frame_to_bb = chi3_frame_to_bb.compose(chi4_frame_to_frame)

    all_frames_to_bb = rigid_type.cat(
        [
            all_frames[..., :5],
            chi2_frame_to_bb.unsqueeze(-1),
            chi3_frame_to_bb.unsqueeze(-1),
            chi4_frame_to_bb.unsqueeze(-1),
        ],
        dim=-1,
    )

    all_frames_to_global = r[..., None].compose(all_frames_to_bb)

    return all_frames_to_global


def frames_and_literature_positions_to_atom14_pos(
    r: Union[Rigid, rigid_matrix_vector.Rigid3Array],
    aatype: torch.Tensor,
    default_frames,
    group_idx,
    atom_mask,
    lit_positions,
):
    # [*, N, 14, 4, 4]
    default_4x4 = default_frames[aatype, ...]

    # [*, N, 14]
    group_mask = group_idx[aatype, ...]

    # [*, N, 14, 8]
    group_mask = nn.functional.one_hot(
        group_mask,
        num_classes=default_frames.shape[-3],
    )

    # [*, N, 14, 8]
    t_atoms_to_global = r[..., None, :] * group_mask

    # [*, N, 14]
    t_atoms_to_global = t_atoms_to_global.map_tensor_fn(
        lambda x: torch.sum(x, dim=-1)
    )

    # [*, N, 14]
    atom_mask = atom_mask[aatype, ...].unsqueeze(-1)

    # [*, N, 14, 3]
    lit_positions = lit_positions[aatype, ...]
    pred_positions = t_atoms_to_global.apply(lit_positions)
    pred_positions = pred_positions * atom_mask

    return pred_positions



class BackboneUpdate(nn.Module):
    """
    Implements part of Algorithm 23.
    """

    def __init__(self, c_s):
        """
        Args:
            c_s:
                Single representation channel dimension
        """
        super(BackboneUpdate, self).__init__()

        self.c_s = c_s

        self.linear = nn.Linear(self.c_s, 6, init="final")

    def forward(self, s: torch.Tensor):
        """
        Args:
            [*, N_res, C_s] single representation
        Returns:
            [*, N_res, 6] update vector
        """
        # [*, 6]
        update = self.linear(s)

        return update


class StructureModuleTransitionLayer(nn.Module):
    def __init__(self, c):
        super(StructureModuleTransitionLayer, self).__init__()

        self.c = c

        self.linear_1 = nn.Linear(self.c, self.c, init="relu")
        self.linear_2 = nn.Linear(self.c, self.c, init="relu")
        self.linear_3 = nn.Linear(self.c, self.c, init="final")

        self.relu = nn.ReLU()

    def forward(self, s):
        s_initial = s
        s = self.linear_1(s)
        s = self.relu(s)
        s = self.linear_2(s)
        s = self.relu(s)
        s = self.linear_3(s)

        s = s + s_initial

        return s

class FoldingTrunk(nn.Module):
    def __init__(self, s_dim_in=1280, s_dim_out=32, z_dim_in=1,z_dim_out=32):
        super().__init__()
        self.s_project = nn.Linear(s_dim_in, s_dim_out)
        self.z_project = nn.Linear(z_dim_in, z_dim_out)
        # this is black magic rn, I have no idea what Mamba does
        self.mamba = Mamba(
            d_model=s_dim_out, # Model dimension d_model
            d_state=16,  # SSM state expansion factor
            d_conv=4,    # Local convolution width
            expand=2,    # Block expansion factor
        )

    def outer_product(self, x, y):
      return torch.einsum("bsf,btf->bstf", x, y)

    def forward(self, s, z):
        s_prime = self.s_project(s)
        z_prime = self.z_project(z)
        z_prime = z_prime + self.outer_product(s_prime, s_prime)
        return s_prime, z_prime


class IPA(nn.Module):
    def __init__(self, dim=32):
        self.ipa = InvariantPointAttention(
            dim=dim,
            heads=8,
            scalar_key_dim=16,
            scalar_value_dim=16,
            point_key_dim=4,
            point_value_dim=4
        )

    def forward(self, s, z, data):
        return self.ipa(s, z, rotations=data.frames_R, translations=data.frames_t, mask=data.mask)


class D3Fold(L.LightningModule):
    def __init__(self, mamba_layers=3, freeze_esm=False):
        super().__init__()
        self.losses = [sequence_loss]
        model, _ = esm.pretrained.esm2_t33_650M_UR50D()
        self.esm = model
        if freeze_esm:
            self.esm.eval()
            self.esm.requires_grad_(False)
        esm_dim = 1280
        self.folding_trunk = FoldingTrunk(
            s_dim_in=esm_dim,
            s_dim_out=32,
            z_dim_in=1,
            z_dim_out=32,
        )

        self.final_z_proj = nn.Linear(32, 1)
        self.final_s_proj = nn.Linear(32, 21)

    @staticmethod
    def get_distance_matrix(data, r=10):
      num_graphs = data.seq.shape[0]
      contact_edges = radius_graph(data.coords, r=r,  batch=data.batch)
      graph_sizes = data.batch.bincount()
      largest_graph = torch.max(graph_sizes)
      num_nodes = data.coords.shape[0]
      contact_mat = torch.zeros((num_nodes, num_nodes))
      src_indices, tgt_indices = contact_edges[0], contact_edges[1]
      contact_mat[src_indices, tgt_indices] = 1
      contact_mat[tgt_indices, src_indices] = 1
      sequence_of_contacts = torch.zeros((num_graphs, largest_graph, largest_graph))
      current_graph_index = 0
      for i in range(num_graphs):
          contacts = contact_mat[current_graph_index:current_graph_index+graph_sizes[i], current_graph_index:current_graph_index+graph_sizes[i]]
          contacts += torch.diag(torch.ones(contacts.shape[0]))

          # if shape is less than max pad with zeros
          if contacts.shape[0] < largest_graph:
              contacts = torch.nn.functional.pad(contacts, (0, largest_graph - contacts.shape[0], 0, largest_graph - contacts.shape[1]))

          sequence_of_contacts[i] = contacts
          current_graph_index += graph_sizes[i]

      return sequence_of_contacts

    @staticmethod
    def get_distance_mat_stack(data, min_radius=5, max_radius=20, num_radii=4):
        radius_list = np.linspace(min_radius, max_radius, num_radii)
        stack = []
        for r in radius_list:
            mat = D3Fold.get_distance_matrix(data, r=r)
            stack.append(mat)

        return torch.stack(stack, dim=-1)

    def forward(self, batch):
        esm_seq = batch.tokens
        seq_embed = self.esm(esm_seq.long(), repr_layers=[33], return_contacts=True)
        # attention based contacts
        att_contacts = seq_embed['contacts'].unsqueeze(-1)
        rep = seq_embed['representations'][33][:,1:-1]
        s, z = self.folding_trunk(rep, att_contacts)
        z = self.final_z_proj(z).squeeze(-1)
        s = self.final_s_proj(s)
        s = s.softmax(dim=-1)
        att_contacts = att_contacts.squeeze(-1)
        return s, z

    def training_step(self, batch, batch_idx):
        s, z = self.forward(batch)
        distance_mats = self.get_distance_mat_stack(batch)
        loss = 0
        for loss_fn in self.losses:
            # this is shit but on right track
            if loss_fn.representation_target == "seq":
                loss += loss_fn(s, batch.seq, mask=batch.mask)
            if loss_fn.representation_target == "pair":
                loss += loss_fn(z, distance_mats)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=1e-4)
        return optimizer


    def _init_residue_constants(self, float_dtype, device):
        if not hasattr(self, "default_frames"):
            self.register_buffer(
                "default_frames",
                torch.tensor(
                    restype_rigid_group_default_frame,
                    dtype=float_dtype,
                    device=device,
                    requires_grad=False,
                ),
                persistent=False,
            )
        if not hasattr(self, "group_idx"):
            self.register_buffer(
                "group_idx",
                torch.tensor(
                    restype_atom14_to_rigid_group,
                    device=device,
                    requires_grad=False,
                ),
                persistent=False,
            )
        if not hasattr(self, "atom_mask"):
            self.register_buffer(
                "atom_mask",
                torch.tensor(
                    restype_atom14_mask,
                    dtype=float_dtype,
                    device=device,
                    requires_grad=False,
                ),
                persistent=False,
            )
        if not hasattr(self, "lit_positions"):
            self.register_buffer(
                "lit_positions",
                torch.tensor(
                    restype_atom14_rigid_group_positions,
                    dtype=float_dtype,
                    device=device,
                    requires_grad=False,
                ),
                persistent=False,
            )
    def torsion_angles_to_frames(self, r, alpha, f):
        # Lazily initialize the residue constants on the correct device
        self._init_residue_constants(alpha.dtype, alpha.device)
        # Separated purely to make testing less annoying
        return torsion_angles_to_frames(r, alpha, f, self.default_frames)

    def frames_and_literature_positions_to_atom14_pos(
            self, r, f  # [*, N, 8]  # [*, N]
    ):
        # Lazily initialize the residue constants on the correct device
        self._init_residue_constants(r.dtype, r.device)
        return frames_and_literature_positions_to_atom14_pos(
            r,
            f,
            self.default_frames,
            self.group_idx,
            self.atom_mask,
            self.lit_positions,
        )
