import torch 
import numpy as np
from typing import Dict, Union

from torch import nn
from torch_geometric.nn import radius_graph
from D3Fold.data.openfold import Rigid
from D3Fold.data.openfold import rigid_matrix_vector

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

def get_distance_mat_stack(data, min_radius=5, max_radius=20, num_radii=4):
    radius_list = np.linspace(min_radius, max_radius, num_radii)
    stack = []
    for r in radius_list:
        mat = get_distance_matrix(data, r=r)
        stack.append(mat)

    return torch.stack(stack, dim=-1)

