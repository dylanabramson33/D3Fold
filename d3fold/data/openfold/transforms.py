# Copyright 2021 AlQuraishi Laboratory
# Copyright 2021 DeepMind Technologies Limited
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import numpy as np
import torch
from contextlib import contextmanager

from d3fold.data.openfold import residue_constants as rc
from d3fold.data.openfold.rigid_utils import Rotation, Rigid
from d3fold.data.openfold.rigid_matrix_vector import Rigid3Array
from d3fold.data.openfold.rotation_matrix import Rot3Array
from d3fold.data.openfold.vector import Vec3Array

from d3fold.data.openfold.tensor_utils import (
    batched_gather,
)
from torch import cdist

@contextmanager
def disable_tf32():
    """temporarily disable 32-bit float ops"""
    if torch.cuda.is_available():
        orig_value = torch.backends.cuda.matmul.allow_tf32
        torch.backends.cuda.matmul.allow_tf32 = False
        yield
        torch.backends.cuda.matmul.allow_tf32 = orig_value
    else:
        yield


def cast_to_64bit_ints(protein):
    # We keep all ints as int64
    for k, v in protein.items():
        if v.dtype == torch.int32:
            protein[k] = v.type(torch.int64)

    return protein


def make_one_hot(x, num_classes):
    x_one_hot = torch.zeros(*x.shape, num_classes, device=x.device)
    x_one_hot.scatter_(-1, x.unsqueeze(-1), 1)
    return x_one_hot


def make_seq_mask(protein):
    protein["seq_mask"] = torch.ones(
        protein["aatype"].shape, dtype=torch.float32
    )
    return protein


def make_template_mask(protein):
    protein["template_mask"] = torch.ones(
        protein["template_aatype"].shape[0], dtype=torch.float32
    )
    return protein


def make_all_atom_aatype(protein):
    protein["all_atom_aatype"] = protein["aatype"]
    return protein


def fix_templates_aatype(protein):
    # Map one-hot to indices
    num_templates = protein["template_aatype"].shape[0]
    protein["template_aatype"] = torch.argmax(
        protein["template_aatype"], dim=-1
    )
    # Map hhsearch-aatype to our aatype.
    new_order_list = rc.MAP_HHBLITS_AATYPE_TO_OUR_AATYPE
    new_order = torch.tensor(
        new_order_list, dtype=torch.int64, device=protein["template_aatype"].device,
    ).expand(num_templates, -1)
    protein["template_aatype"] = torch.gather(
        new_order, 1, index=protein["template_aatype"]
    )

    return protein

def squeeze_features(protein):
    """Remove singleton and repeated dimensions in protein features."""
    protein["aatype"] = torch.argmax(protein["aatype"], dim=-1)
    for k in [
        "domain_name",
        "msa",
        "num_alignments",
        "seq_length",
        "sequence",
        "superfamily",
        "deletion_matrix",
        "resolution",
        "between_segment_residues",
        "residue_index",
        "template_all_atom_mask",
    ]:
        if k in protein:
            final_dim = protein[k].shape[-1]
            if isinstance(final_dim, int) and final_dim == 1:
                if torch.is_tensor(protein[k]):
                    protein[k] = torch.squeeze(protein[k], dim=-1)
                else:
                    protein[k] = np.squeeze(protein[k], axis=-1)

    for k in ["seq_length", "num_alignments"]:
        if k in protein:
            protein[k] = protein[k][0]

    return protein


def pseudo_beta_fn(aatype, all_atom_positions, all_atom_mask):
    """Create pseudo beta features."""
    is_gly = torch.eq(aatype, rc.restype_order["G"])
    ca_idx = rc.atom_order["CA"]
    cb_idx = rc.atom_order["CB"]
    pseudo_beta = torch.where(
        torch.tile(is_gly[..., None], [1] * len(is_gly.shape) + [3]),
        all_atom_positions[..., ca_idx, :],
        all_atom_positions[..., cb_idx, :],
    )

    if all_atom_mask is not None:
        pseudo_beta_mask = torch.where(
            is_gly, all_atom_mask[..., ca_idx], all_atom_mask[..., cb_idx]
        )
        return pseudo_beta, pseudo_beta_mask
    else:
        return pseudo_beta


def make_pseudo_beta(protein, prefix=""):
    """Create pseudo-beta (alpha for glycine) position and mask."""
    assert prefix in ["", "template_"]
    (
        protein[prefix + "pseudo_beta"],
        protein[prefix + "pseudo_beta_mask"],
    ) = pseudo_beta_fn(
        protein["template_aatype" if prefix else "aatype"],
        protein[prefix + "all_atom_positions"],
        protein["template_all_atom_mask" if prefix else "all_atom_mask"],
    )
    return protein


def crop_templates(protein, max_templates):
    for k, v in protein.items():
        if k.startswith("template_"):
            protein[k] = v[:max_templates]
    return protein


def make_atom14_masks(protein):
    """Construct denser atom positions (14 dimensions instead of 37)."""
    restype_atom14_to_atom37 = []
    restype_atom37_to_atom14 = []
    restype_atom14_mask = []

    for rt in rc.restypes:
        atom_names = rc.restype_name_to_atom14_names[rc.restype_1to3[rt]]
        restype_atom14_to_atom37.append(
            [(rc.atom_order[name] if name else 0) for name in atom_names]
        )
        atom_name_to_idx14 = {name: i for i, name in enumerate(atom_names)}
        restype_atom37_to_atom14.append(
            [
                (atom_name_to_idx14[name] if name in atom_name_to_idx14 else 0)
                for name in rc.atom_types
            ]
        )

        restype_atom14_mask.append(
            [(1.0 if name else 0.0) for name in atom_names]
        )

    # Add dummy mapping for restype 'UNK'
    restype_atom14_to_atom37.append([0] * 14)
    restype_atom37_to_atom14.append([0] * 37)
    restype_atom14_mask.append([0.0] * 14)

    restype_atom14_to_atom37 = torch.tensor(
        restype_atom14_to_atom37,
        dtype=torch.int32,
        device=protein["aatype"].device,
    )
    restype_atom37_to_atom14 = torch.tensor(
        restype_atom37_to_atom14,
        dtype=torch.int32,
        device=protein["aatype"].device,
    )
    restype_atom14_mask = torch.tensor(
        restype_atom14_mask,
        dtype=torch.float32,
        device=protein["aatype"].device,
    )
    protein_aatype = protein['aatype'].to(torch.long)

    # create the mapping for (residx, atom14) --> atom37, i.e. an array
    # with shape (num_res, 14) containing the atom37 indices for this protein
    residx_atom14_to_atom37 = restype_atom14_to_atom37[protein_aatype]
    residx_atom14_mask = restype_atom14_mask[protein_aatype]

    protein["atom14_atom_exists"] = residx_atom14_mask
    protein["residx_atom14_to_atom37"] = residx_atom14_to_atom37.long()

    # create the gather indices for mapping back
    residx_atom37_to_atom14 = restype_atom37_to_atom14[protein_aatype]
    protein["residx_atom37_to_atom14"] = residx_atom37_to_atom14.long()

    # create the corresponding mask
    restype_atom37_mask = torch.zeros(
        [21, 37], dtype=torch.float32, device=protein["aatype"].device
    )
    for restype, restype_letter in enumerate(rc.restypes):
        restype_name = rc.restype_1to3[restype_letter]
        atom_names = rc.residue_atoms[restype_name]
        for atom_name in atom_names:
            atom_type = rc.atom_order[atom_name]
            restype_atom37_mask[restype, atom_type] = 1

    residx_atom37_mask = restype_atom37_mask[protein_aatype]
    protein["atom37_atom_exists"] = residx_atom37_mask

    return protein


def make_atom14_positions(protein):
    """Constructs denser atom positions (14 dimensions instead of 37)."""
    residx_atom14_mask = protein["atom14_atom_exists"]
    residx_atom14_to_atom37 = protein["residx_atom14_to_atom37"]

    # Create a mask for known ground truth positions.
    residx_atom14_gt_mask = residx_atom14_mask * batched_gather(
        protein["all_atom_mask"],
        residx_atom14_to_atom37,
        dim=-1,
        no_batch_dims=len(protein["all_atom_mask"].shape[:-1]),
    )

    # Gather the ground truth positions.
    residx_atom14_gt_positions = residx_atom14_gt_mask[..., None] * (
        batched_gather(
            protein["all_atom_positions"],
            residx_atom14_to_atom37,
            dim=-2,
            no_batch_dims=len(protein["all_atom_positions"].shape[:-2]),
        )
    )

    protein["atom14_atom_exists"] = residx_atom14_mask
    protein["atom14_gt_exists"] = residx_atom14_gt_mask
    protein["atom14_gt_positions"] = residx_atom14_gt_positions

    # As the atom naming is ambiguous for 7 of the 20 amino acids, provide
    # alternative ground truth coordinates where the naming is swapped
    restype_3 = [rc.restype_1to3[res] for res in rc.restypes]
    restype_3 += ["UNK"]

    # Matrices for renaming ambiguous atoms.
    all_matrices = {
        res: torch.eye(
            14,
            dtype=protein["all_atom_mask"].dtype,
            device=protein["all_atom_mask"].device,
        )
        for res in restype_3
    }
    for resname, swap in rc.residue_atom_renaming_swaps.items():
        correspondences = torch.arange(
            14, device=protein["all_atom_mask"].device
        )
        for source_atom_swap, target_atom_swap in swap.items():
            source_index = rc.restype_name_to_atom14_names[resname].index(
                source_atom_swap
            )
            target_index = rc.restype_name_to_atom14_names[resname].index(
                target_atom_swap
            )
            correspondences[source_index] = target_index
            correspondences[target_index] = source_index
            renaming_matrix = protein["all_atom_mask"].new_zeros((14, 14))
            for index, correspondence in enumerate(correspondences):
                renaming_matrix[index, correspondence] = 1.0
        all_matrices[resname] = renaming_matrix

    renaming_matrices = torch.stack(
        [all_matrices[restype] for restype in restype_3]
    )

    # Pick the transformation matrices for the given residue sequence
    # shape (num_res, 14, 14).
    renaming_transform = renaming_matrices[protein["aatype"]]

    # Apply it to the ground truth positions. shape (num_res, 14, 3).
    alternative_gt_positions = torch.einsum(
        "...rac,...rab->...rbc", residx_atom14_gt_positions, renaming_transform
    )
    protein["atom14_alt_gt_positions"] = alternative_gt_positions

    # Create the mask for the alternative ground truth (differs from the
    # ground truth mask, if only one of the atoms in an ambiguous pair has a
    # ground truth position).
    alternative_gt_mask = torch.einsum(
        "...ra,...rab->...rb", residx_atom14_gt_mask, renaming_transform
    )
    protein["atom14_alt_gt_exists"] = alternative_gt_mask

    # Create an ambiguous atoms mask.  shape: (21, 14).
    restype_atom14_is_ambiguous = protein["all_atom_mask"].new_zeros((21, 14))
    for resname, swap in rc.residue_atom_renaming_swaps.items():
        for atom_name1, atom_name2 in swap.items():
            restype = rc.restype_order[rc.restype_3to1[resname]]
            atom_idx1 = rc.restype_name_to_atom14_names[resname].index(
                atom_name1
            )
            atom_idx2 = rc.restype_name_to_atom14_names[resname].index(
                atom_name2
            )
            restype_atom14_is_ambiguous[restype, atom_idx1] = 1
            restype_atom14_is_ambiguous[restype, atom_idx2] = 1

    # From this create an ambiguous_mask for the given sequence.
    protein["atom14_atom_is_ambiguous"] = restype_atom14_is_ambiguous[
        protein["aatype"]
    ]

    return protein


def atom37_to_frames(protein, eps=1e-8):
    is_multimer = "asym_id" in protein
    aatype = protein["aatype"]
    all_atom_positions = protein["all_atom_positions"]
    all_atom_mask = protein["all_atom_mask"]

    if is_multimer:
        all_atom_positions = Vec3Array.from_array(all_atom_positions)

    batch_dims = len(aatype.shape[:-1])

    restype_rigidgroup_base_atom_names = np.full([21, 8, 3], "", dtype=object)
    restype_rigidgroup_base_atom_names[:, 0, :] = ["C", "CA", "N"]
    restype_rigidgroup_base_atom_names[:, 3, :] = ["CA", "C", "O"]

    for restype, restype_letter in enumerate(rc.restypes):
        resname = rc.restype_1to3[restype_letter]
        for chi_idx in range(4):
            if rc.chi_angles_mask[restype][chi_idx]:
                names = rc.chi_angles_atoms[resname][chi_idx]
                restype_rigidgroup_base_atom_names[
                    restype, chi_idx + 4, :
                ] = names[1:]

    restype_rigidgroup_mask = all_atom_mask.new_zeros(
        (*aatype.shape[:-1], 21, 8),
    )
    restype_rigidgroup_mask[..., 0] = 1
    restype_rigidgroup_mask[..., 3] = 1
    restype_rigidgroup_mask[..., :20, 4:] = all_atom_mask.new_tensor(
        rc.chi_angles_mask
    )

    lookuptable = rc.atom_order.copy()
    lookuptable[""] = 0
    lookup = np.vectorize(lambda x: lookuptable[x])
    restype_rigidgroup_base_atom37_idx = lookup(
        restype_rigidgroup_base_atom_names,
    )
    restype_rigidgroup_base_atom37_idx = aatype.new_tensor(
        restype_rigidgroup_base_atom37_idx,
    )
    restype_rigidgroup_base_atom37_idx = (
        restype_rigidgroup_base_atom37_idx.view(
            *((1,) * batch_dims), *restype_rigidgroup_base_atom37_idx.shape
        )
    )

    residx_rigidgroup_base_atom37_idx = batched_gather(
        restype_rigidgroup_base_atom37_idx,
        aatype,
        dim=-3,
        no_batch_dims=batch_dims,
    )

    if is_multimer:
        base_atom_pos = [batched_gather(
            pos,
            residx_rigidgroup_base_atom37_idx,
            dim=-1,
            no_batch_dims=len(all_atom_positions.shape[:-1]),
        ) for pos in all_atom_positions]
        base_atom_pos = Vec3Array.from_array(torch.stack(base_atom_pos, dim=-1))
    else:
        base_atom_pos = batched_gather(
            all_atom_positions,
            residx_rigidgroup_base_atom37_idx,
            dim=-2,
            no_batch_dims=len(all_atom_positions.shape[:-2]),
        )

    if is_multimer:
        point_on_neg_x_axis = base_atom_pos[:, :, 0]
        origin = base_atom_pos[:, :, 1]
        point_on_xy_plane = base_atom_pos[:, :, 2]
        gt_rotation = Rot3Array.from_two_vectors(
            origin - point_on_neg_x_axis, point_on_xy_plane - origin)

        gt_frames = Rigid3Array(gt_rotation, origin)
    else:
        gt_frames = Rigid.from_3_points(
            p_neg_x_axis=base_atom_pos[..., 0, :],
            origin=base_atom_pos[..., 1, :],
            p_xy_plane=base_atom_pos[..., 2, :],
            eps=eps,
        )

    group_exists = batched_gather(
        restype_rigidgroup_mask,
        aatype,
        dim=-2,
        no_batch_dims=batch_dims,
    )

    gt_atoms_exist = batched_gather(
        all_atom_mask,
        residx_rigidgroup_base_atom37_idx,
        dim=-1,
        no_batch_dims=len(all_atom_mask.shape[:-1]),
    )
    gt_exists = torch.min(gt_atoms_exist, dim=-1)[0] * group_exists

    rots = torch.eye(3, dtype=all_atom_mask.dtype, device=aatype.device)
    rots = torch.tile(rots, (*((1,) * batch_dims), 8, 1, 1))
    rots[..., 0, 0, 0] = -1
    rots[..., 0, 2, 2] = -1

    if is_multimer:
        gt_frames = gt_frames.compose_rotation(
            Rot3Array.from_array(rots))
    else:
        rots = Rotation(rot_mats=rots)
        gt_frames = gt_frames.compose(Rigid(rots, None))

    restype_rigidgroup_is_ambiguous = all_atom_mask.new_zeros(
        *((1,) * batch_dims), 21, 8
    )
    restype_rigidgroup_rots = torch.eye(
        3, dtype=all_atom_mask.dtype, device=aatype.device
    )
    restype_rigidgroup_rots = torch.tile(
        restype_rigidgroup_rots,
        (*((1,) * batch_dims), 21, 8, 1, 1),
    )

    for resname, _ in rc.residue_atom_renaming_swaps.items():
        restype = rc.restype_order[rc.restype_3to1[resname]]
        chi_idx = int(sum(rc.chi_angles_mask[restype]) - 1)
        restype_rigidgroup_is_ambiguous[..., restype, chi_idx + 4] = 1
        restype_rigidgroup_rots[..., restype, chi_idx + 4, 1, 1] = -1
        restype_rigidgroup_rots[..., restype, chi_idx + 4, 2, 2] = -1

    residx_rigidgroup_is_ambiguous = batched_gather(
        restype_rigidgroup_is_ambiguous,
        aatype,
        dim=-2,
        no_batch_dims=batch_dims,
    )

    residx_rigidgroup_ambiguity_rot = batched_gather(
        restype_rigidgroup_rots,
        aatype,
        dim=-4,
        no_batch_dims=batch_dims,
    )

    if is_multimer:
        ambiguity_rot = Rot3Array.from_array(residx_rigidgroup_ambiguity_rot)

        # Create the alternative ground truth frames.
        alt_gt_frames = gt_frames.compose_rotation(ambiguity_rot)
    else:
        residx_rigidgroup_ambiguity_rot = Rotation(
            rot_mats=residx_rigidgroup_ambiguity_rot
        )
        alt_gt_frames = gt_frames.compose(
            Rigid(residx_rigidgroup_ambiguity_rot, None)
        )

    gt_frames_tensor = gt_frames.to_tensor_4x4()
    alt_gt_frames_tensor = alt_gt_frames.to_tensor_4x4()

    protein["rigidgroups_gt_frames"] = gt_frames_tensor
    protein["rigidgroups_gt_exists"] = gt_exists
    protein["rigidgroups_group_exists"] = group_exists
    protein["rigidgroups_group_is_ambiguous"] = residx_rigidgroup_is_ambiguous
    protein["rigidgroups_alt_gt_frames"] = alt_gt_frames_tensor

    return protein


def atom14_to_atom37(atom14, batch):
    atom37_data = batched_gather(
        atom14,
        batch["residx_atom37_to_atom14"],
        dim=-2,
        no_batch_dims=len(atom14.shape[:-2]),
    )

    atom37_data = atom37_data * batch["atom37_atom_exists"][..., None]

    return atom37_data


def get_chi_atom_indices():
    """Returns atom indices needed to compute chi angles for all residue types.

    Returns:
      A tensor of shape [residue_types=21, chis=4, atoms=4]. The residue types are
      in the order specified in rc.restypes + unknown residue type
      at the end. For chi angles which are not defined on the residue, the
      positions indices are by default set to 0.
    """
    chi_atom_indices = []
    for residue_name in rc.restypes:
        residue_name = rc.restype_1to3[residue_name]
        residue_chi_angles = rc.chi_angles_atoms[residue_name]
        atom_indices = []
        for chi_angle in residue_chi_angles:
            atom_indices.append([rc.atom_order[atom] for atom in chi_angle])
        for _ in range(4 - len(atom_indices)):
            atom_indices.append(
                [0, 0, 0, 0]
            )  # For chi angles not defined on the AA.
        chi_atom_indices.append(atom_indices)

    chi_atom_indices.append([[0, 0, 0, 0]] * 4)  # For UNKNOWN residue.

    return chi_atom_indices


def atom37_to_torsion_angles(
    protein,
    prefix="",
):
    """
    Convert coordinates to torsion angles.

    This function is extremely sensitive to floating point imprecisions
    and should be run with double precision whenever possible.

    Args:
        Dict containing:
            * (prefix)aatype:
                [*, N_res] residue indices
            * (prefix)all_atom_positions:
                [*, N_res, 37, 3] atom positions (in atom37
                format)
            * (prefix)all_atom_mask:
                [*, N_res, 37] atom position mask
    Returns:
        The same dictionary updated with the following features:

        "(prefix)torsion_angles_sin_cos" ([*, N_res, 7, 2])
            Torsion angles
        "(prefix)alt_torsion_angles_sin_cos" ([*, N_res, 7, 2])
            Alternate torsion angles (accounting for 180-degree symmetry)
        "(prefix)torsion_angles_mask" ([*, N_res, 7])
            Torsion angles mask
    """

    device = "cpu" if protein[prefix + "aatype"].device.type == "cpu" else "cuda"
    with disable_tf32(), torch.autocast(device_type=device, enabled=False):
        aatype = protein[prefix + "aatype"]
        all_atom_positions = protein[prefix + "all_atom_positions"]
        all_atom_mask = protein[prefix + "all_atom_mask"]

        aatype = torch.clamp(aatype, max=20)

        pad = all_atom_positions.new_zeros(
            [*all_atom_positions.shape[:-3], 1, 37, 3]
        )
        prev_all_atom_positions = torch.cat(
            [pad, all_atom_positions[..., :-1, :, :]], dim=-3
        )

        pad = all_atom_mask.new_zeros([*all_atom_mask.shape[:-2], 1, 37])
        prev_all_atom_mask = torch.cat([pad, all_atom_mask[..., :-1, :]], dim=-2)

        pre_omega_atom_pos = torch.cat(
            [prev_all_atom_positions[..., 1:3, :], all_atom_positions[..., :2, :]],
            dim=-2,
        )
        phi_atom_pos = torch.cat(
            [prev_all_atom_positions[..., 2:3, :], all_atom_positions[..., :3, :]],
            dim=-2,
        )
        psi_atom_pos = torch.cat(
            [all_atom_positions[..., :3, :], all_atom_positions[..., 4:5, :]],
            dim=-2,
        )

        pre_omega_mask = torch.prod(
            prev_all_atom_mask[..., 1:3], dim=-1
        ) * torch.prod(all_atom_mask[..., :2], dim=-1)
        phi_mask = prev_all_atom_mask[..., 2] * torch.prod(
            all_atom_mask[..., :3], dim=-1, dtype=all_atom_mask.dtype
        )
        psi_mask = (
            torch.prod(all_atom_mask[..., :3], dim=-1, dtype=all_atom_mask.dtype)
            * all_atom_mask[..., 4]
        )

        chi_atom_indices = torch.as_tensor(
            get_chi_atom_indices(), device=aatype.device
        )

        atom_indices = chi_atom_indices[..., aatype, :, :]
        chis_atom_pos = batched_gather(
            all_atom_positions, atom_indices, -2, len(atom_indices.shape[:-2])
        )

        chi_angles_mask = list(rc.chi_angles_mask)
        chi_angles_mask.append([0.0, 0.0, 0.0, 0.0])
        chi_angles_mask = all_atom_mask.new_tensor(chi_angles_mask)

        chis_mask = chi_angles_mask[aatype, :]

        chi_angle_atoms_mask = batched_gather(
            all_atom_mask,
            atom_indices,
            dim=-1,
            no_batch_dims=len(atom_indices.shape[:-2]),
        )
        chi_angle_atoms_mask = torch.prod(
            chi_angle_atoms_mask, dim=-1, dtype=chi_angle_atoms_mask.dtype
        )
        chis_mask = chis_mask * chi_angle_atoms_mask

        torsions_atom_pos = torch.cat(
            [
                pre_omega_atom_pos[..., None, :, :],
                phi_atom_pos[..., None, :, :],
                psi_atom_pos[..., None, :, :],
                chis_atom_pos,
            ],
            dim=-3,
        )

        torsion_angles_mask = torch.cat(
            [
                pre_omega_mask[..., None],
                phi_mask[..., None],
                psi_mask[..., None],
                chis_mask,
            ],
            dim=-1,
        )

        torsion_frames = Rigid.from_3_points(
            torsions_atom_pos[..., 1, :],
            torsions_atom_pos[..., 2, :],
            torsions_atom_pos[..., 0, :],
            eps=1e-8,
        )

        fourth_atom_rel_pos = torsion_frames.invert().apply(
            torsions_atom_pos[..., 3, :]
        )

        torsion_angles_sin_cos = torch.stack(
            [fourth_atom_rel_pos[..., 2], fourth_atom_rel_pos[..., 1]], dim=-1
        )

        denom = torch.sqrt(
            torch.sum(
                torch.square(torsion_angles_sin_cos),
                dim=-1,
                dtype=torsion_angles_sin_cos.dtype,
                keepdims=True,
            )
            + 1e-8
        )
        torsion_angles_sin_cos = torsion_angles_sin_cos / denom

        torsion_angles_sin_cos = torsion_angles_sin_cos * all_atom_mask.new_tensor(
            [1.0, 1.0, -1.0, 1.0, 1.0, 1.0, 1.0],
        )[((None,) * len(torsion_angles_sin_cos.shape[:-2])) + (slice(None), None)]

        chi_is_ambiguous = torsion_angles_sin_cos.new_tensor(
            rc.chi_pi_periodic,
        )[aatype, ...]

        mirror_torsion_angles = torch.cat(
            [
                all_atom_mask.new_ones(*aatype.shape, 3),
                1.0 - 2.0 * chi_is_ambiguous,
            ],
            dim=-1,
        )

        alt_torsion_angles_sin_cos = (
            torsion_angles_sin_cos * mirror_torsion_angles[..., None]
        )

        protein[prefix + "torsion_angles_sin_cos"] = torsion_angles_sin_cos
        protein[prefix + "alt_torsion_angles_sin_cos"] = alt_torsion_angles_sin_cos
        protein[prefix + "torsion_angles_mask"] = torsion_angles_mask

        return protein


def get_backbone_frames(protein):
    # DISCREPANCY: AlphaFold uses tensor_7s here. I don't know why.
    protein["backbone_rigid_tensor"] = protein["rigidgroups_gt_frames"][
        ..., 0, :, :
    ]
    protein["backbone_rigid_mask"] = protein["rigidgroups_gt_exists"][..., 0]

    return protein


def get_chi_angles(protein):
    dtype = protein["all_atom_mask"].dtype
    protein["chi_angles_sin_cos"] = (
        protein["torsion_angles_sin_cos"][..., 3:, :]
    ).to(dtype)
    protein["chi_mask"] = protein["torsion_angles_mask"][..., 3:].to(dtype)

    return protein

def get_distance_mat_stack(protein, min_radius=5, max_radius=26, num_radii=64):
    # this ignores glycine for now
    CA_INDEX = rc.atom_types.index("CA")
    ca_pos = protein["all_atom_positions"][:, CA_INDEX]

    dists = cdist(ca_pos, ca_pos)
    bins = torch.linspace(min_radius, max_radius, num_radii)
    buckets = torch.bucketize(dists, bins)
    scattered = torch.zeros(
      len(buckets), len(buckets), num_radii + 1, device=buckets.device
    )
    scattered.scatter_(2, buckets.unsqueeze(-1), 1)
    protein["distance_mat_stack"] = scattered
    return protein

def get_quantized_phi_psi_omega(protein, n_bins=64):
    phi_psi_omega = protein["torsion_angles_sin_cos"][:, 0:3] / 180.0
    quantized_phi_psi_omega = torch.round(phi_psi_omega * n_bins) + n_bins
    protein["quantized_phi_psi_omega"] = quantized_phi_psi_omega
    return protein

def convert_angles_to_degrees(protein):
    protein["torsion_angles_sin_cos"] = torch.atan2(
        protein["torsion_angles_sin_cos"][..., 0],
        protein["torsion_angles_sin_cos"][..., 1],
    ) * 180.0 / np.pi
    return protein

def relative_positions(protein):
    num_res = protein["aatype"].shape[-1]
    pairwise_dist = protein["residue_index"].expand(num_res, -1) - protein["residue_index"].unsqueeze(-1)
    pairwise_dist = torch.abs(pairwise_dist).clamp(max=32)
    protein["pairwise_seq_dist"] = pairwise_dist
    return protein