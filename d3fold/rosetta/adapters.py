import os

import numpy as np
import torch
import pyrosetta
from pyrosetta import rosetta

from d3fold.data.torch_data import SingleChainData


# Initialize PyRosetta
pyrosetta.init()

def create_from_torsion(
    sequence, 
    phis,
    psis,
    omegas,
):
    pose = rosetta.core.pose.Pose()
    rosetta.core.pose.make_pose_from_sequence(pose, sequence, "fa_standard")
    for i in range(1, len(sequence) + 1):
        pose.set_phi(i, phis[i - 1])
        pose.set_psi(i, psis[i - 1])
        if i < len(sequence):
            pose.set_omega(i, omegas[i])
    
    return pose

def append_torsion(
    pose,
    symbol,
    phi,
    psi,
    omega,
):
    chm = pyrosetta.rosetta.core.chemical.ChemicalManager.get_instance()
    resiset = chm.residue_type_set('fa_standard')
    res_type = resiset.get_representative_type_name1(symbol)
    new_residue = rosetta.core.conformation.ResidueFactory.create_residue(res_type)
    # Append the new residue to the pose
    rosetta.core.pose.remove_upper_terminus_type_from_pose_residue(pose, pose.total_residue())
    pose.append_residue_by_bond(new_residue, build_ideal_bond=True)

    # Get the position of the new residue
    new_residue_index = pose.total_residue()

    # Set torsion angles for the new residue
    pose.set_phi(new_residue_index, phi)
    pose.set_psi(new_residue_index, psi)
    pose.set_omega(new_residue_index, omega)

def poses_to_dataset(
    poses, 
    type_dict,
    pdb_path,
    processed_path,
    output_names,
    filtern_fns_w_fields
):
    os.makedirs(pdb_path, exist_ok=True)

    for pose,name in zip(poses,output_names):
        pose.dump_pdb(f"{pdb_path}/{name}.ent")

    dataset = SingleChainData(
        chain_dir=pdb_path,
        pickled_dir=processed_path,
        use_mask=True,
        force_process=False,
        skip_preprocess=False,
        limit_by=1000,
        type_dict=type_dict,
        filter_fns_with_fields=filtern_fns_w_fields
    )

    return dataset
