import concurrent.futures
import os

from pyrosetta import *
# Initialize PyRosetta
init()

def fast_idealize(pdb: str, threshold: float = 1.0, output_dir="./idealized"):
    """
    Fast idealization of a pdb file
    """
    pose = pose_from_pdb(pdb)
    original_pose = Pose(pose)
    idealize = rosetta.protocols.idealize.IdealizeMover()
    idealize = idealize.fast(True)
    idealize.apply(pose)
    # get rmsd between original and idealized pose
    rmsd = rosetta.core.scoring.CA_rmsd(original_pose, pose)
    if rmsd > threshold:
        print(f"Warning: rmsd of idealized pose is {rmsd} > {threshold}")
        return None
    else:
        os.makedirs(output_dir, exist_ok=True)
        pose.dump_pdb(os.path.join(output_dir, os.path.basename(pdb)))

    return pose, rmsd

def idealize_directory(directory: str, output_dir="./idealized"):
    """
    Idealize all pdbs in a directory
    """
    pdbs = [f for f in os.listdir(directory)]
    rmsds = []
    for pdb in pdbs:
        if os.path.exists(os.path.join(output_dir, pdb)):
            print(f"Skipping {pdb} as it already exists")
            continue
        if pdb.endswith(".pdb") or pdb.endswith(".ent"):
            try:
              rmsd = fast_idealize(os.path.join(directory, pdb), output_dir=output_dir)
            except:
              print(f"Error idealizing {pdb}")
              continue
            if rmsd is not None:
                rmsds.append(rmsd)
    return rmsds

