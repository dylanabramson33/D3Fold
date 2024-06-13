import concurrent.futures
import os

from pyrosetta import *
# Initialize PyRosetta
init()

def fast_idealize(pdb: str, threshold: float = 1.0):
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
    return pose

def idealize_directory(directory: str):
    """
    Idealize all pdbs in a directory
    """
    pdbs = [f for f in os.listdir(directory)]
    with concurrent.futures.ProcessPoolExecutor() as executor:
        poses = executor.map(fast_idealize, [f"{directory}/{pdb}" for pdb in pdbs])
