from dataclasses import dataclass

import random
import numpy as np
import torch

from D3Fold.data.openfold.raw_protein import make_pdb_features
from D3Fold.data.openfold.raw_protein import np_to_tensor_dict
from D3Fold.data.openfold import transforms
from D3Fold.data.openfold.raw_protein import RawProtein

class ProteinDataType:
    def __init__(
          self,
          type=None,
          pad_type="torch_geometric",
          mask_template=None,
          meta_data=False,
          pair_type=False,
          ):
        self.type = type
        self.mask_template = mask_template
        self.pad_type = pad_type
        self.meta_data = meta_data
        self.pair_type = pair_type


    def __repr__(self):
        return self.type

    def __str__(self):
        return self.type


@dataclass
class ProteinData():
    data: torch.Tensor
    type_: ProteinDataType
    masked_data: torch.Tensor = None

    def mask_data(self, mask):
        if self.type_.meta_data or self.type_.mask_template is None:
          return

        if type(self.data) is np.ndarray:
          mask = mask.numpy()
        self.data[mask] = self.type_.mask_template

    def crop_data(self, mask, crop_len):
      if self.type_.meta_data:
        return

      if type(self.data) is np.ndarray:
        mask = mask.numpy()

      self.data = self.data[mask]
      if self.type_.pair_type:
         self.data = self.data[mask, mask]

    def __repr__(self):
        return str(self.data)

    def __str__(self):
        return str(self.data)

AA_TYPE = ProteinDataType("AA_TYPE", pad_type="seq", mask_template=None)
RES_INDEX = ProteinDataType("RES_INDEX", pad_type="seq", mask_template=None)
SEQ_LENGTH = ProteinDataType("SEQ_LENGTH", pad_type="seq", mask_template=None, meta_data=True)
AA_POSITIONS = ProteinDataType("AA_POSITIONS", pad_type="seq", mask_template=None)
ALL_ATOM_MASK = ProteinDataType("ALL_ATOM_MASK", pad_type="seq", mask_template=None)
RESOLUTION = ProteinDataType("RESOLUTION", pad_type="seq", mask_template=None, meta_data=True)
IS_DISTILLATION = ProteinDataType("IS_DISTILLATION", pad_type="seq", mask_template=None, meta_data=True)
ATOM14_EXISTS = ProteinDataType("ATOM_14_EXISTS", pad_type="seq", mask_template=None)
RESIDX_ATOM14_TO_ATOM37 = ProteinDataType("RESIDX_ATOM14_TO_ATOM37", pad_type="seq", mask_template=None)
RESIDX_ATOM37_TO_ATOM14 = ProteinDataType("RESIDX_ATOM37_TO_ATOM14", pad_type="seq", mask_template=None)
ATOM37_ATOM_EXISTS = ProteinDataType("ATOM37_ATOM_EXISTS", pad_type="seq", mask_template=None)
ATOM14_GT_EXISTS = ProteinDataType("ATOM14_GT_EXISTS", pad_type="seq", mask_template=None)
ATOM14_GT_POSITIONS = ProteinDataType("ATOM14_GT_POSITIONS", pad_type="seq", mask_template=None)
ATOM14_ALT_GT_POSITIONS = ProteinDataType("ATOM14_ALT_GT_POSITIONS", pad_type="seq", mask_template=None)
ATOM14_ALT_GT_EXISTS = ProteinDataType("ATOM14_ALT_GT_EXISTS", pad_type="seq", mask_template=None)
ATOM14_ATOM_IS_AMBIGUOUS = ProteinDataType("ATOM14_ATOM_IS_AMBIGUOUS", pad_type="seq", mask_template=None)
RIGIDGROUPS_GT_FRAMES = ProteinDataType("RIGIDGROUPS_GT_FRAMES", pad_type="seq", mask_template=None)
RIGIDGROUPS_GT_EXISTS = ProteinDataType("RIGIDGROUPS_GT_EXISTS", pad_type="seq", mask_template=None)
RIGIDGROUPS_GROUP_EXISTS = ProteinDataType("RIGIDGROUPS_GROUP_EXISTS", pad_type="seq", mask_template=None)
RIGIDGROUPS_GROUP_IS_AMBIGUOUS = ProteinDataType("RIGIDGROUPS_GROUP_IS_AMBIGUOUS", pad_type="seq", mask_template=None)
RIGIDGROUPS_ALT_GT_FRAMES = ProteinDataType("RIGIDGROUPS_ALT_GT_FRAMES", pad_type="seq", mask_template=None)
TORSION_ANGLES_SIN_COS = ProteinDataType("TORSION_ANGLES_SIN_COS", pad_type="seq", mask_template=None)
ALT_TORSION_ANGLES_SIN_COS = ProteinDataType("ALT_TORSION_ANGLES_SIN_COS", pad_type="seq", mask_template=None)
TORSION_ANGLES_MASK = ProteinDataType("TORSION_ANGLES_MASK", pad_type="seq", mask_template=None)
PSEUDO_BETA = ProteinDataType("PSEUDO_BETA", pad_type="seq", mask_template=None)
PSEUDO_BETA_MASK = ProteinDataType("PSEUDO_BETA_MASK", pad_type="seq", mask_template=None)
BACKBONE_RIGID_TENSOR = ProteinDataType("BACKBONE_RIGID_TENSOR", pad_type="seq", mask_template=None)
BACKBONE_RIGID_MASK = ProteinDataType("BACKBONE_RIGID_MASK", pad_type="seq", mask_template=None)
CHI_ANGLES_SIN_COS = ProteinDataType("CHI_ANGLES_SIN_COS", pad_type="seq", mask_template=None)
CHI_MASK = ProteinDataType("CHI_MASK", pad_type="seq", mask_template=None)
raw_mask = np.array(["<mask>"])
RAW_SEQ = ProteinDataType("RAW_SEQ", pad_type="esm", mask_template=raw_mask)
DIST_MAT = ProteinDataType("DIST_MAT", pad_type="seq", mask_template=None)

@dataclass
class TorchProtein:
    aatype: ProteinData
    residue_index: ProteinData
    all_atom_positions: ProteinData
    all_atom_mask: ProteinData
    resolution: ProteinData
    is_distillation: ProteinData
    atom14_atom_exists: ProteinData
    residx_atom14_to_atom37: ProteinData
    residx_atom37_to_atom14: ProteinData
    atom37_atom_exists: ProteinData
    atom14_gt_exists: ProteinData
    atom14_gt_positions: ProteinData
    atom14_alt_gt_positions: ProteinData
    atom14_alt_gt_exists: ProteinData
    rigidgroups_gt_frames: ProteinData
    rigidgroups_gt_exists: ProteinData
    rigidgroups_group_exists: ProteinData
    rigidgroups_gt_group_is_ambiguous: ProteinData
    rigidgroups_alt_gt_frames: ProteinData
    torsion_angles_sin_cos: ProteinData
    alt_torsion_angles_sin_cos: ProteinData
    torsion_angles_mask: ProteinData
    pseudo_beta: ProteinData
    pseudo_beta_mask: ProteinData
    backbone_rigid_tensor: ProteinData
    backbone_rigid_mask: ProteinData
    chi_angles_sin_cos: ProteinData
    chi_mask: ProteinData
    raw_seq: ProteinData
    distance_mat_stack: ProteinData


    @classmethod
    def from_dict(cls, data_dict):
      return cls(
        aatype=ProteinData(data_dict["aatype"], AA_TYPE),
        residue_index=ProteinData(data_dict["residue_index"], RES_INDEX),
        all_atom_positions=ProteinData(data_dict["all_atom_positions"], AA_POSITIONS),
        all_atom_mask=ProteinData(data_dict["all_atom_mask"], ALL_ATOM_MASK),
        resolution=ProteinData(data_dict["resolution"], RESOLUTION),
        is_distillation=ProteinData(data_dict["is_distillation"], IS_DISTILLATION),
        atom14_atom_exists=ProteinData(data_dict["atom14_atom_exists"], ATOM14_EXISTS),
        residx_atom14_to_atom37=ProteinData(data_dict["residx_atom14_to_atom37"], RESIDX_ATOM14_TO_ATOM37),
        residx_atom37_to_atom14=ProteinData(data_dict["residx_atom37_to_atom14"], RESIDX_ATOM37_TO_ATOM14),
        atom37_atom_exists=ProteinData(data_dict["atom37_atom_exists"], ATOM37_ATOM_EXISTS),
        atom14_gt_exists=ProteinData(data_dict["atom14_gt_exists"], ATOM14_GT_EXISTS),
        atom14_gt_positions=ProteinData(data_dict["atom14_gt_positions"], ATOM14_GT_POSITIONS),
        atom14_alt_gt_positions=ProteinData(data_dict["atom14_alt_gt_positions"], ATOM14_ALT_GT_POSITIONS),
        atom14_alt_gt_exists=ProteinData(data_dict["atom14_alt_gt_exists"], ATOM14_ALT_GT_EXISTS),
        rigidgroups_gt_frames=ProteinData(data_dict["rigidgroups_gt_frames"], RIGIDGROUPS_GT_FRAMES),
        rigidgroups_gt_exists=ProteinData(data_dict["rigidgroups_gt_exists"], RIGIDGROUPS_GT_EXISTS),
        rigidgroups_group_exists=ProteinData(data_dict["rigidgroups_group_exists"], RIGIDGROUPS_GROUP_EXISTS),
        rigidgroups_gt_group_is_ambiguous=ProteinData(data_dict["rigidgroups_group_is_ambiguous"], RIGIDGROUPS_GROUP_IS_AMBIGUOUS),
        rigidgroups_alt_gt_frames=ProteinData(data_dict["rigidgroups_alt_gt_frames"], RIGIDGROUPS_ALT_GT_FRAMES),
        torsion_angles_sin_cos=ProteinData(data_dict["torsion_angles_sin_cos"], TORSION_ANGLES_SIN_COS),
        alt_torsion_angles_sin_cos=ProteinData(data_dict["alt_torsion_angles_sin_cos"], ALT_TORSION_ANGLES_SIN_COS),
        torsion_angles_mask=ProteinData(data_dict["torsion_angles_mask"], TORSION_ANGLES_MASK),
        pseudo_beta=ProteinData(data_dict["pseudo_beta"], PSEUDO_BETA),
        pseudo_beta_mask=ProteinData(data_dict["pseudo_beta_mask"], PSEUDO_BETA_MASK),
        backbone_rigid_tensor=ProteinData(data_dict["backbone_rigid_tensor"], BACKBONE_RIGID_TENSOR),
        backbone_rigid_mask=ProteinData(data_dict["backbone_rigid_mask"], BACKBONE_RIGID_MASK),
        chi_angles_sin_cos=ProteinData(data_dict["chi_angles_sin_cos"], CHI_ANGLES_SIN_COS),
        chi_mask=ProteinData(data_dict["chi_mask"], CHI_MASK),
        raw_seq=ProteinData(data_dict["sequence"], RAW_SEQ),
        distance_mat_stack=ProteinData(data_dict["distance_mat_stack"], DIST_MAT)
      )

    def mask_fields(self, ignore_mask_fields=(), mask_percent=0.15):
      num_samples = len(self.aatype.data) * mask_percent
      num_samples = int(round(num_samples))
      indices = torch.arange(len(self.aatype.data))
      perm = torch.randperm(indices.size(0))
      mask = perm[:num_samples]
      for field in self.__dataclass_fields__.keys():
        if field in ignore_mask_fields:
            continue
        field_data = getattr(self, field)
        field_data.mask_data(mask)

      return mask

    def random_crop_mask(self, crop_len=400):
      start = random.randint(0, len(self.aatype.data) - crop_len)
      end = start + crop_len
      mask = torch.zeros(len(self.aatype.data))
      mask[start:end] = 1
      return mask.bool()

    def center_crop_mask(self, crop_len=400):
        start = len(self.aatype.data) // 2 - crop_len // 2
        end = start + crop_len
        mask = torch.zeros(len(self.aatype.data))
        mask[start:end] = 1
        return mask.bool()

    def crop_fields(self, ignore_mask_fields=(), crop_strategy="mix", crop_len=400):
      crop_fns = {
        "random": self.random_crop_mask,
        "center": self.center_crop_mask
      }
      if len(self.aatype.data) < crop_len:
        return
      elif crop_strategy == "random":
        mask = crop_fns[crop_strategy](crop_len)
      elif crop_strategy == "center":
        mask = crop_fns[crop_strategy](crop_len)
      elif crop_strategy == "mix":
        # randomly select strategy
        crop_strategy = random.choice(list(crop_fns.keys()))
        mask = crop_fns[crop_strategy](crop_len)
      else:
        raise ValueError("Invalid crop strategy")

      for field in self.__dataclass_fields__.keys():
        if field in ignore_mask_fields:
            continue
        field_data = getattr(self, field)
        field_data.crop_data(mask, crop_len)

    @classmethod
    def from_pdb(cls, pdb_file):

      protein = RawProtein.from_pdb_path(pdb_file)
      feats = make_pdb_features(protein, "no desc", is_distillation=False)
      tensor_dic = np_to_tensor_dict(feats, feats.keys())
      tensor_dic = transforms.squeeze_features(tensor_dic)
      tensor_dic = transforms.make_atom14_masks(tensor_dic)
      tensor_dic = transforms.make_atom14_positions(tensor_dic)
      tensor_dic = transforms.atom37_to_frames(tensor_dic)
      tensor_dic = transforms.atom37_to_torsion_angles(tensor_dic)
      tensor_dic = transforms.make_pseudo_beta(tensor_dic)
      tensor_dic = transforms.get_backbone_frames(tensor_dic)
      tensor_dic = transforms.get_chi_angles(tensor_dic)
      tensor_dic = transforms.get_distance_mat_stack(tensor_dic)
      return cls.from_dict(tensor_dic)