from dataclasses import dataclass
import os
import random

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
import hydra

from D3Fold.data.openfold.raw_protein import make_pdb_features
from D3Fold.data.openfold.raw_protein import np_to_tensor_dict
from D3Fold.data.openfold import transforms
from D3Fold.data.openfold.raw_protein import RawProtein
from D3Fold.constants import CONFIG_PATH

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

      if self.type_.pair_type:
         self.data = self.data[mask]
         self.data = self.data[:, mask]
      else:
        self.data = self.data[mask]

    def __repr__(self):
        return str(self.data)

    def __str__(self):
        return str(self.data)

feature_config_path = os.path.join(CONFIG_PATH, "features")
@hydra.main(version_base=None, config_path=feature_config_path, config_name="config")
def build_types(cfg: DictConfig):
  types = {}
  features = cfg.features
  for feature in features:
    type = ProteinDataType(
      type=feature.type,
      pad_type=feature.pad_type,
      meta_data=feature.meta_data if "meta_data" in feature else False,
      pair_type=feature.pair_type if "pair_type" in feature else False
    )
    types[feature.name] = type

  print(types)


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
    type_dict: dict

    @classmethod
    def from_dict(cls, data_dict):
      type_dict = build_types()
      data_dict = {key: ProteinData(data_dict[key], type_dict[key]) for key in data_dict.keys()}
      return cls(**data_dict, type_dict=type_dict)

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