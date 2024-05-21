import os

import wandb
import torch
import lightning as L

class SaveTrainableParamsCallback(L.Callback):
    def __init__(self, artifact_name, artifact_type, ckpt_dir="./ckpts"):
        super().__init__()
        self.artifact_name = artifact_name
        self.artifact_type = artifact_type
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir, exist_ok=True)
        self.ckpt_dir = ckpt_dir

    def on_train_epoch_end(self, trainer, pl_module):
        # Create a wandb artifact
        artifact = wandb.Artifact(self.artifact_name, type=self.artifact_type)
        trainable_params = {}
        # Iterate over the model's named_parameters and save only the trainable ones
        for name, param in pl_module.named_parameters():
            if param.requires_grad:
                trainable_params[name] = param.data
        torch.save(trainable_params, f"{self.ckpt_dir}/{self.artifact_name}.pth")
        artifact.add_file(f"{self.ckpt_dir}/{self.artifact_name}.pth")
        wandb.log_artifact(artifact)

class SaveOptimizerState(L.CallBack):
    def __init__(self, artifact_name, artifact_type, ckpt_dir="./ckpts"):
        super().__init__()
        self.artifact_name = artifact_name
        self.artifact_type = artifact_type
        if not os.path.exists(ckpt_dir):
            os.makedirs(ckpt_dir, exist_ok=True)
        self.ckpt_dir = ckpt_dir
    
    def on_train_epoch_end(self, trainer, pl_module):
        # Create a wandb artifact
        artifact = wandb.Artifact(self.artifact_name, type=self.artifact_type)
        optimizer_state = trainer.optimizers.state_dict()
        torch.save(optimizer_state, f"{self.ckpt_dir}/{self.artifact_name}.pth")
        artifact.add_file(f"{self.ckpt_dir}/{self.artifact_name}.pth")
        wandb.log_artifact(artifact)