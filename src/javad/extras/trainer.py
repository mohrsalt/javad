import numpy as np
import torch
from torchvision.transforms import v2
from torch.optim.adam import Adam
from torch.utils.data import DataLoader
from torch.amp.grad_scaler import GradScaler
from spgdataset import SpectrogramDataset
from tqdm import tqdm
from types import SimpleNamespace
import os
import random
import logging
from javad import MODELINFO, from_pretrained, load_checkpoint
from typing import Union

##################################################
# Checkpointing
##################################################


class Checkpointer:
    def __init__(
        self,
        run_name: str,
        model_name: str,
        model,
        optimizer,
        scaler,
        dir: str = ".",
        period: int = 1,
    ) -> None:
        """
        Class to manage saving checkpoints during training.

        Args:
            run_name (str): The name of the training run.
            model_name (str): The name of the model.
            model: The model to be trained.
            optimizer: The optimizer to be used for training.
            scaler: The scaler for gradient scaling.
            dir (str, optional): The directory to save checkpoints. Defaults to ".".
            period (int, optional): The period (in epochs) to save checkpoints. Defaults to 1.

        Raises:
            Warning: If existing checkpoints for the given run_name are detected in the specified directory.
        """
        self.run_name = run_name
        self.model_name = model_name
        self.checkpoints_dir = dir
        self.period = period
        self.model = model
        self.optimizer = optimizer
        self.scaler = scaler
        if (
            len([f for f in os.listdir(self.checkpoints_dir) if f.startswith(run_name)])
            > 0
        ):
            logging.warning(
                f"[!] Detected existing checkpoints for '{run_name}' in folder '{self.checkpoints_dir}'"
            )

    def update(self, epoch):
        """
        Updates the model checkpoint at specified epochs.

        Args:
            epoch (int): The current epoch number.

        Returns:
            dict: A dictionary containing the model state, optimizer state, scaler state, and epoch number if a checkpoint is saved.
            None: If the checkpoint is not saved due to the epoch not matching the specified period.
        The function saves the model state, optimizer state, and scaler state (if available) to a checkpoint file
        in the specified directory. The checkpoint is saved only if the current epoch is a multiple of the specified period.
        """
        if epoch % self.period != 0:
            return
        output = {
            "model_name": self.model_name,
            "state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": (
                self.scaler.state_dict() if self.scaler is not None else None
            ),
            "epoch": epoch,
        }
        if not os.path.isdir(self.checkpoints_dir):
            os.makedirs(self.checkpoints_dir, exist_ok=True)
        cpt_name = self.run_name + f"_{epoch:02}.chpt"
        target_file = os.path.join(self.checkpoints_dir, cpt_name)
        with open(target_file, "wb") as f:
            torch.save(output, f)
            logging.info(f"Checkpoint {target_file} saved")
        return output


############################################
# Augmentation
############################################


class ZeroSilence(torch.nn.Module):
    """
    Class to zero out the input tensor where the mask is zero.
    Created for the cases with short audio clips or initial steps of training,
    when data is small than the window size.
    """

    def __init__(self) -> None:
        """
        Initializes the instance of the class.
        This constructor calls the parent class's constructor to ensure proper initialization.
        """

        super().__init__()

    def forward(self, inputs: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
        """
        Perform a forward pass by zeroing out the input tensor where the mask is zero.
        Args:
            inputs (torch.Tensor): The input tensor of shape (N, C, H, W).
            masks (torch.Tensor): The mask tensor of shape (N, W).
        Returns:
            torch.Tensor: The masked input tensor of the same shape as `inputs`.
        """
        inputs[:, :, :, :] = inputs[:, :, :, :] * masks[:, None, None, :]
        return inputs


class Augmentor:
    """
    Class to apply augmentations to the input tensor and mask.
    """

    def __init__(
        self,
        mix_chance: float = 0.5,
        erase_chance: float = 0.5,
        zero_chance: float = 0.1,
    ) -> None:
        """
        Initialize the training configuration with specified chances for various augmentations.
        Args:
            mix_chance (float): Probability of applying mix augmentation. Default is 0.5.
            erase_chance (float): Probability of applying erase augmentation. Default is 0.5.
            zero_chance (float): Probability of applying zero augmentation. Default is 0.1.
        """
        self.chances = SimpleNamespace(
            mix=mix_chance,
            erase=erase_chance,
            zero=zero_chance,
        )
        self.cutmix = v2.CutMix()
        self.mixup = v2.MixUp()
        self.erase = v2.RandomErasing(p=1.0)
        self.zero = ZeroSilence()

    def __call__(
        self,
        inputs: torch.Tensor,
        masks: torch.Tensor,
    ):
        assert len(inputs.shape) == 4, "Inputs should have shape (bs,n_ch,h,w)"
        assert len(masks.shape) == 2, "Masks should have shape (bs,w)"
        # Zeroing
        if random.random() < self.chances.zero:
            inputs = self.zero(inputs, masks)

        do_mix = random.random() < self.chances.mix
        do_erase = random.random() < self.chances.erase

        if do_mix or do_erase:
            # Create fake labels
            fake_labels = torch.nn.functional.one_hot(torch.arange(inputs.shape[0]))
            # Broadcast and concatenate masks with inputs along channels
            # This way we can apply transformations to both inputs and masks
            cim = torch.cat(
                (inputs, masks.unsqueeze(1).unsqueeze(1).broadcast_to(inputs.shape)),
                dim=1,
            )
        else:
            return inputs, masks

        if do_mix:
            if random.random() < 0.5:
                # CutMix
                cim, _ = self.cutmix(cim, fake_labels)
            else:
                # MixUp
                cim, _ = self.mixup(cim, fake_labels)
        if do_erase:
            # RandomErasing
            cim = self.erase(cim)
        return cim[:, :-1, :, :], cim[:, -1:, :, :].squeeze().mean(dim=-2)


############################################
# Collate
############################################


class CollateFN:
    """
    Collate function to process the output of the dataset and apply augmentations.
    Since the dataset returns a list of dictionaries, this function requires to extract and stack the inputs and masks.
    In addition, it applies the augmentations to the inputs and masks.
    """

    def __init__(
        self,
        augmentor: Union[Augmentor, None] = None,
    ):
        self.augm = augmentor

    def __call__(self, data):
        inputs = torch.stack([d["spectrogram"] for d in data]).unsqueeze(1)
        masks = torch.stack([d["masks"]["speech"] for d in data])
        if self.augm is not None:
            inputs, masks = self.augm(inputs, masks)
        return inputs, masks


############################################
# Trainer
############################################


class Trainer(torch.nn.Module):
    """
    Trainer class for re-training/fine-tuning JaVAD models.

    Example:
        .. code-block:: python

            trainer = Trainer(
                run_name="balanced_test",
                dataset_settings={
                    "audio_root": "path-to-dir-with-audio",
                    "spectrograms_root": "path-to-dir-to-save-generated-spectrograms-aka-cache",
                    "index_root": "path-to-dir-to-save-index-files-of-the-dataset",
                    "metadata_json_path": "path-to-metadata-json-file",
                    "max_memory_cache": 16000,  # allow to use up to 16Gb of RAM to retain spectrograms
                },
                use_mixed_precision=True,
                use_scheduler=True,
                window_min_content_ratio=0.5,
                window_offset_sec=0.5,
                device=torch.device("cuda:0"),
                learning_rate=1e-4,
                num_workers=2,
                total_epochs=20,
                augmentations={
                    "mix_chance": 0.0,
                    "erase_chance": 0.0,
                    "zero_chance": 0.00,
                },
            )
            trainer.train()
    """

    # fmt: off
    def __init__(
        self,
        run_name: str,                               # name of the run, affects checkpoint names
        model_name: str = "balanced",                # model name to use from javad, e.g. "balanced", "tiny", "precise"
        checkpoint: Union[str, None] = None,               # path to checkpoint to resume training from
        learning_rate: float = 1e-4,                 # learning rate for the optimizer
        use_scheduler: bool = False,                 # use cosine annealing scheduler
        total_epochs: int = 20,                        # number of epochs to train
        freeze_backbone: bool = False,               # freeze backbone layers
        use_mixed_precision: bool = False,           # use mixed precision training (for CUDA)
        dataset_settings: dict = {                   # dataset configuration
            "audio_root": "path_to_dataset_audio_dir",                     # path to audio files
            "spectrograms_root": "path_to_spectrograms_cache_dir_to_save", # path to save spectrograms
            "index_root": "path_to_index_dir_to_save",                     # path to save dataset index
            "metadata_json_path": "path_to_metadata.json",                 # path to metadata json file to load information about audio files (see docs)
            "max_memory_cache": 0,                                         # maximum memory cache to retain spectrograms in, in MB
        },
        augmentations: dict = {                      # augmentations to use
            "mix_chance": 0.0,                          # chance of applying cutmix/mixup
            "erase_chance": 0.0,                        # chance of applying random erasing
            "zero_chance": 0.00,                        # chance of zeroing inputs where mask is zero (should be small value, like 0.01)
        },
        device: torch.device = torch.device("cpu"),  # torch.device to use for training
        window_offset_sec: Union[float, None] = None,      # window offset in seconds
        window_min_content_ratio: float = 0.5,       # minimum content ratio in the window (e.g. speech to overall length)
        batch_size: int = 32,                        # batch size
        num_workers: int = 0,                        # number of workers for DataLoader
    ) -> None:
    # fmt: on
        """
        Trainer class for training a neural network model using PyTorch.

        Args:
            run_name (str): Name of the run, affects checkpoint names.
            model_name (str, optional): Model name to use at the beginning of the training e.g. epoch 0, e.g. "balanced", "tiny", "precise". Defaults to "balanced".
            checkpoint (Union[str, None], optional): Path to checkpoint file to resume training from. Defaults to None.
            learning_rate (float, optional): Learning rate for the optimizer. Defaults to 1e-4.
            use_scheduler (bool, optional): Use cosine annealing scheduler. Defaults to False.
            total_epochs (int, optional): Number of epochs to train. Defaults to 20.
            freeze_backbone (bool, optional): Freeze backbone layers. Defaults to False.
            use_mixed_precision (bool, optional): Use mixed precision training (for CUDA). Defaults to False.
            dataset_settings (dict, optional): Dataset configuration. Defaults to a predefined dictionary. Dict consists of:
                audio_root (str): Path to audio files.
                spectrograms_root (str): Path to save spectrograms.
                index_root (str): Path to save dataset index.
                metadata_json_path (str): Path to metadata json file to load information about audio files.
                max_memory_cache (int): Maximum memory cache to retain spectrograms in, in MB.
            augmentations (dict, optional): Augmentations to use. Defaults to a predefined dictionary.
                mix_chance (float): Chance of applying cutmix/mixup. Defaults to 0.0.
                erase_chance (float): Chance of applying random erasing. Defaults to 0.0.
                zero_chance (float): Chance of zeroing inputs where mask is zero. Defaults to 0.00. Expected to be very small value, around 0.01
            device (torch.device, optional): Torch device to use for training. Defaults to torch.device("cpu").
            window_offset_sec (Union[float, None], optional): Window offset in seconds. Defaults to None. If None, defaults to half of the window size.
            window_min_content_ratio (float, optional): Minimum content ratio in the window (e.g. speech to overall length). Defaults to 0.5.
            batch_size (int, optional): Batch size. Defaults to 32.
            num_workers (int, optional): Number of workers for DataLoader. Defaults to 0.

        Methods:
            train(): Trains the model for the specified number of epochs. Saves the model weights on completion.
            save_model_weights(path: str): Saves the model weights to the specified path.
        """
        super().__init__()
        self.run_name = run_name
        self.model_name = model_name
        self.device = device
        # Initialize modules
        if checkpoint is not None:
            logging.info(f"Resuming training using checkpoint {checkpoint}")
            cpt = load_checkpoint(checkpoint, is_asset=False)
            self.model_name = cpt['model_name']
        self.model = from_pretrained(self.model_name).float().to(device)
        self.optimizer = Adam(self.model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer, T_max=total_epochs, eta_min=1e-7
            ) if use_scheduler is True else None
        self.scaler = GradScaler("cuda") if use_mixed_precision is True else None
        self.checkpointer = Checkpointer(
            run_name=run_name,
            model_name=self.model_name,
            model=self.model,
            optimizer=self.optimizer,
            scaler=self.scaler,
        )
        self.criterion = torch.nn.MSELoss()
        # Initialize base parameters
        self.current_epoch = 0
        self.total_epochs = total_epochs
        self.__modelinfo = MODELINFO[self.model_name]
        # Initialize dataset, collate function for processing 
        # dataset output and augmentations and dataloader
        self.dataset = SpectrogramDataset(
            audio_root=dataset_settings["audio_root"],
            spectrograms_root=dataset_settings["spectrograms_root"],
            index_root=dataset_settings["index_root"],
            metadata_json_path=dataset_settings["metadata_json_path"],
            window_content_ratio=window_min_content_ratio,
            window_size_sec=self.__modelinfo["input_length"],
            window_offset_sec=window_offset_sec
            or (self.__modelinfo["input_length"] / 2),
            n_mels=self.__modelinfo["n_mels"],
            max_memory_cache=dataset_settings["max_memory_cache"],
            output_configuration={
                "audio": False,
                "spectrogram": True,
                "masks": ["speech"],
                "meta": [],
                "label": None,
            },
        )
        self.collate_fn = CollateFN(augmentor=Augmentor(**augmentations))
        self.dataloader = DataLoader(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            collate_fn=self.collate_fn,
        )

        # Load resume checkpoint if provided
        if checkpoint is not None:
            self.model.load_state_dict(cpt["state_dict"])
            self.optimizer.load_state_dict(cpt["optimizer_state_dict"])
            if use_mixed_precision is True:
                self.scaler.load_state_dict(cpt["scaler_state_dict"])
            self.scheduler.last_epoch = self.current_epoch = cpt["epoch"]

        # Freeze backbone if requested
        if freeze_backbone is True:
            logging.info("Freezing all layers except classification head")
            for param in self.model.parameters():
                param.requires_grad = False
            for param in self.model.collapsing_weights.parameters():
                param.requires_grad = True
            for param in self.model.classification_head.parameters():
                param.requires_grad = True

        # Initialize other parameters
        self.model.train()
        self.dataset.train()

    def train(self):
        for epoch in range(self.current_epoch, self.total_epochs):
            losses = []

            for inputs, masks in (
                pbar := tqdm(self.dataloader, desc=f"Epoch {epoch:02d}/{self.total_epochs:02d}, training")
            ):
                inputs = inputs.to(self.device)
                masks = masks.to(self.device)
                self.optimizer.zero_grad()
                if self.scaler is not None:
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        y_pred = self.model(inputs)
                        loss = self.criterion(y_pred, masks)
                    self.scaler.scale(loss).backward()
                    self.scaler.step(self.optimizer)
                    self.scaler.update()
                else:
                    y_pred = self.model(inputs)
                    loss = self.criterion(y_pred, masks)
                    loss.backward()
                    self.optimizer.step()
                losses = losses[-1000:] + [loss.item()]
                pbar.set_postfix_str(f"loss: {np.mean(losses):02.8f}")
            self.checkpointer.update(epoch)

            print(f"Learning rate: {self.optimizer.param_groups[0]['lr']:.8f}")
            if self.scheduler is not None:
                self.scheduler.step()
        # save model weights on completion
        self.save_model_weights(f"{self.run_name}.pt")

    def save_model_weights(self, path: str):
        torch.save({
            'model_name':  self.model_name,
            'state_dict': self.model.state_dict()}, path)
        logging.info(f"Model saved to {path}")
