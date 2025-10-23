import shutil
import os
import torch

from pylognet.client import LoggingClient, LogLevel

from audioldm_train.utilities.data.dataset import AudioDataset
from audioldm_train.utilities.model_util import instantiate_from_config

from torch.utils.data import DataLoader
from pytorch_lightning import seed_everything


class AudioLDMController:
    def __init__(
        self,
        checkpoint_path: str,
        config: dict,
        logger: LoggingClient,
        target_arch: str = "cuda",
    ):
        self.__config = config
        self.__logger = logger
        self.__addons = []
        self.__save_path = ""

        if "seed" in self.__config.keys():
            seed_everything(self.__config["seed"])
        else:
            seed_everything(0)

        if "precision" in self.__config.keys():
            torch.set_float32_matmul_precision(self.__config["precision"])

        if "dataloader_add_ons" in self.__config["data"].keys():
            self.__addons = self.__config["data"]["dataloader_add_ons"]

        self.__gscale = self.__config["model"]["params"]["evaluation_params"][
            "unconditional_guidance_scale"
        ]
        self.__ddim_steps = self.__config["model"]["params"]["evaluation_params"][
            "ddim_sampling_steps"
        ]
        self.__nc_per_sample = self.__config["model"]["params"]["evaluation_params"][
            "n_candidates_per_samples"
        ]

        if "reload_from_ckpt" in self.__config.keys():
            reload_from_ckpt = self.__config["reload_from_ckpt"]
        else:
            reload_from_ckpt = None

        resume_from_checkpoint = ""
        if os.path.exists(checkpoint_path):
            resume_from_checkpoint = checkpoint_path
            self.__logger.log(
                f"Load and resume checkpoint from {checkpoint_path}",
                LogLevel.INFO,
            )
        elif reload_from_ckpt is not None:
            resume_from_checkpoint = reload_from_ckpt
            self.__logger.log(
                f"Resume checkpoint from {checkpoint_path}",
                LogLevel.INFO,
            )
        else:
            raise ValueError(
                f"Cannot find checkpoint at {checkpoint_path}, and reload_from_ckpt is None."
            )
        checkpoint = torch.load(resume_from_checkpoint)

        self.__model = instantiate_from_config(self.__config["model"])
        if self.__model is None:
            raise ValueError("Model is None after instantiation.")

        self.__model.load_state_dict(checkpoint["state_dict"])
        self.__model.eval()

        if target_arch == "cuda":
            self.__model.cuda()
        else:
            self.__model.cpu()

    def set_savepath(self, path: str):
        if self.__model is None:
            return

        self.__save_path = path
        self.__model.set_log_dir(path, "", "")

    def generate_audio(self, uuid: str, prompt: str):
        if self.__model is None:
            raise ValueError("Model is None, cannot generate audio.")

        dataset = AudioDataset(
            self.__config,
            split="test",
            add_ons=self.__addons,
            dataset_json={
                "data": [
                    {
                        "wav": f"{uuid}.wav",
                        "caption": f"Sound of eating {prompt}",
                    }
                ]
            },
        )
        loader = DataLoader(
            dataset,
            batch_size=1,
        )

        self.__model.generate_sample(
            loader,
            unconditional_guidance_scale=self.__gscale,
            ddim_steps=self.__ddim_steps,
            n_gen=self.__nc_per_sample,
        )

        for dir in os.listdir(self.__save_path):
            for file in os.listdir(f"{self.__save_path}/{dir}"):
                shutil.move(f"{self.__save_path}/{dir}/{file}", f"{self.__save_path}")
                shutil.rmtree(f"{self.__save_path}/{dir}")
