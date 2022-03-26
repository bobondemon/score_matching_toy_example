import logging
import hydra
from omegaconf import DictConfig, OmegaConf

from scorematchingproj.train_flow.trainer import Trainer


@hydra.main(config_path="conf", config_name="train_config")
def train(cfg: DictConfig) -> None:
    dataloader = hydra.utils.instantiate(cfg.dataloader)
    model = hydra.utils.instantiate(cfg.model)

    trainer = Trainer(model, dataloader, **cfg.trainer_init_kwargs)

    # Train
    print("Starting training!")
    logging.info("Starting training!")
    trainer.learn(**cfg.trainer_learn_kwargs)


if __name__ == "__main__":
    train()
