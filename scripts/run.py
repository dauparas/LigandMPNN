import os.path

import ligandmpnn
from ligandmpnn import MPNN_designer

import hydra
from omegaconf import DictConfig, OmegaConf

config_dir = os.path.join(os.path.dirname(ligandmpnn.__file__), "config")


@hydra.main(config_path=config_dir, config_name="ligandmpnn", version_base=None)
def main(cfg: DictConfig) -> None:
    """
    Inference function
    """

    # instantializing design
    magician: MPNN_designer = MPNN_designer(cfg)

    if (mode := cfg.runtime.mode.use) == "design":
        print(f"Mode: {mode}")
        magician.design_proteins()

    else:
        print(f"Mode: {mode}")
        magician.score_proteins()


if __name__ == "__main__":
    main()
