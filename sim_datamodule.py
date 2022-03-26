import pathlib
import hydra
from omegaconf import DictConfig
import matplotlib.pyplot as plt
import numpy as np

from scorematchingproj.helper.vis import plot_data


@hydra.main(config_path="conf", config_name="sim_datamodule")
def sim_and_plot(cfg: DictConfig) -> None:
    """
    An example usage of datamodule. Then plot the toy dataset
    """
    dataloader = hydra.utils.instantiate(cfg.dataloader)
    # print(dataloader)
    data = np.asarray(dataloader.get_data())
    print(data.shape)

    # https://matplotlib.org/stable/plot_types/stats/hist2d.html#sphx-glr-plot-types-stats-hist2d-py
    fig, ax = plt.subplots()
    plot_data(ax, data)
    plt.show()


if __name__ == "__main__":
    sim_and_plot()
