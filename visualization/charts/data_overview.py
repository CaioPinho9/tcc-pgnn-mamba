from __future__ import annotations

from matplotlib import pyplot as plt

from config.config import OPENFWI_CONFIG


def plot_dataset_overview(
    seismic,
    velocity,
    *,
    physics_config=None,
    save_path=None,
    show=True,
):
    """Plota o resumo de três painéis de sísmica e velocidade usado na análise de dados.

        ``seismic`` tem formato ``[source, time, receiver]`` e ``velocity`` tem
        formato ``[depth, width]``. O argumento opcional ``save_path`` permite
        reutilizar a figura em notebooks, scripts e no gerador de apresentações.
    """
    physics_config = physics_config or OPENFWI_CONFIG
    fig, axes = plt.subplots(1, 3, figsize=(16, 4), constrained_layout=True)
    plots = [
        (seismic[0], "Aquisição sísmica - fonte 0", "gray", None, None),
        (seismic.mean(axis=0), "Média das aquisições sobre as fontes", "gray", None, None),
        (
            velocity,
            "Modelo de velocidade",
            "jet",
            physics_config["velocity_min"],
            physics_config["velocity_max"],
        ),
    ]
    for ax, (values, title, cmap, vmin, vmax) in zip(axes, plots):
        image = ax.imshow(values, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)
        ax.set_title(title)
        fig.colorbar(image, ax=ax, fraction=0.046)
    for ax in axes[:2]:
        ax.set_xlabel("Índice do receptor")
        ax.set_ylabel("Índice temporal")
    axes[2].set_xlabel("Posição x (célula)")
    axes[2].set_ylabel("Profundidade z (célula)")
    if save_path is not None:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
    if show:
        plt.show()
        return fig, axes
    plt.close(fig)
    return fig, axes
