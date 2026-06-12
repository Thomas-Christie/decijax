from pathlib import Path

import matplotlib.pyplot as plt


def use_mpl_style():
    style_file = Path(__file__).parent / "gpjax.mplstyle"
    plt.style.use(style_file)
