"""Host plotting primitives; importing this module does not import matplotlib."""
import os

PAPER_STYLE = {
    'figure.dpi': 120, 'savefig.dpi': 300, 'savefig.bbox': 'tight',
    'font.size': 10, 'font.family': 'serif', 'mathtext.fontset': 'dejavuserif',
    'axes.titlesize': 11, 'axes.titleweight': 'bold', 'axes.labelsize': 10,
    'axes.grid': True, 'axes.axisbelow': True, 'grid.alpha': .3, 'grid.linewidth': .5,
    'axes.spines.top': False, 'axes.spines.right': False, 'lines.linewidth': 1.7,
    'legend.frameon': False, 'legend.fontsize': 8.5,
}


def apply_paper_style(overrides=None):
    import matplotlib.pyplot as plt
    plt.rcParams.update({**PAPER_STYLE, **(overrides or {})})


def save_figure(fig, out_dir, stem):
    import matplotlib.pyplot as plt
    png = os.path.join(out_dir, stem + '.png')
    pdf = os.path.join(out_dir, stem + '.pdf')
    fig.savefig(png)
    fig.savefig(pdf)
    plt.close(fig)
    return png, pdf
