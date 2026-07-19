STATE_DIR = "state"
PLOTS_DIR = "plots"
SEED = 7
N_WORKERS = 8

ALL_EXPFAM_DISTS = [
    "poisson", "exponential", "gaussian", "gamma_shape",
    "binomial", "pascal", "gamma_rate",
]
 
DIST_HYPERPARAMS = {
    "poisson":     {},
    "exponential": {},
    "gaussian":    {"sigma2": 1.0},
    "gamma_shape": {"alpha": 2.0},
    "binomial":    {"n_trials": 5},
    "pascal":      {"r": 3},
    "gamma_rate":  {"beta": 1.0},
}

DIST_STYLE = {
    "poisson":     {"color": "#1F77B4", "marker": "o", "label": "Poisson"},
    "exponential": {"color": "#D62728", "marker": "s", "label": "Exponential"},
    "gaussian":    {"color": "#2CA02C", "marker": "^", "label": "Gaussian"},
    "gamma_shape": {"color": "#FF7F0E", "marker": "D", "label": r"Gamma ($\alpha$)"},
    "binomial":    {"color": "#9467BD", "marker": "v", "label": "Binomial"},
    "pascal":      {"color": "#17BECF", "marker": "P", "label": "Pascal"},
    "gamma_rate":  {"color": "#E377C2", "marker": "X", "label": r"Gamma ($\beta$)"},
}

DIST_MODES = {
    "all_poisson":     {"color": "#1F77B4", "marker": "o", "label": "All Poisson"},
    "all_exponential": {"color": "#D62728", "marker": "s", "label": "All Exponential"},
    "all_gaussian":    {"color": "#2CA02C", "marker": "^", "label": "All Gaussian"},
    "all_gamma_shape": {"color": "#FF7F0E", "marker": "D", "label": r"All Gamma ($\alpha$)"},
    "all_binomial":    {"color": "#9467BD", "marker": "v", "label": "All Binomial"},
    "all_pascal":      {"color": "#17BECF", "marker": "P", "label": "All Pascal"},
    "all_gamma_rate":  {"color": "#E377C2", "marker": "X", "label": r"All Gamma ($\beta$)"},
    "full_mix":        {"color": "#000000", "marker": "p", "label": "Mixed (all 7)"},
}

PLOT_LOG_FLOOR = 2e-5

def apply_plot_style():
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import rcParams
    rcParams.update({
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "Times New Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 16,
        "axes.labelsize": 18,
        "axes.titlesize": 19,
        "legend.fontsize": 13,
        "xtick.labelsize": 14,
        "ytick.labelsize": 14,
        "axes.linewidth": 1.2,
        "axes.edgecolor": "#1C2833",
        "axes.labelcolor": "#1C2833",
        "axes.labelweight": "bold",
        "xtick.color": "#1C2833",
        "ytick.color": "#1C2833",
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.major.size": 6,
        "ytick.major.size": 6,
        "xtick.major.width": 1.2,
        "ytick.major.width": 1.2,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "xtick.minor.size": 3.5,
        "ytick.minor.size": 3.5,
        "xtick.minor.width": 0.8,
        "ytick.minor.width": 0.8,
        "legend.framealpha": 0.95,
        "legend.edgecolor": "#7B7D7D",
        "legend.fancybox": False,
        "lines.solid_capstyle": "round",
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "savefig.facecolor": "white",
        "savefig.edgecolor": "none",
    })