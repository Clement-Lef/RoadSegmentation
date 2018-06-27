# -*- coding: utf-8 -*-

import argparse as argp
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import seaborn as sns


palette_c = sns.color_palette('muted')

SAVE_FIG = True


import os
import matplotlib as mpl

# reportdir = '/Volumes/Macintosh HD/Documents/EPFL/EPFL 2016-17/Comput-Sim-I/REPORT'
# basedir = reportdir
# basedir = os.path.join(os.getcwd(), '..')
basedir = '.'
figdir = os.path.join(basedir, 'figures')

globalfsize = 12

figsize = (8, 6)
linewidth = 2
fontsize = globalfsize
legendsize = globalfsize
labelsize = globalfsize + 2
labelsize_tex = 14
titlesize = globalfsize
ticksize = globalfsize + 2
fmt = 'pdf'

# LaTeX symbols shortcuts
mfH2 = r"$f_{\mathrm{H}_2}$"

mpl.rcParams['axes.labelpad'] = 5
mpl.rcParams['axes.labelsize'] = labelsize
mpl.rcParams['axes.linewidth'] = 1
mpl.rcParams['axes.titlesize'] = titlesize
mpl.rcParams['figure.figsize'] = figsize
# mpl.rcParams['font.family'] = 'serif'
# mpl.rcParams['font.serif']           = ['Palatino']
mpl.rcParams['font.size'] = fontsize
mpl.rcParams['grid.color'] = '#8A8A8A'
mpl.rcParams['grid.linewidth'] = 0.5
mpl.rcParams['legend.fontsize'] = legendsize
mpl.rcParams['legend.numpoints'] = 0
mpl.rcParams['legend.scatterpoints'] = 0
mpl.rcParams['legend.markerscale'] = 1
mpl.rcParams['lines.color'] = 'b'
mpl.rcParams['lines.linewidth'] = linewidth
mpl.rcParams['patch.linewidth'] = 1
mpl.rcParams['savefig.bbox'] = 'tight'
mpl.rcParams['savefig.dpi'] = 100
mpl.rcParams['savefig.format'] = fmt
mpl.rcParams['savefig.pad_inches'] = 0.05
mpl.rcParams['text.usetex'] = False  # buuuug
mpl.rcParams['xtick.labelsize'] = ticksize
mpl.rcParams['xtick.major.pad'] = 10
mpl.rcParams['xtick.major.size'] = 6
mpl.rcParams['xtick.major.width'] = 1
mpl.rcParams['xtick.minor.pad'] = 10
mpl.rcParams['xtick.minor.size'] = 4
mpl.rcParams['xtick.minor.width'] = 1
mpl.rcParams['ytick.labelsize'] = ticksize
mpl.rcParams['ytick.major.pad'] = 10
mpl.rcParams['ytick.major.size'] = 6
mpl.rcParams['ytick.major.width'] = 1
mpl.rcParams['ytick.minor.pad'] = 10
mpl.rcParams['ytick.minor.size'] = 4
mpl.rcParams['ytick.minor.width'] = 1

# FIXES FOR 2.0
# mpl.style.use('classic') # reset all back to 1.x version
#** math mode
mpl.rcParams['mathtext.fontset'] = 'cm'
mpl.rcParams['mathtext.rm'] = 'serif'
#** tick placements
mpl.rcParams['xtick.direction'] = 'in'
mpl.rcParams['ytick.direction'] = 'in'
mpl.rcParams['xtick.top'] = True
mpl.rcParams['ytick.right'] = True
#** legend styling
mpl.rcParams['legend.fancybox'] = False
mpl.rcParams['legend.loc'] = 'upper right'
mpl.rcParams['legend.numpoints'] = 2
mpl.rcParams['legend.fontsize'] = 'large'
mpl.rcParams['legend.framealpha'] = None
mpl.rcParams['legend.scatterpoints'] = 3
mpl.rcParams['legend.edgecolor'] = 'inherit'
#** figure display
# mpl.rcParams['figure.dpi'] = 80

#


def savefig(figobj, figsubdir, figname):
    savedir = os.path.join(figdir, figsubdir)
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    savepath = os.path.join(savedir, figname)
    print("Saving figure to {}.{} ...".format(savepath, fmt))
    figobj.savefig(savepath)
    print("Figure saved.")


def figsize(scale):
    fig_width_pt = 483.697                          # Get this from LaTeX using \the\textwidth
    inches_per_pt = 1.0 / 72.27                       # Convert pt to inch
    golden_mean = (np.sqrt(5.0) - 1.0) / 2.0            # Aesthetic ratio (you could change this)
    fig_width = fig_width_pt * inches_per_pt * scale    # width in inches
    fig_height = fig_width * golden_mean              # height in inches
    fig_size = [fig_width, fig_height]
    return fig_size



def compute_mean_score_cv(scores, folds=10):
    if not (scores.shape[0] % folds == 0):
        raise ValueError("Array size not divisible by number of CV folds.")
    scores_reshape = scores.reshape(-1, folds)
    scores_mean = np.mean(scores_reshape, axis=1)
    scores_std = np.std(scores_reshape, axis=1)
    return scores_mean, scores_std


def output_tf_aerial_fig1(showplot=True, savefig=False):
    """Function to output and save plots :
    - F1 score vs train size for various numbe of epochs
    """
    epochs_range = (1, 5, 10)
    training_size = (50, 100, 200, 400, 800, 1600)
    cv_folds = 10


    # --- num epochs plots --- #
    fig1 = plt.figure(figsize=(8, 5))
    gs1 = GridSpec(1, 1, width_ratios=[1, ],
                   height_ratios=[1, ])
    # gs.update(wspace=0.5, hspace=0.05)

    ax11 = plt.subplot(gs1[0])
    # ax12 = plt.subplot(gs1[1])
    for idx, num_epochs in enumerate(epochs_range):

        datafile = 'datas/output_cv{}_e{}.dat'.format(cv_folds, num_epochs)

        data = np.loadtxt(datafile)
        tr_f1_mean, tr_f1_std = compute_mean_score_cv(data[:, 7],
                                                      folds=cv_folds)
        va_f1_mean, va_f1_std = compute_mean_score_cv(data[:, 8],
                                                      folds=cv_folds)

        # print(tr_f1_mean, va_f1_mean)

        ax11.plot(training_size, tr_f1_mean,
                  label="{} epochs, training".format(num_epochs),
                  linewidth=3, markersize=8, marker='d', linestyle=':',
                  color=palette_c[idx])
        # ax11.errorbar(training_size, tr_f1_mean, yerr=tr_f1_std,
        #               capsize=5, capthick=1.5, elinewidth=1.5,
        #               color=palette_c[idx],
        #               linewidth=0, alpha=0.4, zorder=-10)
        ax11.plot(training_size, va_f1_mean,
                  label="{} epochs, validation".format(num_epochs),
                  linewidth=3, markersize=9, marker='o', linestyle='-',
                  color=palette_c[idx])
        # ax11.errorbar(training_size, va_f1_mean, yerr=va_f1_std,
        #               capsize=5, capthick=1.5, elinewidth=1.5,
        #               color=palette_c[idx],
        #               linewidth=0, alpha=0.4, zorder=-10)

    ax11.set_xlim(0, 1650)
    ax11.set_ylim(0.44, 0.7)
    ax11.set_ylabel("mean F1 score")
    ax11.set_xlabel("dataset size")
    ax11.legend(loc='lower right')

    if showplot:
        plt.show()

    # if savefig:
    #     dirname = ''
    #     filename1 = 'epochs_vs_dataset'
    #     savefig(fig1, dirname, filename1)


def output_tf_aerial_fig2(showplot=True, savefig=False):
    """Function to output and save plots :
    - F1 score vs train size for various type of image augmentation
    """
    augment_range = ('rot', 'rotzoom', 'rotzoomshear')
    augment_labels = ('rotation', 'r. + zoom', 'r. + z. + shear')
    training_size = (50, 100, 200, 400, 800, 1600)
    cv_folds = 10

    # --- data augmentation plots --- #
    fig2 = plt.figure(figsize=(8, 5))
    gs2 = GridSpec(1, 1, width_ratios=[1, ],
                   height_ratios=[1, ])
    # gs.update(wspace=0.5, hspace=0.05)

    ax21 = plt.subplot(gs2[0])
    # ax22 = plt.subplot(gs2[1])
    for idx, aug_type in enumerate(augment_range):

        datafile = 'datas/output_cv{}_e{}_{}.dat'.format(cv_folds, 5,
                                                         aug_type)

        data = np.loadtxt(datafile)
        tr_f1_mean, tr_f1_std = compute_mean_score_cv(data[:, 7],
                                                      folds=cv_folds)
        va_f1_mean, va_f1_std = compute_mean_score_cv(data[:, 8],
                                                      folds=cv_folds)

        # print(tr_f1_mean, va_f1_mean)

        ax21.plot(training_size[:-1], tr_f1_mean,
                  label="{}, training".format(augment_labels[idx]),
                  linewidth=3, markersize=8, marker='d', linestyle=':',
                  color=palette_c[idx])
        # ax21.errorbar(training_size[:-1], tr_f1_mean, yerr=tr_f1_std,
        #               capsize=5, capthick=1.5, elinewidth=1.5,
        #               color=palette_c[idx],
        #               linewidth=0, alpha=0.4, zorder=-10)
        ax21.plot(training_size[:-1], va_f1_mean,
                  label="{}, validation".format(augment_labels[idx]),
                  linewidth=3, markersize=9, marker='o', linestyle='-',
                  color=palette_c[idx])
        # ax21.errorbar(training_size[:-1], va_f1_mean, yerr=va_f1_std,
        #               capsize=5, capthick=1.5, elinewidth=1.5,
        #               color=palette_c[idx],
        #               linewidth=0, alpha=0.4, zorder=-10)

    # ax1.set_xscale('log', basex=2)
    # ax1.set_xlabel("training dataset size")
    ax21.set_xlim(0, 850)
    ax21.set_ylim(0.53, 0.69)
    ax21.set_xlabel("dataset size")
    ax21.set_ylabel("mean F1 score")
    ax21.legend(loc='lower right')

    if showplot:
        plt.show()

    # if savefig:
    #     dirname = ''
    #     filename2 = 'augment_vs_dataset'
    #     savefig(fig2, dirname, filename2)


def output_CNN1(showplot=True, savefig=False):

    datafile_new = 'datas/history.csv'
    data_new = np.genfromtxt(datafile_new, delimiter=',', skip_header=1)

    datafile_nft = 'datas/history_no_fine_tune.csv'
    data_nft = np.genfromtxt(datafile_nft, delimiter=',', skip_header=1)

    datafile_ft = 'datas/history_fine_tune.csv'
    data_ft = np.genfromtxt(datafile_ft, delimiter=',', skip_header=1)

    fig = plt.figure(figsize=(8, 3))
    gs = GridSpec(2, 2, width_ratios=[1, 1],
                  height_ratios=[0.05, 1])
    gs.update(wspace=0.05, hspace=0.05)
    ax1 = plt.subplot(gs[1, 0])
    ax2 = plt.subplot(gs[1, 1], sharey=ax1)

    ax1.plot(data_nft[:, 0] + 1, data_nft[:, 3],
             label="training",
             linewidth=2, markersize=8, marker='d', linestyle='-',
             color=palette_c[0])
    ax1.plot(data_nft[:, 0] + 1, data_nft[:, 5],
             label="validation",
             linewidth=2, markersize=8, marker='o', linestyle='-',
             color=palette_c[0])

    ax1.plot(data_new[:, 0] + 1, data_new[:, 3],
             label="training",
             linewidth=2, markersize=8, marker='d', linestyle='-',
             color=palette_c[2])
    ax1.plot(data_new[:, 0] + 1, data_new[:, 5],
             label="validation",
             linewidth=2, markersize=8, marker='o', linestyle='-',
             color=palette_c[2])

    # just for the legend to be black
    ax1.plot(data_nft[:, 0] + 1, data_nft[:, 3],
             label="training",
             linewidth=1.5, markersize=7, marker='d', linestyle='-',
             color='black', zorder=-10)
    ax1.plot(data_nft[:, 0] + 1, data_nft[:, 5],
             label="validation",
             linewidth=1.5, markersize=7, marker='o', linestyle='-',
             color='black', zorder=-10)
    # trick legend

    ax2.plot(data_ft[:, 0] + 1, data_ft[:, 3],
             label="training",
             linewidth=2, markersize=8, marker='d', linestyle='-',
             color=palette_c[1])
    ax2.plot(data_ft[:, 0] + 1, data_ft[:, 5],
             label="validation",
             linewidth=2, markersize=8, marker='o', linestyle='-',
             color=palette_c[1])

    plt.setp(ax2.get_yticklabels(), visible=False)

    ax1.set_xlim(0, data_nft[:, 0].max() + 2)
    ax2.set_xlim(0, data_ft[:, 0].max() + 2)
    ax1.set_ylim(0.85, 0.98)
    ax1.set_xlabel("number of epochs")
    ax2.set_xlabel("number of epochs")
    ax1.set_ylabel("micro F1 score")
    # ax1.legend(bbox_to_anchor=(0, 1.1, 1, 0.12), loc='upper center',
    #            borderaxespad=0)
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend([handles[4], handles[5]],
               [labels[4], labels[5]], bbox_to_anchor=(0., 0.99, 1., .102),
               loc='upper center', ncol=2, borderaxespad=0)
    # ax2.legend(bbox_to_anchor=(0, 1.1, 1, 0.12), loc='upper center',
    #            borderaxespad=0)

    if showplot:
        plt.show()

    # if savefig:
    #     dirname = ''
    #     filename2 = 'cnn1_history'
    #     savefig(fig, dirname, filename2)


def str2bool(v):
    if v.lower() in ('yes', 'y', '1'):
        return True
    elif v.lower() in ('no', 'n', '0'):
        return False
    else:
        raise argp.ArgumentTypeError('Boolean value expected.')


if __name__ == '__main__':

    # set arguments parser
    parser = argp.ArgumentParser(formatter_class=argp.RawTextHelpFormatter)
    help_mode = """select data to show :
 0 F1 score vs train size for various number of epochs (tf_aerial),
 1 F1 score vs train size for various type of image augmentation (tf_aerial),
 2 F1 score vs number of epochs (our custom CNN)"""
 #    help_save = """Save plot as pdf file. Accepted choices (default 'n') :
 # 'no', 'n', '0'
 # 'yes', y', '1'"""
 #    help_show = """Show plot. Accepted choices (default 'y') :
 # 'no', 'n', '0'
 # 'yes', 'y', '1'"""
    parser.add_argument('mode', metavar='mode', type=int, nargs='?',
                        help=help_mode)
    # parser.add_argument('-s', metavar='save', dest='savefig', type=str2bool,
    #                     nargs='?', default=False, help=help_save)
    # parser.add_argument('-d', metavar='display', dest='display', type=str2bool,
    #                     nargs='?', default=True, help=help_show)
    args = parser.parse_args()

    if args.mode == 0:
        output_tf_aerial_fig1()

    elif args.mode == 1:
        output_tf_aerial_fig2()

    elif args.mode == 2:
        output_CNN1()
