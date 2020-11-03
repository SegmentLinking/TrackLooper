#!/bin/env python

if __name__ == "__main__":

    import matplotlib
    matplotlib.use('pdf')
    import matplotlib.pyplot as plt
    import pickle

    ax_xy = pickle.load(file('/nfs-7/userdata/phchang/detector_layout_matplotlib_pickle/detxy.pickle'))
    plt.savefig("detxy.pdf")

    ax_rz = pickle.load(file('/nfs-7/userdata/phchang/detector_layout_matplotlib_pickle/detrz.pickle'))
    plt.savefig("detrz.pdf")
