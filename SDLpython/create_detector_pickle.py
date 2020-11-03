#!/bin/env python

if __name__ == "__main__":

    import matplotlib
    matplotlib.use('pdf')
    import matplotlib.pyplot as plt
    import SDLDisplay
    import pickle
    from matplotlib.ticker import AutoMinorLocator

    sdlDisplay = SDLDisplay.getDefaultSDLDisplay()

    fig, ax = plt.subplots(figsize=(6,6))
    sdlDisplay.display_detector_xy(ax)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    fig.savefig("detxy.pdf")

    pickle.dump(fig, file('figxy.pickle', 'w'))
    pickle.dump(ax, file('detxy.pickle', 'w'))

    fig, ax = plt.subplots(figsize=(10,4))
    sdlDisplay.display_detector_rz(ax)
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    fig.savefig("detrz.pdf")

    pickle.dump(fig, file('figrz.pickle', 'w'))
    pickle.dump(ax, file('detrz.pickle', 'w'))

