#!/bin/env python

if __name__ == "__main__":

    import matplotlib
    matplotlib.use('pdf')
    import matplotlib.pyplot as plt
    import SDLDisplay

    sdlDisplay = SDLDisplay.getDefaultSDLDisplay()

    fig, ax = plt.subplots(figsize=(6,6))
    sdlDisplay.display_detector_xy(ax)
    fig.savefig("detxy.pdf")

    fig, ax = plt.subplots(figsize=(10,4))
    sdlDisplay.display_detector_rz(ax)
    fig.savefig("detrz.pdf")

