import uproot
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import awkward as ak
import mplhep
import sys
from yahist import Hist1D

plt.style.use(mplhep.style.CMS)

f = uproot.open(sys.argv[1])
tree = f["tree"]


def make_histograms(obj):
    global tree

    # one big histogram, one split by subdets alone, one split by subdet and layer
    # and for endcap, one split by layer and ring

    branchesList = ["module_*", "{}_occupancies".format(obj)]
    branches = tree.arrays(filter_name=branchesList, library="pd")

    # create the slices
    view = branches["{}_occupancies".format(obj)]
    overall_hist = Hist1D(view.values, bins = np.linspace(view.min(), view.max(),min(500,view.max() - view.min() + 1)), label = "{} Occupancy".format(obj))

    subdetLayerHists = []
    subdetLayerRingHists = []
    for subdet in [4, 5]:
        subdetName = "Barrel" if subdet == 5 else "Endcap"
        layerRange = range(1, 7) if subdet == 5 else range(1, 6)
        for layer in layerRange:
            subdetLayerView = branches.loc[(branches["module_subdets"] == subdet) & (branches["module_layers"] == layer),"{}_occupancies".format(obj)]
            if len(subdetLayerView) == 0:
                continue
            offset = 1
            if subdetLayerView.min() == subdetLayerView.max():
                offset = 2

            subdetLayerHists.append(Hist1D(subdetLayerView.values, bins = np.linspace(subdetLayerView.min(), subdetLayerView.max(), min(500,subdetLayerView.max() - subdetLayerView.min() + offset)), label = "{} Occupancy in {} Layer {}".format(obj,subdetName,layer)))

            if subdet == 4:
                for ring in range(1,16):
                    subdetLayerRingView = branches.loc[(branches["module_subdets"] == subdet) & (branches["module_layers"] == layer) & (branches["module_rings"] == ring),"{}_occupancies".format(obj)]
                    if len(subdetLayerRingView) == 0:
                        continue
                    offset = 1
                    if subdetLayerRingView.min() == subdetLayerRingView.max():
                        offset = 2
                    subdetLayerRingHists.append(Hist1D(subdetLayerRingView.values, bins = np.linspace(subdetLayerRingView.min(), subdetLayerRingView.max(), min(500,subdetLayerRingView.max() - subdetLayerRingView.min() + offset)), label = "{} Occupancy in {} Layer {} Ring {}".format(obj,subdetName,layer, ring)))

        plot_histograms(overall_hist)
        plot_histograms(subdetLayerHists)
        plot_histograms(subdetLayerRingHists)


def plot_histograms(hists):
    if type(hists) is not list:
        hists = [hists]

    for hist in hists:
        fig = plt.figure()
        plt.yscale("log")
        hist.plot(alpha = 0.8, color = "C0")
        plt.title(hist.metadata["label"])
        plt.suptitle("Sample = {} Tag = {}".format(sys.argv[3], sys.argv[4]))
        outputName = hist.metadata["label"].replace(" ","_")
        plt.savefig("{}/{}.pdf".format(sys.argv[2],outputName))
        plt.close()


objects = ["md","sg","t3","t4","tc"]
for obj in objects:
    make_histograms(obj)
