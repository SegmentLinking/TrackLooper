import uproot
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import awkward as ak
import mplhep
import sys
from yahist import Hist1D
import os

plt.style.use(mplhep.style.CMS)

f = uproot.open(sys.argv[1])
tree = f["tree"]


def make_histograms(obj):
    global tree
    mdf = open("{}_summaries.md".format(obj),"w")
    mdf.write("% {} Occupancy summary\n".format(obj))
    mdf.write("### 100% of the distribution between lower and upper limits\n")  
    mdf.write("| Region | Lower limit &nbsp; &nbsp; | Upper limit &nbsp; &nbsp; |  99.9% Upper limit &nbsp; &nbsp; |Link|\n")
    mdf.write("| :---: | :---: | :---: | :---: | :---: |\n")
 
    # one big histogram, one split by subdets alone, one split by subdet and layer
    # and for endcap, one split by layer and ring

    branchesList = ["module_*", "{}_occupancies".format(obj)]
    branches = tree.arrays(filter_name=branchesList, library="pd")
    if type(branches) is tuple:
        branches, occupancies = branches
        branches.drop(index = 13296, level = 1, inplace = True)
        branches["{}_occupancies".format(obj)] = occupancies
    print(branches)

    print("dataframe obtained!")
    # create the slices
    view = branches.loc[branches["module_subdets"] != 0,"{}_occupancies".format(obj)]
    overall_hist = Hist1D(view.values, bins = np.linspace(view.min(), view.max(),min(500,view.max() - view.min() + 1)), label = "{} Occupancy".format(obj))

    subdetLayerHists = []
    subdetLayerRingHists = []
    for subdet in [0, 4, 5]:
        if subdet == 5:
            subdetName = "Barrel"
            layerRange = range(1,7)
        elif subdet == 4:
            subdetName = "Endcap"
            layerRange = range(1,6)
        else:
            subdetName = "Pixel"
            layerRange = [0]

        for layer in layerRange:
            subdetLayerView = branches.loc[(branches["module_subdets"] == subdet) & (branches["module_layers"] == layer),"{}_occupancies".format(obj)]
            if len(subdetLayerView) == 0:
                continue
            offset = 1
            if subdetLayerView.min() == subdetLayerView.max():
                offset = 2

            subdetLayerHists.append(Hist1D(subdetLayerView.values, bins = np.linspace(subdetLayerView.min(), subdetLayerView.max() + 1, min(500,subdetLayerView.max() - subdetLayerView.min() + offset)), label = "{} Occupancy in {} Layer {}".format(obj,subdetName,layer)))

            if subdet == 4:
                for ring in range(1,16):
                    subdetLayerRingView = branches.loc[(branches["module_subdets"] == subdet) & (branches["module_layers"] == layer) & (branches["module_rings"] == ring),"{}_occupancies".format(obj)]
                    if len(subdetLayerRingView) == 0:
                        continue
                    offset = 1
                    if subdetLayerRingView.min() == subdetLayerRingView.max():
                        offset = 2
                    subdetLayerRingHists.append(Hist1D(subdetLayerRingView.values, bins = np.linspace(subdetLayerRingView.min(), subdetLayerRingView.max() + 1, min(500,subdetLayerRingView.max() - subdetLayerRingView.min() + offset)), label = "{} Occupancy in {} Layer {} Ring {}".format(obj,subdetName,layer, ring)))

    plot_histograms(mdf,overall_hist)
    plot_histograms(mdf,subdetLayerHists)
    plot_histograms(mdf,subdetLayerRingHists)

    mdf.close()     

def plot_histograms(mdf,hists):
    if type(hists) is not list:
        hists = [hists]

    print(len(hists))
    for hist in hists:
        fig = plt.figure()
        plt.yscale("log")
        hist.plot(alpha = 0.8, color = "C0")
        plt.title(hist.metadata["label"])
        plt.suptitle("Sample = {} Tag = {}".format(sys.argv[3], sys.argv[4]))
        outputName = hist.metadata["label"].replace(" ","_")
        url_prefix = sys.argv[2].replace("/home/users/bsathian/public_html","http://uaf-10.t2.ucsd.edu/~bsathian")
        mdf.write("|{}|{:0.1f}|{:0.1f}|{:0.1f}|[plot]({})\n".format(hist.metadata["label"],hist.edges[0],hist.edges[-1],hist.quantile(0.999),"{}/{}.pdf".format(url_prefix,outputName)))
        os.system("mkdir -p {}".format(sys.argv[2]))
        plt.savefig("{}/{}.pdf".format(sys.argv[2],outputName))
        print(hist.metadata["label"])
        plt.close()

    
objects = ["md","sg","t3","t4","tc", "t5", "pT3"]
for obj in objects:
    make_histograms(obj)
