import uproot
import numpy as np
import matplotlib.pyplot as plt
import awkward as ak
import mplhep
import sys
from yahist import Hist1D
from matplotlib.ticker import LogLocator, NullFormatter
plt.style.use(mplhep.style.CMS)

f = uproot.open(sys.argv[1])
tree = f["tree"]


def parseLayers(layer_binary):
    layers = []
    for i in range(12):
        if layer_binary & (1<<i):
            layers.append(str(i))
        if layers[0] == '0':
            layers = ['0'] + layers
    return layers

def make_single_plots(qArray, quantity, layerOverlap, hitOverlap, layer_binary):
    minValue = min(qArray[qArray > -999])
    maxValue = max(qArray)
    histMinLimit = 9e-6
    histMaxLimit = 1.1e2
    if abs(histMaxLimit - histMinLimit) > 10 and histMinLimit > 0 or "/" in quantity:
        binning = np.logspace(np.log10(histMinLimit), np.log10(histMaxLimit), 500)
    else:
        binning = np.linspace(histMinLimit, histMaxLimit, 500)

    allHist = Hist1D(ak.to_numpy(qArray[qArray > -999]), bins=binning, label="{}".format(quantity))
    fig, ax = plt.subplots()
    ax.set_yscale("log")
    if abs(histMaxLimit - histMinLimit) > 10 and histMinLimit > 0 or "/" in quantity:
        x_major = LogLocator(base=10.0, numticks=5)
        x_minor = LogLocator(base=10.0, subs=np.arange(1.0, 10.0) * 0.1, numticks=10)
        ax.xaxis.set_major_locator(x_major)
        ax.xaxis.set_minor_locator(x_minor)
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.set_xscale("log")

    #find the 99%

    print("99% limit at {}".format(allHist.quantile(0.99)))

    allHist.plot(alpha=0.8, color="C0", label="all", histtype="stepfilled")

    layers = parseLayers(layer_binary)
    plt.title("{}, layer overlap = {}, hit overlap = {}".format((",").join(layers), layerOverlap, hitOverlap))
    plt.xlabel(quantity)

    plt.suptitle("Sample = {}".format(sys.argv[3]))

    if "inner" in quantity:
        plt.savefig("{}/inner_v_sim_layerOverlap_{}_hitOverlap_{}_{}.pdf".format(sys.argv[2], layerOverlap, hitOverlap, ("").join(layers)))
    else:
        plt.savefig("{}/outer_v_sim_layerOverlap_{}_hitOverlap_{}_{}.pdf".format(sys.argv[2], layerOverlap, hitOverlap, ("").join(layers)))

    plt.close()


def make_radius_difference_distributions():
    global tree

    dfs = tree.arrays(filter_name = "T3T3_*", library = "ak")


    unique_layers = np.unique(ak.flatten(dfs["T3T3_layer_binary"]).to_numpy())

    innerRadiusVariable = "T3T3_innerT3Radius"
    outerRadiusVariable = "T3T3_outerT3Radius"

    for layer_binary in unique_layers:
        for layerOverlap in range(3):
            for hitOverlap in range(5):
                tempSelection = (dfs.T3T3_layer_binary == layer_binary) & (dfs.T3T3_nLayerOverlaps == layerOverlap) & (dfs.T3T3_nHitOverlaps == hitOverlap) & (dfs.T3T3_isFake == 0)

                if "highE" in sys.argv[3]:
                    tempSelection = tempSelection & (dfs["T3T3_matched_pt"] > 2.0)

                innerRadius = dfs.T3T3_innerRadius[tempSelection]
                outerRadius = dfs.T3T3_outerRadius[tempSelection]
                simRadius = dfs["T3T3_matched_pt"]/(2.99792458e-3 * 3.8)

                qArrayInner = abs(1.0/innerRadius - 1.0/simRadius)/(1.0/innerRadius)
                qArrayOuter = abs(1.0/outerRadius - 1.0/simRadius)/(1.0/outerRadius)

                if len(ak.flatten(qArrayInner)) == 0 or len(ak.flatten(qArrayOuter)) == 0:
                    continue

                make_single_plots(qArrayInner, "T3T3 inner sim fraction difference", layerOverlap, hitOverlap, layer_binary)

def compute_interval_overlap(firstMin, firstMax, secondMin, secondMax):
    intervalLength = np.zeros_like(firstMin)
    intervalLength[firstMin < secondMin] = secondMin[firstMin < secondMin] - firstMax[firstMin < secondMin]
    intervalLength[secondMin < firstMin] = firstMin[secondMin < firstMin] - secondMax[secondMin < firstMin]
    return intervalLength

make_radius_difference_distributions("T3T3")
