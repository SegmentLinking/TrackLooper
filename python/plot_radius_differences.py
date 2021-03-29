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


def process_layers(layer_binary):
    layers = []
    for i in range(1,12):
        if ((layer_binary & 1 << i) >> i):
            layers.append(i)
    return np.array(layers)

def process_moduleTypes(moduleType_binary,objectType = "T4"):
    moduleTypes = []
    if objectType == "T3":
        layers = [0,2,6]
    elif objectType == "T4":
        layers = [0,2,4,6]
    elif objectType == "sg":
        layers = [0,2]
    elif objectType == "t5":
        layers = [0,2,4,6,8]

    for i in layers:
        moduleTypes.append((moduleType_binary & (1 << i)) >> i)
    return np.array(moduleTypes)

def process_layerType(layers):
    layerType = np.empty_like(layers, dtype = "str")
    layerType[layers <= 6] = "B"
    layerType[layers > 6] = "E"
    return "".join(layerType)


def process_numbers(layers):
    numbers = layers.astype(str)
    return "".join(numbers)

def make_plots(qArray, qArraySimTrackMatched, quantity, layerType):
    if len(qArray) == 0 or len(qArraySimTrackMatched) == 0:
        print("{} has no entries. Skipping".format(layerType))
        return
    minValue = min(qArray[qArray > -999])
    maxValue = max(qArray)
    histMinLimit = minValue * 1.1 if minValue < 0 else minValue * 0.9
    histMaxLimit = maxValue * 0.9 if maxValue < 0 else maxValue * 1.1
    if abs(histMaxLimit - histMinLimit) > 10 and histMinLimit > 0 or "/" in quantity:
        binning = np.logspace(np.log10(histMinLimit), np.log10(histMaxLimit), 1000)
    else:
        binning = np.linspace(histMinLimit, histMaxLimit, 1000)

    allHist = Hist1D(ak.to_numpy(qArray[qArray > -999]), bins=binning, label="{}".format(quantity))
    simtrackMatchedHist = Hist1D(ak.to_numpy(qArraySimTrackMatched[qArraySimTrackMatched > -999]), bins=binning, label="Sim track matched {}".format(quantity))

    plt.yscale("log")
    if abs(histMaxLimit - histMinLimit) > 10 and histMinLimit > 0 or "/" in quantity:
        plt.xscale("log")

    allHist.plot(alpha=0.8, color="C0", label="all", histtype="stepfilled")
    simtrackMatchedHist.plot(alpha=0.8, color="C3", label="sim track matched", histtype="stepfilled")
    if layerType == "":
        if "TripletPt" in quantity:
            title = quantity.replace("TripletPt", "Triplet radius")
        else:
            title = quantity
        plt.title("{}".format(title))
    else:
        plt.title("{} type {}".format(quantity, layerType))

    plt.suptitle("Sample = {} Tag = {}".format(sys.argv[3], sys.argv[4]))
    # extra job for the composite dudes
    quantity = quantity.replace("(", " ")
    quantity = quantity.replace(")", "")
    quantity = quantity.replace("/", "by")
    quantity = quantity.replace("-", "minus")
    quantity = quantity.replace(" ", "_")
    if len(sys.argv) > 2:
        if layerType != "":
            plt.savefig("{}/{}_{}.pdf".format(sys.argv[2], quantity, layerType))

        else:
            plt.savefig("{}/{}.pdf".format(sys.argv[2], quantity))
    else:
        if layerType != "":
            plt.savefig("{}_{}.pdf".format(quantity, layerType))
        else:
            plt.savefig("{}.pdf".format(quantity))
    plt.close()


def make_single_plots(qArray, quantity, layerType):
    minValue = min(qArray[qArray > -999])
    maxValue = max(qArray)
    histMinLimit = minValue * 1.1 if minValue < 0 else minValue * 0.9
    histMaxLimit = maxValue * 0.9 if maxValue < 0 else maxValue * 1.1
    if abs(histMaxLimit - histMinLimit) > 10 and histMinLimit > 0 or "/" in quantity:
        binning = np.logspace(np.log10(histMinLimit), np.log10(histMaxLimit), 1000)
    else:
        binning = np.linspace(histMinLimit, histMaxLimit, 100)

    allHist = Hist1D(ak.to_numpy(qArray[qArray > -999]), bins=binning, label="{}".format(quantity))
    plt.yscale("log")
    if abs(histMaxLimit - histMinLimit) > 10 and histMinLimit > 0 or "/" in quantity:
        plt.xscale("log")

    allHist.plot(alpha=0.8, color="C0", label="all", histtype="stepfilled")

    if layerType == "":
        title = quantity
        plt.title("{}".format(title))
    else:
        plt.title("{} type {}".format(quantity, layerType))
    plt.suptitle("Sample = {} Tag = {}".format(sys.argv[3], sys.argv[4]))
    # extra job for the composite dudes
    quantity = quantity.replace("(", " ")
    quantity = quantity.replace(")", "")
    quantity = quantity.replace("/", "by")
    quantity = quantity.replace("-", "minus")
    quantity = quantity.replace(" ", "_")
    if len(sys.argv) > 2:
        if layerType != "":
            plt.savefig("{}/{}_{}.pdf".format(sys.argv[2], quantity, layerType))

        else:
            plt.savefig("{}/{}.pdf".format(sys.argv[2], quantity))
    else:
        if layerType != "":
            plt.savefig("{}_{}.pdf".format(quantity, layerType))
        else:
            plt.savefig("{}.pdf".format(quantity))
    plt.close()


def make_radius_difference_distributions():
    global tree
    matchedMask = tree["t5_isFake"].array() == 0
    layers = np.array(list(map(process_layers, ak.flatten(tree["t5_layer_binary"].array()))))
    layerTypes = np.array(list(map(process_layerType, layers)))
#    layerTypes = np.array(list(map(process_numbers, layers)))
    unique_layerTypes = np.unique(layerTypes, axis=0)
    unique_layerTypes = np.append(unique_layerTypes,"")
    print(unique_layerTypes)

    for layerType in unique_layerTypes:
        print("layerType = {}".format(layerType))
        innerRadius = ak.to_numpy(ak.flatten(tree["t5_innerRadius"].array()))
        bridgeRadius = ak.to_numpy(ak.flatten(tree["t5_bridgeRadius"].array()))
        outerRadius = ak.to_numpy(ak.flatten(tree["t5_outerRadius"].array()))
        simRadius = ak.to_numpy(ak.flatten(tree["t5_matched_pt"].array()))/(2.99792458e-3 * 3.8)

        qArrayInner = abs(1.0/innerRadius - 1.0/simRadius)/(1.0/innerRadius)
        qArrayBridge = abs(1.0/bridgeRadius - 1.0/simRadius)/(1.0/bridgeRadius)
        qArrayOuter = abs(1.0/outerRadius - 1.0/simRadius)/(1.0/outerRadius)

        for name,qArray in {"inner":qArrayInner, "bridge":qArrayBridge, "outer":qArrayOuter}.items():
            if layerType == "":
                qArraySimTrackMatched = qArray[ak.to_numpy(ak.flatten(matchedMask))]
            else:
                qArray = qArray[layerTypes == layerType]
                qArraySimTrackMatched = qArray[ak.to_numpy(ak.flatten(matchedMask)[layerTypes == layerType])]

            print("{} integral = {}, {} sim-track integral = {}".format(name, len(qArray), name, len(qArraySimTrackMatched)))

            make_single_plots(qArraySimTrackMatched, "(1/{} - 1/sim_radius)/(1/{{})".format(name + " radius"), layerType)


objects = ["t5"]
for i in objects:
    make_radius_difference_distributions(i)
