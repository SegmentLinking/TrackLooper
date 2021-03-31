import uproot
import numpy as np
import matplotlib.pyplot as plt
import awkward as ak
import mplhep
import sys
from yahist import Hist1D
from matplotlib.ticker import LogLocator, NullFormatter

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
    histMinLimit = 1e-5
    histMaxLimit = 1e2
    if abs(histMaxLimit - histMinLimit) > 10 and histMinLimit > 0 or "/" in quantity:
        binning = np.logspace(np.log10(histMinLimit), np.log10(histMaxLimit), 1000)
    else:
        binning = np.linspace(histMinLimit, histMaxLimit, 1000)

    allHist = Hist1D(ak.to_numpy(qArray[qArray > -999]), bins=binning, label="{}".format(quantity))
    simtrackMatchedHist = Hist1D(ak.to_numpy(qArraySimTrackMatched[qArraySimTrackMatched > -999]), bins=binning, label="Sim track matched {}".format(quantity))

    ax.set_yscale("log")
    if abs(histMaxLimit - histMinLimit) > 10 and histMinLimit > 0 or "/" in quantity:
        plt.set_xscale("log")

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
    if quantity[0] == "_":
        quantity = quantity[1:]
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
    if quantity[0] == "_":
        quantity = quantity[1:]
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
    all_t5_arrays = tree.arrays(filter_name = "t5*", entry_start = 0, entry_stop = -1, library = "ak")
    matchedMask = all_t5_arrays.t5_isFake == 0
    layers = np.array(list(map(process_layers, ak.flatten(all_t5_arrays.t5_layer_binary))))
    layerTypes = np.array(list(map(process_layerType, layers)))
#    layerTypes = np.array(list(map(process_numbers, layers)))
    unique_layerTypes = np.unique(layerTypes, axis=0)
    unique_layerTypes = np.append(unique_layerTypes,"")
    print(unique_layerTypes)

    for layerType in unique_layerTypes:
        print("layerType = {}".format(layerType))
        innerRadius = ak.to_numpy(ak.flatten(all_t5_arrays.t5_innerRadius))
        bridgeRadius = ak.to_numpy(ak.flatten(all_t5_arrays.t5_bridgeRadius))
        outerRadius = ak.to_numpy(ak.flatten(all_t5_arrays.t5_outerRadius))
        simRadius = ak.flatten(all_t5_arrays.t5_matched_pt/(2.99792458e-3 * 3.8))
        simRadius = ak.flatten(simRadius)

        qArrayInner = abs(1.0/innerRadius - 1.0/simRadius)/(1.0/innerRadius)
        qArrayBridge = abs(1.0/bridgeRadius - 1.0/simRadius)/(1.0/bridgeRadius)
        qArrayOuter = abs(1.0/outerRadius - 1.0/simRadius)/(1.0/outerRadius)

        for name,qArray in {"inner": qArrayInner, "bridge": qArrayBridge, "outer": qArrayOuter}.items():
            print("qName = ",name)
            if layerType == "":
                qArraySimTrackMatched = qArray[ak.to_numpy(ak.flatten(matchedMask))]
            else:
                qArray = qArray[layerTypes == layerType]
                qArraySimTrackMatched = qArray[ak.to_numpy(ak.flatten(matchedMask)[layerTypes == layerType])]

            #print("{} integral = {}, {} sim-track integral = {}".format(name, len(qArray), name, len(qArraySimTrackMatched)))

            make_single_plots(qArraySimTrackMatched, "(1/{} - 1/sim_radius)/(1/{})".format(name + " radius", name + " radius"), layerType)


def compute_interval_overlap(firstMin, firstMax, secondMin, secondMax):
    intervalLength = np.zeros_like(firstMin)
    intervalLength[firstMin < secondMin] = secondMin[firstMin < secondMin] - firstMax[firstMin < secondMin]
    intervalLength[secondMin < firstMin] = firstMin[secondMin < firstMin] - secondMax[secondMin < firstMin]
    return intervalLength

def make_radius_compatibility_distributions():
    global tree
    all_t5_arrays = tree.array(filter_name = "t5*", entry_start = 0, entry_stop = -1, library = "ak")
    matchedMask = all_t5_arrays.t5_isFake == 0
    layers = np.array(list(map(process_layers, ak.flatten(all_t5_arrays.t5_layer_binary))))
    layerTypes = np.array(list(map(process_layerType, layers)))
#    layerTypes = np.array(list(map(process_numbers, layers)))
    unique_layerTypes = np.unique(layerTypes, axis=0)
    unique_layerTypes = np.append(unique_layerTypes,"")
    print(unique_layerTypes)

    for layerType in unique_layerTypes:
        print("layerType = {}".format(layerType))

        innerRadius = ak.to_numpy(ak.flatten(all_t5_arrays.t5_innerRadius))
        innerRadiusResMin = ak.to_numpy(ak.flatten(all_t5_arrays.t5_innerRadiusMin))
        innerRadius2SMin = ak.to_numpy(ak.flatten(all_t5_ararys.t5_innerRadiusMin2S))
        innerRadiusResMax = ak.to_numpy(ak.flatten(all_t5_arrays.t5_innerRadiusMax))
        innerRadius2SMax = ak.to_numpy(ak.flatten(all_t5_ararys.t5_innerRadiusMax2S))

        bridgeRadius = ak.to_numpy(ak.flatten(all_t5_arrays.t5_bridgeRadius))
        bridgeRadiusResMin = ak.to_numpy(ak.flatten(all_t5_arrays.t5_bridgeRadiusMin))
        bridgeRadius2SMin = ak.to_numpy(ak.flatten(all_t5_ararys.t5_bridgeRadiusMin2S))
        bridgeRadiusResMax = ak.to_numpy(ak.flatten(all_t5_arrays.t5_bridgeRadiusMax))
        bridgeRadius2SMax = ak.to_numpy(ak.flatten(all_t5_ararys.t5_bridgeRadiusMax2S))

        outerRadius = ak.to_numpy(ak.flatten(all_t5_arrays.t5_outerRadius))
        outerRadiusResMin = ak.to_numpy(ak.flatten(all_t5_arrays.t5_outerRadiusMin))
        outerRadius2SMin = ak.to_numpy(ak.flatten(all_t5_ararys.t5_outerRadiusMin2S))
        outerRadiusResMax = ak.to_numpy(ak.flatten(all_t5_arrays.t5_outerRadiusMax))
        outerRadius2SMax = ak.to_numpy(ak.flatten(all_t5_ararys.t5_outerRadiusMax2S))

        simRadius = ak.flatten(all_t5_arrays.t5_matched_pt/(2.99792458e-3 * 3.8))
        simRadius = ak.flatten(simRadius)

        innerRadiusMin = ak.to_numpy(ak.min([innerRadiusResMin, innerRadius2SMin], axis = 0))
        innerRadiusMax = ak.to_numpy(ak.max([innerRadiusResMax, innerRadius2SMax], axis = 0))
        bridgeRadiusMin = ak.to_numpy(ak.min([bridgeRadiusResMin, bridgeRadius2SMin], axis = 0))
        bridgeRadiusMax = ak.to_numpy(ak.max([bridgeRadiusResMax, bridgeRadius2SMax], axis = 0))
        outerRadiusMin = ak.to_numpy(ak.min([outerRadiusResMin, outerRadius2SMin], axis = 0))
        outerRadiusMax = ak.to_numpy(ak.max([outerRadiusResMax, outerRadius2SMax], axis = 0))

        qArrayInnerBridge = compute_interval_overlap(1.0/innerRadiusMax, 1.0/innerRadiusMin, 1.0/bridgeRadiusMax, 1.0/bridgeRadiusMin)
        qArrayInnerOuter = compute_interval_overlap(1.0/innerRadiusMax, 1.0/innerRadiusMin, 1.0/outerRadiusMax, 1.0/outerRadiusMin)


        for name,qArray in {"innerBridge":qArrayInnerBridge, "innerOuter":qArrayInnerOuter}:
            print("qName = ",name)
            if layerType == "":
                qArraySimTrackMatched = qArray[ak.to_numpy(ak.flatten(matchedMask))]
            else:
                qArray = qArray[layerTypes == layerType]
                qArraySimTrackMatched = qArray[ak.to_numpy(ak.flatten(matchedMask)[layerTypes == layerType])]

            make_plots(qArray, qArraySimTrackMatched, "overlap between 1/{} and 1/{}".format("Inner", name[5:]), layerType)

objects = ["t5"]
for i in objects:
    make_radius_difference_distributions()
    make_radius_compatibility_distributions()
