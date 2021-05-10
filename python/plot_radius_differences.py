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
    elif objectType == "t5" or objectType == "pT3":
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
    histMinLimit = 1.1 * minValue if minValue < 0 else 0.9 * minValue
    histMaxLimit = 1.1 * maxValue if maxValue > 0 else 0.9 * maxValue
    binning = np.linspace(histMinLimit, histMaxLimit, 1000)

    allHist = Hist1D(ak.to_numpy(qArray[qArray > -999]), bins=binning, label="{}".format(quantity))
    simtrackMatchedHist = Hist1D(ak.to_numpy(qArraySimTrackMatched[qArraySimTrackMatched > -999]), bins=binning, label="Sim track matched {}".format(quantity))
    fig, ax = plt.subplots()
    ax.set_yscale("log")
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
    all_pT3_arrays = tree.arrays(filter_name = "pT3*", entry_start = 0, entry_stop = -1, library = "ak")
    matchedMask = all_pT3_arrays.pT3_isFake == 0
    layers = np.array(list(map(process_layers, ak.flatten(all_pT3_arrays.pT3_layer_binary)))

    layerTypes = np.array(list(map(process_layerType, layers)))
#    layerTypes = np.array(list(map(process_numbers, layers)))

    unique_layerTypes = np.unique(layerTypes, axis=0)
    unique_layerTypes = np.append(unique_layerTypes,"")
    print(unique_layerTypes)

    for layerType in unique_layerTypes:
        print("layerType = {}".format(layerType))
        pixelRadius = ak.to_numpy(ak.flatten(all_pT3_arrays.pT3_pixelRadius))
        tripletRadius = ak.to_numpy(ak.flatten(all_pT3_arrays.pT3_tripletRadius))
        simRadius = ak.flatten(all_pT3_arrays.pT3_matched_pt/(2.99792458e-3 * 3.8))
        simRadius = ak.flatten(simRadius)

        qArrayInner = abs(1.0/pixelRadius - 1.0/simRadius)/(1.0/pixelRadius)
        qArrayOuter = abs(1.0/tripletRadius - 1.0/simRadius)/(1.0/tripletRadius)

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
    all_pT3_arrays = tree.arrays(filter_name = "pT3*", entry_start = 0, entry_stop = 5, library = "ak")
    matchedMask = all_pT3_arrays.pT3_isFake == 0
    layers = np.array(list(map(process_layers, ak.flatten(all_pT3_arrays.pT3_layer_binary))))
    layerTypes = np.array(list(map(process_layerType, layers)))
#    layerTypes = np.array(list(map(process_numbers, layers)))
    unique_layerTypes = np.unique(layerTypes, axis=0)
    unique_layerTypes = np.append(unique_layerTypes,"")
    print(unique_layerTypes)

    for layerType in unique_layerTypes:
        print("layerType = {}".format(layerType))

        pixelRadius = ak.to_numpy(ak.flatten(all_pT3_arrays.pT3_pixelRadius))
        pixelRadiusResMin = ak.to_numpy(ak.flatten(all_pT3_arrays.pT3_pixelRadiusMin))
        pixelRadiusResMax = ak.to_numpy(ak.flatten(all_pT3_arrays.pT3_pixelRadiusMax))

        tripletRadius = ak.to_numpy(ak.flatten(all_pT3_arrays.pT3_tripletRadius))
        tripletRadiusResMin = ak.to_numpy(ak.flatten(all_pT3_arrays.pT3_tripletRadiusMin))
        tripletRadiusResMax = ak.to_numpy(ak.flatten(all_pT3_arrays.t5_tripletRadiusMax))

        simRadius = ak.flatten(all_t5_arrays.t5_matched_pt/(2.99792458e-3 * 3.8))
        simRadius = ak.flatten(simRadius)

        pixelRadiusMin = ak.to_numpy(ak.min([pixelRadiusResMin, pixelRadius2SMin], axis = 0))
        pixelRadiusMax = ak.to_numpy(ak.max([pixelRadiusResMax, pixelRadius2SMax], axis = 0))
        tripletRadiusMin = ak.to_numpy(ak.min([tripletRadiusResMin, tripletRadius2SMin], axis = 0))
        tripletRadiusMax = ak.to_numpy(ak.max([tripletRadiusResMax, tripletRadius2SMax], axis = 0))

        qArrayInnerOuter = compute_interval_overlap(1.0/pixelRadiusMax, 1.0/pixelRadiusMin, 1.0/tripletRadiusMax, 1.0/tripletRadiusMin)


        for name,qArray in {"innerOuter":qArrayInnerOuter}.items():
            print("qName = ",name)
            if layerType == "":
                qArraySimTrackMatched = qArray[ak.to_numpy(ak.flatten(matchedMask))]
            else:
                qArray = qArray[layerTypes == layerType]
                qArraySimTrackMatched = qArray[ak.to_numpy(ak.flatten(matchedMask)[layerTypes == layerType])]
            print("{} total integral = {}, {} integral below zero = {}, sim-matched {} total integral = {}, sim-matched {} integral above zero = {}".format(name, len(qArray), name, sum(qArray < 0), name, len(qArraySimTrackMatched), name, sum(qArraySimTrackMatched > 0)))
            make_plots(qArray, qArraySimTrackMatched, "overlap between 1/{} and 1/{}".format("Inner", name[5:]), layerType)

objects = ["t5"]
for i in objects:
    make_radius_difference_distributions()
    #make_radius_compatibility_distributions()
