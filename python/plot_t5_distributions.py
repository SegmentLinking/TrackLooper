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
        binning = np.linspace(histMinLimit, histMaxLimit, 100)

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





def plot_distributions(obj):
    global tree
    blacklist = ["hitIdx","simTrkIdx","layer","pt","eta","phi","sim_pt","sim_eta","sim_phi","type", "ring", "moduleType_binary","layer_binary","isFake","isDuplicate"]
    print("object = ",obj)
    quantities = []
    for name in tree.keys():
        if name[:len(obj)] == obj and name not in map(lambda x : "{}_{}".format(obj,x),blacklist):
            quantities.append(name)
    matchedMask = tree["{}_isFake".format(obj)].array() == 0

    layers = np.array(list(map(process_layers,ak.flatten(tree["{}_layer_binary".format(obj)].array()))))
    #moduleTypes = np.array(list(map(process_moduleTypes,ak.flatten(tree["{}_moduleType_binary".format(obj)].array()))))
    layerTypes = np.array(list(map(process_layerType,layers)))
#    layerTypes = np.array(list(map(process_numbers, layers)))
#    print(layerTypes)
    unique_layerTypes = np.unique(layerTypes, axis = 0)
    unique_layerTypes = np.append(unique_layerTypes,"")
    print(unique_layerTypes)
    #Generic
    for layerType in unique_layerTypes:
        print("layerType = {}".format(layerType))
        for quantity in quantities:
            print("quantity = {}".format(quantity))
            if layerType == "":
                qArray = ak.flatten(tree[quantity].array())
                qArraySimTrackMatched = qArray[ak.flatten(matchedMask)]
            else:
                qArray = ak.flatten(tree[quantity].array())[layerTypes == layerType]
                qArraySimTrackMatched = qArray[ak.flatten(matchedMask)[layerTypes == layerType]]


            if all(qArray == -999):
                continue
            make_plots(qArray,qArraySimTrackMatched,quantity,layerType)


def make_composite_distributions():
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
        innerRadiusMin = ak.to_numpy(ak.flatten(tree["t5_innerRadiusMin"].array()))
        innerRadiusMax = ak.to_numpy(ak.flatten(tree["t5_innerRadiusMax"].array()))
        outerRadius = ak.to_numpy(ak.flatten(tree["t5_outerRadius"].array()))
        outerRadiusMin = ak.to_numpy(ak.flatten(tree["t5_outerRadiusMin"].array()))
        outerRadiusMax = ak.to_numpy(ak.flatten(tree["t5_outerRadiusMax"].array()))

        qArray = (outerRadiusMin - innerRadiusMax) / innerRadiusMax
        qArray[innerRadius > outerRadius] = (innerRadiusMin[innerRadius > outerRadius] - outerRadiusMax[innerRadius > outerRadius])/ innerRadiusMin[innerRadius > outerRadius]

        qArrayInv = (1.0/innerRadiusMax - 1.0/outerRadiusMin) / (1.0/innerRadiusMax)
        qArrayInv[innerRadius > outerRadius] = (1.0/ outerRadiusMax[innerRadius > outerRadius] - 1.0/innerRadiusMin[innerRadius > outerRadius])/(1.0/innerRadiusMin[innerRadius > outerRadius])

        if layerType == "":
            qArraySimTrackMatched = qArray[ak.to_numpy(ak.flatten(matchedMask))]
            qArrayInvSimTrackMatched = qArrayInv[ak.to_numpy(ak.flatten(matchedMask))]
        else:
            qArray = qArray[layerTypes == layerType]
            qArrayInv = qArrayInv[layerTypes == layerType]
            qArraySimTrackMatched = qArray[ak.to_numpy(ak.flatten(matchedMask)[layerTypes == layerType])]
            qArrayInvSimTrackMatched = qArrayInv[ak.to_numpy(ak.flatten(matchedMask)[layerTypes == layerType])]

        print("deltaR integral = ", len(qArray), "deltaR sim track matched integral = ", len(qArraySimTrackMatched))
        print("deltaR integral above 0 = ", sum(qArray >= 0), "deltaR sim track matched integral above 0 = ", sum(qArraySimTrackMatched >= 0))

        print("deltaInvR integral = ", len(qArrayInv), "deltaInvR sim track matched integral = ", len(qArrayInvSimTrackMatched))
        print("deltaInvR integral above 0 = ", sum(qArray >= 0), "deltaInvR sim track matched integral above 0 = ", sum(qArrayInvSimTrackMatched >= 0))

        make_plots(abs(qArray[qArray > 0]), abs(qArraySimTrackMatched[qArraySimTrackMatched > 0]), "deltaR/innerRadius",layerType)

        make_plots(abs(qArrayInv[qArrayInv > 0]), abs(qArrayInvSimTrackMatched[qArrayInvSimTrackMatched > 0]), "delta(1/R)/1/innerRadius", layerType)


#objects = ["t4","t3","pT4"]
objects = ["t5"]
for i in objects:
    make_composite_distributions()
    plot_distributions(i)
