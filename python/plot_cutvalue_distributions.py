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

    for i in layers:
        moduleTypes.append((moduleType_binary & (1 << i)) >> i)
    return np.array(moduleTypes)

def process_layerType(layers):
    layerType = np.empty_like(layers, dtype = "str")
    layerType[layers <= 6] = "B"
    layerType[layers > 6] = "E"
    return "".join(layerType)


def make_plots(qArray,qArraySimTrackMatched,quantity,layerType):
    minValue = min(qArray[qArray > -999])
    maxValue = max(qArray)
    histMinLimit = minValue * 1.1 if minValue < 0 else minValue * 0.9
    histMaxLimit = maxValue * 0.9 if maxValue < 0 else maxValue * 1.1
    if abs(histMaxLimit - histMinLimit) > 10 and histMinLimit > 0 or "/" in quantity:
        binning = np.logspace(np.log10(histMinLimit),np.log10(histMaxLimit),1000)
    else:
        binning = np.linspace(histMinLimit,histMaxLimit,1000)

    allHist = Hist1D(ak.to_numpy(qArray[qArray > -999]),bins = binning,label = "{}".format(quantity))
    simtrackMatchedHist = Hist1D(ak.to_numpy(qArraySimTrackMatched[qArraySimTrackMatched > -999]),bins = binning, label = "Sim track matched {}".format(quantity))

    fig = plt.figure()
    plt.yscale("log")
    if abs(histMaxLimit - histMinLimit) > 10 and histMinLimit > 0 or "/" in quantity:
        plt.xscale("log")

    allHist.plot(alpha = 0.8, color = "C0", label = "all")
    simtrackMatchedHist.plot(alpha = 0.8, color = "C3", label = "sim track matched")
    if layerType == "":
        plt.title("{}".format(quantity))
    else:
        plt.title("{} type {}".format(quantity, layerType))

    plt.suptitle("Sample = {} Tag = {}".format(sys.argv[3],sys.argv[4]))
    #extra job for the composite dudes
    quantity = quantity.replace("("," ")
    quantity = quantity.replace(")","")
    quantity = quantity.replace("/","by")
    quantity = quantity.replace("-","minus")
    quantity = quantity.replace(" ","_")
    if len(sys.argv) > 2:
        if layerType != "":
            plt.savefig("{}/{}_{}.pdf".format(sys.argv[2],quantity,layerType))
        else:
            plt.savefig("{}/{}.pdf".format(sys.argv[2],quantity))
    else:
        if layerType != "":
            plt.savefig("{}_{}.pdf".format(quantity,layerType))
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
    moduleTypes = np.array(list(map(process_moduleTypes,ak.flatten(tree["{}_moduleType_binary".format(obj)].array()))))
    layerTypes = np.array(list(map(process_layerType,layers)))
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

def plot_composite_distributions(obj):
    global tree
    composite_quantities = [("betaInCut","-","abs(betaIn)"),("betaOutCut","-","abs(betaOut)"),("deltaBetaCut","-","abs(deltaBeta)"),("zOut","-","zLo"),("zHi","-","zOut"),("abs(betaIn)","/","betaInCut"),("abs(betaOut)","/","betaOutCut"),("abs(deltaBeta)","/","deltaBetaCut")]

    matchedMask = tree["{}_isFake".format(obj)].array() == 0
    layers = np.array(list(map(process_layers,ak.flatten(tree["{}_layer_binary".format(obj)].array()))))
    moduleTypes = np.array(list(map(process_moduleTypes,ak.flatten(tree["{}_moduleType_binary".format(obj)].array()))))
    layerTypes = np.array(list(map(process_layerType,layers)))
    unique_layerTypes = np.unique(layerTypes, axis = 0)
    unique_layerTypes = np.append(unique_layerTypes,"")
    print(unique_layerTypes)

    for layerType in unique_layerTypes:
        print("layerType = {}".format(layerType))
        for composite_quantity in composite_quantities:
            print("composite quantity = {} {} {}".format(composite_quantity[0], composite_quantity[1], composite_quantity[2]))

            if composite_quantity[0][:4] == "abs(":
                firstArray = abs(tree["{}_{}".format(obj,composite_quantity[0][4:-1])].array())
            else:
                firstArray = tree["{}_{}".format(obj,composite_quantity[0])].array()

            if composite_quantity[2][:4] == "abs(":
                secondArray = abs(tree["{}_{}".format(obj,composite_quantity[2][4:-1])].array())
            else:
                secondArray = tree["{}_{}".format(obj,composite_quantity[2])].array()

            if composite_quantity[1] == "-":
                qArray = firstArray - secondArray
            elif composite_quantity[1] == "/":
                qArray = firstArray / secondArray
            else:
                print("Operator {} not defined!".format(composite_quantity[1]))
                sys.exit(1)
            qArray = ak.flatten(qArray)
            if layerType == "":
                qArraySimTrackMatched = qArray[ak.flatten(matchedMask)]
            else:
                qArray = qArray[layerTypes == layerType]
                qArraySimTrackMatched = qArray[ak.flatten(matchedMask)[layerTypes == layerType]]


            if all(qArray == -999):
                continue

            make_plots(qArray,qArraySimTrackMatched,"{} {} {} {}".format(obj,*composite_quantity),layerType)



objects = ["t4","t3"]
for i in objects:
    plot_composite_distributions(i)
    plot_distributions(i)
