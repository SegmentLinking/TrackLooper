#!/bin/env python

import ROOT as r
import json
from tqdm import tqdm
from Module import Module

class DetectorGeometry:

    def __init__(self, data):
        self.data = data
        self.datafile = open(self.data)
        self.geom_data_raw = json.load(self.datafile)
        self.geom_data = {}
        for key in tqdm(self.geom_data_raw, "Loading detector geometries (i.e. boundaries)"):
            detid = int(key)
            new_list = []
            for x in self.geom_data_raw[key]:
                new_x = [ float(y) for y in x ]
                new_list.append(new_x)
            self.geom_data[detid] = new_list

        # average values
        self.average_radii = []
        self.average_radii.append(24.726409077007705) # Layer 1 average radius
        self.average_radii.append(37.059873804403495) # Layer 2 average radius
        self.average_radii.append(52.17677700048082)  # Layer 3 average radius
        self.average_radii.append(68.61016946477243)  # Layer 4 average radius
        self.average_radii.append(85.91013998484999)  # Layer 5 average radius
        self.average_radii.append(110.71009476599565) # Layer 6 average radius
        self.average_zs = []
        self.average_zs.append(130.93374689440995) # Layer 1 average Z (endcap)
        self.average_zs.append(154.74990605590062) # Layer 2 average Z (endcap)
        self.average_zs.append(185.1167890070922)  # Layer 3 average Z (endcap)
        self.average_zs.append(221.39607712765957) # Layer 4 average Z (endcap)
        self.average_zs.append(264.76252304964544) # Layer 5 average Z (endcap)

    def getData(self, filt=None):
        if filt:
            rtndict = dict(filter(filt, self.geom_data.items()))
            return rtndict
        else:
            return self.geom_data

    def getDetIds(self, filt=None):
        if filt:
            rtndict = dict(filter(filt, self.geom_data.items()))
            return rtndict.keys()
        else:
            return self.geom_data.keys()

    def buildByLayer(self):
        self.barrel_lower_det_ids = []
        print("Building barrel detIds")
        for i in tqdm(range(1, 7)):
            self.barrel_lower_det_ids.append(
                    self.getDetIds(lambda x:
                            Module(x[0]).subdet() == 5 and
                            Module(x[0]).layer() == i and
                            Module(x[0]).isLower() == 1
                            )
                    )
        self.endcap_lower_det_ids = []
        print("Building endcap detIds")
        for i in tqdm(range(1, 6)):
            self.endcap_lower_det_ids.append(
                    self.getDetIds(lambda x:
                            Module(x[0]).subdet() == 4 and
                            Module(x[0]).layer() == i and
                            Module(x[0]).isLower() == 1
                            )
                    )

    def getBarrelLayerDetIds(self, layer):
        return self.barrel_lower_det_ids[layer-1]

    def getEndcapLayerDetIds(self, layer):
        return self.endcap_lower_det_ids[layer-1]

    def getBarrelLayerAverageRadius(self, layer):
        return self.average_radii[layer-1]

    def getEndcapLayerAverageAbsZ(self, layer):
        return self.average_zs[layer-1]
