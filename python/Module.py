#!/bin/env python
from __future__ import print_function
import ROOT as r
import sys
import os
dirpath = os.path.dirname(os.path.abspath(__file__))

r.gROOT.ProcessLine(".L {}/ModuleDetIdParser.cxx".format(dirpath))

class Module:

    def __init__(self, detid):
        self.detid = int(detid)
        self.moduleSDL = r.SDL.Module(self.detid)

    maxBarrelxyModules = {1:18,2:26,3:36,4:48,5:60,6:78}
    #[layer][ring]
    maxEndcapxyModules ={
        1:{1:20,2:24,3:24,4:28,5:32,6:32,7:36,8:40,9:40,10:44,11:52,12:60,13:64,14:62,15:76},
        2:{1:20,2:24,3:24,4:28,5:32,6:32,7:36,8:40,9:40,10:44,11:52,12:60,13:64,14:62,15:76},
        3:{1:28,2:28,3:32,4:36,5:36,6:40,7:44,8:52,9:56,10:64,11:72,12:76},
        4:{1:28,2:28,3:32,4:36,5:36,6:40,7:44,8:52,9:56,10:64,11:72,12:76},
        5:{1:28,2:28,3:32,4:36,5:36,6:40,7:44,8:52,9:56,10:64,11:72,12:76},
    }
    maxBarrelFlatZModules = {1:7,2:11,3:15,4:24,5:24,6:24}
    maxBarrelTiltedZModules = {1:12,2:12,3:12,4:12,5:12,6:12}

    maxRings = {1:15,2:15,3:12,4:12,5:12}

    def detId(self): return int(self.detid)
    def partnerDetId(self): return int(self.moduleSDL.partnerDetId())
    def subdet(self): return int(self.moduleSDL.subdet())
    def side(self): return int(self.moduleSDL.side())
    def layer(self): return int(self.moduleSDL.layer())
    def logicalLayer(self): return int(self.moduleSDL.layer()) + (6 if self.subdet() == 4 else 0)
    def rod(self): return int(self.moduleSDL.rod())
    def ring(self): return int(self.moduleSDL.ring())
    def module(self): return int(self.moduleSDL.module())
    def isLower(self): return int(self.moduleSDL.isLower())
    def isInverted(self): return int(self.moduleSDL.isInverted())
    def moduleType(self): return int(self.moduleSDL.moduleType())
    def moduleLayerType(self): return int(self.moduleSDL.moduleLayerType())
    def isBarrelFlat(self): return (self.side() ==3 and self.subdet() == 5)
    def isBarrelTilted(self): return (self.subdet() ==5 and self.side() != 3)

    def plusPhiDetId(self):
        plusPhiDetId = int(self.detid)
        if self.isBarrelFlat():
            if self.rod() == self.maxBarrelxyModules[self.layer()]:
                plusPhiDetId -= (self.maxBarrelxyModules[self.layer()] - 1) << 10
            else:
                plusPhiDetId += 1<<10

        elif self.isBarrelTilted():
            if self.module() == self.maxBarrelxyModules[self.layer()]:
                plusPhiDetId -= (self.maxBarrelxyModules[self.layer()] - 1) << 2
            else:
                plusPhiDetId += 1<<2

        else: #endcap
            if self.module() == self.maxEndcapxyModules[self.layer()][self.ring()]:
                plusPhiDetId -= (self.maxEndcapxyModules[self.layer()][self.ring()] - 1) << 2
            else:
                plusPhiDetId += 1<<2

        return int(plusPhiDetId)

    def minusPhiDetId(self):
        minusPhiDetId = int(self.detid)
        if self.isBarrelFlat():
            if self.rod() == 1:
                minusPhiDetId += (self.maxBarrelxyModules[self.layer()] - 1) << 10
            else:
                minusPhiDetId -= 1<<10

        elif self.isBarrelTilted():
            if self.module() == 1:
                minusPhiDetId += (self.maxBarrelxyModules[self.layer()] - 1) << 2
            else:
                minusPhiDetId -= 1<<2

        else: #endcap
            if self.module() == 1:
                minusPhiDetId += (self.maxEndcapxyModules[self.layer()][self.ring()] - 1) << 2
            else:
                minusPhiDetId -= 1<<2

        return int(minusPhiDetId)

    def plusEtaDetId(self):
        plusEtaDetId = int(self.detid)

        if self.isBarrelFlat():
            if self.module() == self.maxBarrelFlatZModules[self.layer()] and self.layer() <= 3:
                plusEtaDetId -= 1<<18 #side changed from 3 to 2
                plusEtaDetId -= self.rod()<<10    #Rod reset
                plusEtaDetId -= self.module() <<2 #Module reset
                plusEtaDetId += self.rod()<<2   #rod of flat assigned to module of tilted
                plusEtaDetId += 1<<10 #rod of tilted reset to 1 (1st module in tilted)
            elif self.layer() > 3:
                print("Max eta range reached in flat barrel! Transition not implemented yet!")
            else:
                plusEtaDetId += 1<<2

        elif self.isBarrelTilted():
            if self.rod() == self.maxBarrelTiltedZModules[self.layer()]:
                print("Max eta range reached in tilted barrel! Transition not implemented yet!")
            else:
                plusEtaDetId += 1<<10
        else: #Endcap - jump to next layer
            if self.layer() < 5:
                plusEtaDetId += 1<<18

        return plusEtaDetId

    def minusEtaDetId(self):

        minusEtaDetId = int(self.detid)
        if self.isBarrelFlat():
            if self.module() == 1 and self.layer() <=3 :
                #Implement the jump from barrel to tilted
                minusEtaDetId -= 2<<18 #side changed from 3 to 1
                minusEtaDetId -= self.rod()<<10    #Rod reset
                minusEtaDetId -= self.module() <<2 #Module reset
                minusEtaDetId += self.rod()<<2   #rod of flat assigned to module of tilted
                minusEtaDetId += 12<<10 #rod of tilted reset to 12 (last module in tilted)
            elif self.layer() > 3:
                print("Max eta range reached in flat barrel! Transition not implemented yet!")
            else:
                minusEtaDetId -= 1<<2

        elif self.isBarrelTilted():
            if self.rod() == 1:
                print("Max eta range reached in tilted barrel! Transition not implemented yet!")
            else:
                minusEtaDetId -= 1<<10
        else: #Endcap - jump to previous layer
            if self.layer() > 1:
                minusEtaDetId -= 1<<18

        return minusEtaDetId

    def plusRDetId(self):
        plusRDetId = int(self.detid)
        if self.subdet() == 4:
            if self.ring() < self.maxRings[self.layer()]:
                plusRDetId += 1<<12
        else:
            print("No Plus R for barrel module!")

        return plusRDetId

    def minusRDetId(self):
        minusRDetId = int(self.detid)
        if self.subdet() == 4:
            if self.ring() > 1:
                minusRDetId -= 1<<12
        else:
            print("No Minus R for barrel module!")

        return minusRDetId

#    def __str__(self):
#        return "detid={} logicalLayer={} side={} moduleType={} ring={}".format(self.detId(), self.logicalLayer(), self.side(), self.moduleType(), self.ring())
    def __str__(self):
        printstr = "detId={}, partnerDetId={},subdet={},side={},layer={},rod={},ring={},module={},moduleType={},isLower={}".format(self.detId(),self.partnerDetId(),self.subdet(),self.side(),self.layer(),self.rod(),self.ring(),self.module(),self.moduleType(),self.isLower())
        return printstr


if __name__ == "__main__":

    try:
        module = Module(int(sys.argv[1]))
    except:
        module = Module(443354118)
    print(module.subdet())
    print(module.layer())
    print(module.isLower())
    print(module.ring())
    print(module.rod())
    print(module.moduleType())
    print(module.moduleLayerType())
    print("Current module\n",module)
    print("Next module in phi\n",Module(module.plusPhiDetId()))
    print("Previous module in phi\n",Module(module.minusPhiDetId()))
    print("Next module in eta\n",Module(module.plusEtaDetId()))
    print("Previous module in eta\n",Module(module.minusEtaDetId()))
    if module.subdet() == 4:
        print("Next module in R\n",Module(module.plusRDetId()))
        print("Previous module in R\n",Module(module.minusRDetId()))
