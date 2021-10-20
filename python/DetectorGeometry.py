#!/bin/env python

import ROOT as r
import json
from tqdm import tqdm
from Module import Module
import math

ptthresh = 0.8

# #                       # {
# #       detId           #     "411309061": [
# #       boundary 1      #         [
# #          z coord      #             -129.13503188311316,
# #          x coord      #             27.99223639171157,
# #          y coord      #             -0.7045785443457637
# #                       #         ],
# #       boundary 2      #         [
# #          z coord      #             -129.13503188311316,
# #          x coord      #             26.454739008300713,
# #          y coord      #             9.176519663647351
# #                       #         ],
# #       boundary 3      #         [
# #          z coord      #             -129.13503188311316,
# #          x coord      #             21.514189904304153,
# #          y coord      #             8.407770971941925
# #                       #         ],
# #       boundary 4      #         [
# #          z coord      #             -129.13503188311316,
# #          x coord      #             23.051687287715012,
# #          y coord      #             -1.4733272360511913
# #                       #         ]
# #                       #     ],
# #                       #     "411309062": [
# #                       #         [
# #                       #             -128.7349884581733,
# #                       #             28.00071896416048,
# #                       #             -0.693051092358993
# #                       #         ],
# #                       #         [
# #                       #             -128.7349884581733,
# #                       #             26.4596055174114,
# #                       #             9.187483779005468
# #                       #         ],
# #                       #         [
# #                       #             -128.7349884581733,
# #                       #             21.51933808172917,
# #                       #             8.416927055630929
# #                       #         ],
# #                       #         [
# #                       #             -128.7349884581733,
# #                       #             23.06045152847825,
# #                       #             -1.4636078157335337
# #                       #         ]
# #                       #     ],
# #                       #     ...
# #                       #     ...
# #                       #     ...
# #                       #     ...
# #                       #     ...
# #                       # }

k = 0.00299792458
B = 3.8
A = k * B / 2.

def Phi_mpi_pi(phi):
    while phi >= math.pi: phi -= 2. * math.pi;
    while phi < -math.pi: phi += 2. * math.pi;
    return phi;

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

    def getMinR(self, detid):
        points = self.geom_data[detid]
        rs = []
        for point in points:
            rs.append(math.sqrt(point[1]**2 + point[2]**2))
        return min(rs)

    def getMaxR(self, detid):
        points = self.geom_data[detid]
        rs = []
        for point in points:
            rs.append(math.sqrt(point[1]**2 + point[2]**2))
        return max(rs)

    def getMinPhi(self, detid):
        points = self.geom_data[detid]
        phis = []
        posphis = []
        negphis = []
        signs = []
        bigger_than_pi_over_2 = []
        for point in points:
            phi = Phi_mpi_pi(math.pi + math.atan2(-point[2],-point[1]))
            phis.append(phi)
            if phi > 0:
                posphis.append(phi)
            else:
                negphis.append(phi)
            signs.append(phi > 0)
            bigger_than_pi_over_2.append(abs(phi) > math.pi / 2.)
        if sum(signs) == 4 or sum(signs) == 0:
            return min(phis)
        elif sum(bigger_than_pi_over_2) == 4:
            return min(posphis)
        else:
            return min(phis)

    def getMaxPhi(self, detid):
        points = self.geom_data[detid]
        phis = []
        posphis = []
        negphis = []
        signs = []
        bigger_than_pi_over_2 = []
        for point in points:
            phi = Phi_mpi_pi(math.pi + math.atan2(-point[2],-point[1]))
            phis.append(phi)
            if phi > 0:
                posphis.append(phi)
            else:
                negphis.append(phi)
            signs.append(phi > 0)
            bigger_than_pi_over_2.append(abs(phi) > math.pi / 2.)
        if sum(signs) == 4 or sum(signs) == 0:
            return max(phis)
        elif sum(bigger_than_pi_over_2) == 4:
            return max(negphis)
        else:
            return max(phis)

    def getMinZ(self, detid):
        points = self.geom_data[detid]
        zs = []
        for point in points:
            zs.append(point[0])
        return min(zs)

    def getMaxZ(self, detid):
        points = self.geom_data[detid]
        zs = []
        for point in points:
            zs.append(point[0])
        return max(zs)

    def getCompatiblePhiRange(self, detid, ptmin, ptmax):
        minr = self.getMinR(detid)
        maxr = self.getMaxR(detid)
        minphi = self.getMinPhi(detid)
        maxphi = self.getMaxPhi(detid)
        pos_q_phi_lo_bound = Phi_mpi_pi(A * minr / ptmax + minphi)
        pos_q_phi_hi_bound = Phi_mpi_pi(A * maxr / ptmin + maxphi)
        neg_q_phi_lo_bound = Phi_mpi_pi(-A * maxr / ptmin + minphi)
        neg_q_phi_hi_bound = Phi_mpi_pi(-A * minr / ptmax + maxphi)
        return [[pos_q_phi_lo_bound, pos_q_phi_hi_bound], [neg_q_phi_lo_bound, neg_q_phi_hi_bound]]

    def getCompatibleEtaRange(self, detid, zmin_bound, zmax_bound):
        minr = self.getMinR(detid)
        maxr = self.getMaxR(detid)
        minz = self.getMinZ(detid)
        maxz = self.getMaxZ(detid)
        if minz > 0:
            maxeta = -math.log(math.tan(math.atan2(maxr, (minz - zmin_bound)) / 2. ))
        else:
            maxeta = -math.log(math.tan(math.atan2(minr, (minz - zmin_bound)) / 2. ))
        if maxz > 0:
            mineta = -math.log(math.tan(math.atan2(minr, (maxz - zmax_bound)) / 2. ))
        else:
            mineta = -math.log(math.tan(math.atan2(maxr, (maxz - zmax_bound)) / 2. ))
        return sorted([mineta, maxeta], key=lambda eta: eta)

    def isConnected(self, detid, etamin, etamax, phimin, phimax, ptmin, ptmax, zmin=-30, zmax=30, verbose=False):

        # Check Phi
        phirange = self.getCompatiblePhiRange(detid, ptmin, ptmax)
        if verbose:
            print(phimin, phimax, phirange)
        if verbose:
            print(Phi_mpi_pi(phimin - phirange[0][0]))
            print(Phi_mpi_pi(phimin - phirange[0][1]))
            print(Phi_mpi_pi(phimax - phirange[0][0]))
            print(Phi_mpi_pi(phimax - phirange[0][1]))
        data = []
        if abs(Phi_mpi_pi(phimin - phirange[0][0])) < math.pi/2.: data.append(Phi_mpi_pi(phimin - phirange[0][0]) > 0)
        if abs(Phi_mpi_pi(phimin - phirange[0][1])) < math.pi/2.: data.append(Phi_mpi_pi(phimin - phirange[0][1]) > 0)
        if abs(Phi_mpi_pi(phimax - phirange[0][0])) < math.pi/2.: data.append(Phi_mpi_pi(phimax - phirange[0][0]) > 0)
        if abs(Phi_mpi_pi(phimax - phirange[0][1])) < math.pi/2.: data.append(Phi_mpi_pi(phimax - phirange[0][1]) > 0)
        if len(data) != 4:
            return False;
        if all(data) or not any(data):
            is_phi_in_range_0 = False
        else:
            is_phi_in_range_0 = True

        if verbose:
            print(data)
            print(all(data))
            print(any(data))
        data = []
        if abs(Phi_mpi_pi(phimin - phirange[1][0])) < math.pi/2.: data.append(Phi_mpi_pi(phimin - phirange[1][0]) > 0)
        if abs(Phi_mpi_pi(phimin - phirange[1][1])) < math.pi/2.: data.append(Phi_mpi_pi(phimin - phirange[1][1]) > 0)
        if abs(Phi_mpi_pi(phimax - phirange[1][0])) < math.pi/2.: data.append(Phi_mpi_pi(phimax - phirange[1][0]) > 0)
        if abs(Phi_mpi_pi(phimax - phirange[1][1])) < math.pi/2.: data.append(Phi_mpi_pi(phimax - phirange[1][1]) > 0)
        if len(data) != 4:
            return False;
        if all(data) or not any(data):
            is_phi_in_range_1 = False
        else:
            is_phi_in_range_1 = True

        if verbose:
            print(data)
            print(all(data))
            print(any(data))

        if verbose:
            print(is_phi_in_range_0)
            print(is_phi_in_range_1)

        if not is_phi_in_range_0 and not is_phi_in_range_1:
            return False

        # Check Eta
        etarange = self.getCompatibleEtaRange(detid, zmin, zmax)
        if verbose:
            print(etamin, etamax, etarange)
        data = []
        data.append((etamin - etarange[0]) > 0)
        data.append((etamin - etarange[1]) > 0)
        data.append((etamax - etarange[0]) > 0)
        data.append((etamax - etarange[1]) > 0)
        if all(data) or not any(data):
            is_eta_in_range = False
        else:
            is_eta_in_range = True
        if not is_eta_in_range:
            return False

        return True

def printPixelMap():
    printPixelMap_v3()

def printPixelMap_v1():

    import os

    dirpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    det_geom = DetectorGeometry("/data2/segmentlinking/phase2_2020_0428.txt".format(dirpath))

    super_bins = {}

    neta = 40.
    nphi = 72.

    isuper_bin = 0
    pt_bounds = [0.9, 2.0, 4.0, 10., 50.]
    # pt_bounds = [0.9, 2.0]
    for ipt in xrange(len(pt_bounds)-1):
        pt_lo = pt_bounds[ipt]
        pt_hi = pt_bounds[ipt+1]
        for ieta in xrange(int(neta)):
            eta_lo = -2.6 + (5.2 / neta) * ieta
            eta_hi = -2.6 + (5.2 / neta) * (ieta + 1)
            for iphi in xrange(int(nphi)):
                phi_lo = -math.pi + (2*math.pi / nphi) * iphi
                phi_hi = -math.pi + (2*math.pi / nphi) * (iphi + 1)
                super_bins[isuper_bin] = (pt_lo, pt_hi, eta_lo, eta_hi, phi_lo, phi_hi)
                isuper_bin += 1
    print(len(super_bins))

    maps = {}
    for layer in [1, 2, 3, 4, 5, 6]:
        for subdet in [4, 5]:
            for isuper_bin in super_bins.keys():
                bounds = super_bins[isuper_bin]
                maps[isuper_bin] = []
                for detid in det_geom.getDetIds():
                    if Module(detid).layer() == layer and Module(detid).isLower() == 1 and Module(detid).moduleType() == 0 and Module(detid).subdet() == subdet:
                        if det_geom.isConnected(detid, bounds[2], bounds[3], bounds[4], bounds[5], bounds[0], bounds[1], -30, 30):
                            maps[isuper_bin].append(detid)
                print(isuper_bin, layer, subdet, bounds[2], bounds[3], bounds[4], bounds[5], maps[isuper_bin])

def printPixelMap_v2():

    import os

    dirpath = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    det_geom = DetectorGeometry("/data2/segmentlinking/phase2_2020_0428.txt".format(dirpath))

    neta = 40.
    nphi = 72.

    maps = {}
    pt_bounds = [0.9, 2.0, 4.0, 10., 50.]
    for ipt in xrange(len(pt_bounds)-1):
        for ieta in xrange(int(neta)):
            for iphi in xrange(int(nphi)):
                maps[(ipt, ieta, iphi)] = {}
                maps[(ipt, ieta, iphi, 1)] = {}
                maps[(ipt, ieta, iphi, -1)] = {}
                for layer in [1, 2, 3, 4, 5, 6]:
                    for subdet in [4, 5]:
                        maps[(ipt, ieta, iphi)][(layer, subdet)] = []
                        maps[(ipt, ieta, iphi, 1)][(layer, subdet)] = []
                        maps[(ipt, ieta, iphi, -1)][(layer, subdet)] = []

    for detid in tqdm(det_geom.getDetIds()):
        module = Module(detid)
        layer = module.layer()
        subdet = module.subdet()
        if module.isLower() != 1 or module.moduleType() != 0:
            continue
        # if module.subdet() == 4:
        #     if module.ring() != 1 and module.ring() != 2:
                # continue
        for ipt in xrange(len(pt_bounds)-1):
            pt_lo = pt_bounds[ipt]
            pt_hi = pt_bounds[ipt+1]
            etamin = det_geom.getCompatibleEtaRange(detid, -30, 30)[0]
            etamax = det_geom.getCompatibleEtaRange(detid, -30, 30)[1]
            ietamin = int((etamin + 2.6) / (5.2 / neta))
            ietamax = int((etamax + 2.6) / (5.2 / neta))
            prelim_etabins = range(ietamin, ietamax+1)
            etabins = []
            for ieta in prelim_etabins:
                if ieta >= 0 and ieta < neta:
                    etabins.append(ieta)
            iphimin_pos = int((det_geom.getCompatiblePhiRange(detid, pt_lo, pt_hi)[0][0] + math.pi) / (2*math.pi / nphi))
            iphimax_pos = int((det_geom.getCompatiblePhiRange(detid, pt_lo, pt_hi)[0][1] + math.pi) / (2*math.pi / nphi))
            iphimin_neg = int((det_geom.getCompatiblePhiRange(detid, pt_lo, pt_hi)[1][0] + math.pi) / (2*math.pi / nphi))
            iphimax_neg = int((det_geom.getCompatiblePhiRange(detid, pt_lo, pt_hi)[1][1] + math.pi) / (2*math.pi / nphi))
            if iphimin_pos <= iphimax_pos:
                phibins_pos = range(iphimin_pos, iphimax_pos)
            else:
                phibins_pos = range(0, iphimax_pos) + range(iphimin_pos, 72)
            if iphimin_neg <= iphimax_neg:
                phibins_neg = range(iphimin_neg, iphimax_neg)
            else:
                phibins_neg = range(0, iphimax_neg) + range(iphimin_neg, 72)
            for ieta in etabins:
                for iphi in phibins_pos:
                    maps[(ipt, ieta, iphi)][(layer, subdet)].append(detid)
                for iphi in phibins_neg:
                    maps[(ipt, ieta, iphi)][(layer, subdet)].append(detid)
                for iphi in phibins_pos:
                    maps[(ipt, ieta, iphi, 1)][(layer, subdet)].append(detid)
                for iphi in phibins_neg:
                    maps[(ipt, ieta, iphi, -1)][(layer, subdet)].append(detid)

    import os
    os.system("mkdir -p pixelmap")
    g = open("pixelmap/pLS_map_ElCheapo.txt", "w")
    g_pos = open("pixelmap/pLS_map_pos_ElCheapo.txt", "w")
    g_neg = open("pixelmap/pLS_map_neg_ElCheapo.txt", "w")
    fs = {}
    for layer in [1, 2, 3, 4, 5, 6]:
        for subdet in [4, 5]:
            fs[(layer, subdet)] = open("pixelmap/pLS_map_layer{}_subdet{}.txt".format(layer, subdet), "w")
            fs[(layer, subdet, 1)] = open("pixelmap/pLS_map_pos_layer{}_subdet{}.txt".format(layer, subdet), "w")
            fs[(layer, subdet, -1)] = open("pixelmap/pLS_map_neg_layer{}_subdet{}.txt".format(layer, subdet), "w")
    for ipt in xrange(len(pt_bounds)-1):
        for ieta in xrange(int(neta)):
            for iphi in xrange(int(nphi)):
                isuperbin = (ipt * nphi * neta) + (ieta * nphi) + iphi
                all_detids = []
                all_pos_detids = []
                all_neg_detids = []
                for layer in [1, 2, 3, 4, 5, 6]:
                    for subdet in [4, 5]:
                        maps[(ipt, ieta, iphi)][(layer, subdet)] = list(set(maps[(ipt, ieta, iphi)][(layer, subdet)]))
                        maps[(ipt, ieta, iphi, 1)][(layer, subdet)] = list(set(maps[(ipt, ieta, iphi, 1)][(layer, subdet)]))
                        maps[(ipt, ieta, iphi, -1)][(layer, subdet)] = list(set(maps[(ipt, ieta, iphi, -1)][(layer, subdet)]))
                        fs[(layer, subdet)].write("{} {} {}\n".format(int(isuperbin), len(maps[(ipt, ieta, iphi)][(layer, subdet)]), " ".join([ str(x) for x in maps[(ipt, ieta, iphi)][(layer, subdet)] ])))
                        fs[(layer, subdet, 1)].write("{} {} {}\n".format(int(isuperbin), len(maps[(ipt, ieta, iphi, 1)][(layer, subdet)]), " ".join([ str(x) for x in maps[(ipt, ieta, iphi, 1)][(layer, subdet)] ])))
                        fs[(layer, subdet, -1)].write("{} {} {}\n".format(int(isuperbin), len(maps[(ipt, ieta, iphi, -1)][(layer, subdet)]), " ".join([ str(x) for x in maps[(ipt, ieta, iphi, -1)][(layer, subdet)] ])))
                        all_detids += maps[(ipt, ieta, iphi)][(layer, subdet)]
                        all_pos_detids += maps[(ipt, ieta, iphi, 1)][(layer, subdet)]
                        all_neg_detids += maps[(ipt, ieta, iphi, -1)][(layer, subdet)]
                g.write("{} {} {}\n".format(int(isuperbin), len(all_detids), " ".join([ str(x) for x in all_detids ])))
                g_pos.write("{} {} {}\n".format(int(isuperbin), len(all_pos_detids), " ".join([ str(x) for x in all_pos_detids ])))
                g_neg.write("{} {} {}\n".format(int(isuperbin), len(all_neg_detids), " ".join([ str(x) for x in all_neg_detids ])))

def printPixelMap_v3():

    """
    To print out pixel maps
    """

    import os

    # The text file is a json file with "detid" -> {xyz of 4 corners of the module}
    det_geom = DetectorGeometry("/data2/segmentlinking/phase2_2020_0428.txt")

    # Define the binning of "super bins"
    neta = 25.
    nphi = 72.
    nz = 25.
    # pt_bounds = [0.9, 2.0, 4.0, 10., 50.]
    pt_bounds = [ptthresh, 2.0, 10000.]

    # Grand map object that will hold the mapping of the pixel map
    # maps[(ipt, ieta, iphi, iz)][(layer, subdet)] = [] # for both positive and negative
    # maps[(ipt, ieta, iphi, iz, 1)][(layer, subdet)] = [] # for positive oly
    # maps[(ipt, ieta, iphi, iz,-1)][(layer, subdet)] = [] # for negative oly
    maps = {}

    # Initialize empty lists for the pixel map 
    for ipt in xrange(len(pt_bounds)-1):
        for ieta in xrange(int(neta)):
            for iphi in xrange(int(nphi)):
                for iz in xrange(int(nz)):

                    # Maps without split by charge
                    maps[(ipt, ieta, iphi, iz)] = {}

                    # Maps with split by charge (positive)
                    maps[(ipt, ieta, iphi, iz, 1)] = {}

                    # Maps with split by charge (negative)
                    maps[(ipt, ieta, iphi, iz, -1)] = {}

                    # The maps will then be split by (layer, subdet)
                    for layer in [1, 2, 3, 4, 5, 6]:
                        for subdet in [4, 5]:

                            # Maps without split by charge
                            maps[(ipt, ieta, iphi, iz)][(layer, subdet)] = []

                            # Maps split by charge
                            maps[(ipt, ieta, iphi, iz, 1)][(layer, subdet)] = []

                            # Maps split by charge
                            maps[(ipt, ieta, iphi, iz, -1)][(layer, subdet)] = []

    # Loop over the detids and for each detid compute which superbins it is connected to
    for detid in tqdm(det_geom.getDetIds()):

        # Parse the layer and subdet
        module = Module(detid)
        layer = module.layer()
        subdet = module.subdet()

        # Skip if the module is not PS module and is not lower module
        if module.isLower() != 1 or module.moduleType() != 0:
            continue

        # For this module, now compute which super bins they belong to
        # To compute which super bins it belongs to, one needs to provide at least pt and z window to compute compatible eta and phi range
        # So we have a loop in pt and Z
        for ipt in xrange(len(pt_bounds)-1):
            for iz in xrange(int(nz)):

                # The zmin, zmax of consideration
                zmin = -30 + iz * (60. / nz)
                zmax = -30 + (iz + 1) * (60. / nz)

                zmin -= 0.05
                zmin += 0.05

                # The ptmin, ptmax of consideration
                pt_lo = pt_bounds[ipt]
                pt_hi = pt_bounds[ipt+1]

                # Based on the zmin and zmax, and the detid we can get eta min and eta max of compatible range
                etamin = det_geom.getCompatibleEtaRange(detid, zmin, zmax)[0]
                etamax = det_geom.getCompatibleEtaRange(detid, zmin, zmax)[1]

                etamin -= 0.05
                etamax += 0.05

                # Compute the indices of the compatible eta range
                ietamin = int((etamin + 2.6) / (5.2 / neta))
                ietamax = int((etamax + 2.6) / (5.2 / neta))

                # Since the etas are restricted to 2.6, need to chop it off if it's out of bounds
                # prelim_etabins = range(ietamin-1, ietamax+2) # add -1 and +1 to cover some inefficiencies
                prelim_etabins = range(ietamin, ietamax+1) # add -1 and +1 to cover some inefficiencies
                etabins = []
                for ieta in prelim_etabins:
                    if ieta >= 0 and ieta < neta:
                        etabins.append(ieta)

                # Now compute the ranges of iphi min and max for given pt ranges
                iphimin_pos = int((det_geom.getCompatiblePhiRange(detid, pt_lo, pt_hi)[0][0] + math.pi) / (2*math.pi / nphi))
                iphimax_pos = int((det_geom.getCompatiblePhiRange(detid, pt_lo, pt_hi)[0][1] + math.pi) / (2*math.pi / nphi))
                iphimin_neg = int((det_geom.getCompatiblePhiRange(detid, pt_lo, pt_hi)[1][0] + math.pi) / (2*math.pi / nphi))
                iphimax_neg = int((det_geom.getCompatiblePhiRange(detid, pt_lo, pt_hi)[1][1] + math.pi) / (2*math.pi / nphi))

                # if the range is crossing the -pi v. pi boundary special care is needed
                if iphimin_pos <= iphimax_pos:
                    phibins_pos = range(iphimin_pos, iphimax_pos)
                else:
                    phibins_pos = range(0, iphimax_pos) + range(iphimin_pos, int(nphi))
                if iphimin_neg <= iphimax_neg:
                    phibins_neg = range(iphimin_neg, iphimax_neg)
                else:
                    phibins_neg = range(0, iphimax_neg) + range(iphimin_neg, int(nphi))

                # Now we have a list of (ipt, ieta, iphi, iz)'s that are compatible so we fill the map
                for ieta in etabins:
                    for iphi in phibins_pos:
                        maps[(ipt, ieta, iphi, iz)][(layer, subdet)].append(detid)
                    for iphi in phibins_neg:
                        maps[(ipt, ieta, iphi, iz)][(layer, subdet)].append(detid)
                    for iphi in phibins_pos:
                        maps[(ipt, ieta, iphi, iz, 1)][(layer, subdet)].append(detid)
                    for iphi in phibins_neg:
                        maps[(ipt, ieta, iphi, iz, -1)][(layer, subdet)].append(detid)

    # Writing out the pixel map results
    import os
    os.system("mkdir -p pixelmap")

    # Grand map txt file that will hold everything regardless of the layers
    g = open("pixelmap/pLS_map_ElCheapo.txt", "w")
    g_pos = open("pixelmap/pLS_map_pos_ElCheapo.txt", "w")
    g_neg = open("pixelmap/pLS_map_neg_ElCheapo.txt", "w")

    # pixel maps split by layers
    fs = {}
    for layer in [1, 2, 3, 4, 5, 6]:
        for subdet in [4, 5]:
            fs[(layer, subdet)] = open("pixelmap/pLS_map_layer{}_subdet{}.txt".format(layer, subdet), "w")
            fs[(layer, subdet, 1)] = open("pixelmap/pLS_map_pos_layer{}_subdet{}.txt".format(layer, subdet), "w")
            fs[(layer, subdet, -1)] = open("pixelmap/pLS_map_neg_layer{}_subdet{}.txt".format(layer, subdet), "w")

    # Loop over the super bins
    for ipt in xrange(len(pt_bounds)-1):
        for ieta in xrange(int(neta)):
            for iphi in xrange(int(nphi)):
                for iz in xrange(int(nz)):

                    # Compute the superbin index the phase-space belongs to
                    isuperbin = (ipt * nphi * neta * nz) + (ieta * nphi * nz) + nz * iphi + iz

                    # Temporary list to aggregate the detids
                    all_detids = []
                    all_pos_detids = []
                    all_neg_detids = []

                    # Loop over the possible layer and subdets
                    for layer in [1, 2, 3, 4, 5, 6]:
                        for subdet in [4, 5]:

                            # Obtain unique set of detids
                            maps[(ipt, ieta, iphi, iz)][(layer, subdet)] = list(set(maps[(ipt, ieta, iphi, iz)][(layer, subdet)]))
                            maps[(ipt, ieta, iphi, iz, 1)][(layer, subdet)] = list(set(maps[(ipt, ieta, iphi, iz, 1)][(layer, subdet)]))
                            maps[(ipt, ieta, iphi, iz, -1)][(layer, subdet)] = list(set(maps[(ipt, ieta, iphi, iz, -1)][(layer, subdet)]))

                            # Write out to the individual map files
                            fs[(layer, subdet)].write("{} {} {}\n".format(int(isuperbin), len(maps[(ipt, ieta, iphi, iz)][(layer, subdet)]), " ".join([ str(x) for x in maps[(ipt, ieta, iphi, iz)][(layer, subdet)] ])))
                            fs[(layer, subdet, 1)].write("{} {} {}\n".format(int(isuperbin), len(maps[(ipt, ieta, iphi, iz, 1)][(layer, subdet)]), " ".join([ str(x) for x in maps[(ipt, ieta, iphi, iz, 1)][(layer, subdet)] ])))
                            fs[(layer, subdet, -1)].write("{} {} {}\n".format(int(isuperbin), len(maps[(ipt, ieta, iphi, iz, -1)][(layer, subdet)]), " ".join([ str(x) for x in maps[(ipt, ieta, iphi, iz, -1)][(layer, subdet)] ])))

                            # Aggregate all the detids for a given superbin in order to later write out to the grand map txt file that holds all detids regardless of layers
                            all_detids += maps[(ipt, ieta, iphi, iz)][(layer, subdet)]
                            all_pos_detids += maps[(ipt, ieta, iphi, iz, 1)][(layer, subdet)]
                            all_neg_detids += maps[(ipt, ieta, iphi, iz, -1)][(layer, subdet)]

                    maps[(ipt, ieta, iphi, iz)]["all"] = list(set(all_detids))
                    maps[(ipt, ieta, iphi, iz, 1)]["all"] = list(set(all_pos_detids))
                    maps[(ipt, ieta, iphi, iz,-1)]["all"] = list(set(all_neg_detids))

                    # Now write the entire detid for the entire outer tracker regardless of layers for the superbin at hand
                    g.write("{} {} {}\n".format(int(isuperbin), len(maps[(ipt, ieta, iphi, iz)]["all"]), " ".join([ str(x) for x in maps[(ipt, ieta, iphi, iz)]["all"] ])))
                    g_pos.write("{} {} {}\n".format(int(isuperbin), len(maps[(ipt, ieta, iphi, iz, 1)]["all"]), " ".join([ str(x) for x in maps[(ipt, ieta, iphi, iz, 1)]["all"] ])))
                    g_neg.write("{} {} {}\n".format(int(isuperbin), len(maps[(ipt, ieta, iphi, iz,-1)]["all"]), " ".join([ str(x) for x in maps[(ipt, ieta, iphi, iz,-1)]["all"] ])))

    # Declaring histograms for mapping multiplicities
    ofile = r.TFile("pixelmap/pLS_map.root", "recreate")
    nconns = {}
    for ipt in xrange(len(pt_bounds)-1):
        for iz in xrange(int(nz)):
            nconns[(ipt, iz)] = r.TH2F("pLS_map_ElCheapo_ipt{}_iz{}".format(ipt, iz), "", int(nphi), -math.pi, math.pi, int(neta), -2.6, 2.6)
            nconns[(ipt, iz, 1)] = r.TH2F("pLS_map_pos_ElCheapo_ipt{}_iz{}".format(ipt, iz), "", int(nphi), -math.pi, math.pi, int(neta), -2.6, 2.6)
            nconns[(ipt, iz,-1)] = r.TH2F("pLS_map_neg_ElCheapo_ipt{}_iz{}".format(ipt, iz), "", int(nphi), -math.pi, math.pi, int(neta), -2.6, 2.6)
            for layer in [1, 2, 3, 4, 5, 6]:
                for subdet in [4, 5]:
                    nconns[(ipt, iz, layer, subdet)] = r.TH2F("pLS_map_layer{}_subdet{}_ipt{}_iz{}".format(layer, subdet, ipt, iz), "", int(nphi), -math.pi, math.pi, int(neta), -2.6, 2.6)
                    nconns[(ipt, iz, layer, subdet, 1)] = r.TH2F("pLS_map_pos_layer{}_subdet{}_ipt{}_iz{}".format(layer, subdet, ipt, iz), "", int(nphi), -math.pi, math.pi, int(neta), -2.6, 2.6)
                    nconns[(ipt, iz, layer, subdet,-1)] = r.TH2F("pLS_map_neg_layer{}_subdet{}_ipt{}_iz{}".format(layer, subdet, ipt, iz), "", int(nphi), -math.pi, math.pi, int(neta), -2.6, 2.6)

    # Now filling the histograms
    for ipt in xrange(len(pt_bounds)-1):
        for iz in xrange(int(nz)):
            for ieta in xrange(int(neta)):
                for iphi in xrange(int(nphi)):
                    nconns[(ipt, iz)].SetBinContent(iphi + 1, ieta + 1, len(maps[(ipt, ieta, iphi, iz)]["all"]))
                    nconns[(ipt, iz, 1)].SetBinContent(iphi + 1, ieta + 1, len(maps[(ipt, ieta, iphi, iz, 1)]["all"]))
                    nconns[(ipt, iz,-1)].SetBinContent(iphi + 1, ieta + 1, len(maps[(ipt, ieta, iphi, iz,-1)]["all"]))
            for layer in [1, 2, 3, 4, 5, 6]:
                for subdet in [4, 5]:
                    for ieta in xrange(int(neta)):
                        for iphi in xrange(int(nphi)):
                            nconns[(ipt, iz, layer, subdet)].SetBinContent(iphi + 1, ieta + 1, len(maps[(ipt, ieta, iphi, iz)][(layer, subdet)]))
                            nconns[(ipt, iz, layer, subdet, 1)].SetBinContent(iphi + 1, ieta + 1, len(maps[(ipt, ieta, iphi, iz, 1)][(layer, subdet)]))
                            nconns[(ipt, iz, layer, subdet,-1)].SetBinContent(iphi + 1, ieta + 1, len(maps[(ipt, ieta, iphi, iz,-1)][(layer, subdet)]))

    # Write the Histograms
    ofile.cd()
    for ipt in xrange(len(pt_bounds)-1):
        for iz in xrange(int(nz)):
            nconns[(ipt, iz)].Write()
            nconns[(ipt, iz, 1)].Write()
            nconns[(ipt, iz,-1)].Write()
            for layer in [1, 2, 3, 4, 5, 6]:
                for subdet in [4, 5]:
                    nconns[(ipt, iz, layer, subdet)].Write()
                    nconns[(ipt, iz, layer, subdet, 1)].Write()
                    nconns[(ipt, iz, layer, subdet,-1)].Write()

if __name__ == "__main__":

    printPixelMap()
