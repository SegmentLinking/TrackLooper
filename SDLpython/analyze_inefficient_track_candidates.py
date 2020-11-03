#!/bin/env python

import ROOT as r

### Constants

segment_layers = [
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
        (5, 6),
        ]

segment_ascii = [
        "o-o",
        "  o-o",
        "    o-o",
        "      o-o",
        "        o-o",
        ]

triplet_layers = [
        (1, 2, 3),
        (2, 3, 4),
        (3, 4, 5),
        (4, 5, 6),
        ]

triplet_ascii = [
        "o-o-o",
        "  o-o-o",
        "    o-o-o",
        "      o-o-o",
        ]

tracklet_layers = [
        (1, 2, 3, 4),
        (2, 3, 4, 5),
        (3, 4, 5, 6),
        (1, 2, 4, 5),
        (2, 3, 5, 6),
        (1, 2, 5, 6),
        ]

tracklet_ascii = [
        "o-o-o-o",
        "  o-o-o-o",
        "    o-o-o-o",
        "o-o---o-o",
        "  o-o---o-o",
        "o-o-----o-o",
        ]

trackcandidate_layers = [
        (1, 2, 3, 4, 5, 6),
        (1, 2, 3, 4, 5   ),
        (1, 2, 3,    5, 6),
        (1, 2,    4, 5, 6),
        (   2, 3, 4, 5, 6),
        ]

trackcandidate_ascii = [
        "o-o-o-o-o-o",
        "o-o-o-o-o  ",
        "o-o-o---o-o",
        "o-o---o-o-o",
        "  o-o-o-o-o",
        ]

indent = "               "

### Functions

def get_good_pair_layers(isim, event):
    neglayers = []
    poslayers = []
    # print event.sim_simHitLayer[isim]
    # print event.sim_simHitBoth[isim]
    for layer, both in zip(event.sim_simHitLayer[isim], event.sim_simHitBoth[isim]):
        if layer > 0 and both > 0: poslayers.append(layer)
        else: neglayers.append(-layer)
    lst3 = [value for value in neglayers if value in poslayers] 
    return list(set(lst3))

def make_md_layer_score_list(layers):
    scores = []
    for ilayer_to_check in range(1, 7):
        if ilayer_to_check in layers:
            scores.append("o")
        else:
            scores.append("x")
    return " ".join(scores)

def make_sg_layer_score_list(layers):
    scores = []
    for index, isg_layer_to_check in enumerate(segment_layers):
        if isg_layer_to_check in layers:
            scores.append("SG" + indent + segment_ascii[index])
    return "\n".join(scores)

def make_tl_layer_score_list(layers):
    scores = []
    for index, itl_layer_to_check in enumerate(tracklet_layers):
        if itl_layer_to_check in layers:
            scores.append("TL" + indent + tracklet_ascii[index])
    return "\n".join(scores)

def make_tp_layer_score_list(layers):
    scores = []
    for index, itp_layer_to_check in enumerate(triplet_layers):
        if itp_layer_to_check in layers:
            scores.append("TP" + indent + triplet_ascii[index])
    return "\n".join(scores)

def make_tc_layer_score_list(layers):
    scores = []
    for index, itc_layer_to_check in enumerate(trackcandidate_layers):
        if itc_layer_to_check in layers:
            scores.append("TC" + indent + trackcandidate_ascii[index])
    return "\n".join(scores)




def print_sim_trk_info(isim, event, status):
    layers = get_good_pair_layers(isim, event)
    layers = make_md_layer_score_list(layers)
    print "{} {:5.2f} {:5.2f} {}".format(status, event.sim_pt[isim], event.sim_eta[isim], layers)
    return layers

def print_sim_trk_md_mtv_match_info(isim, event):
    layers_with_md = []
    for imd in event.sim_mdIdx_isMTVmatch[isim]:
        layers_with_md.append(event.md_layer[imd][0])
    print "MD               " + make_md_layer_score_list(layers_with_md)
    return make_md_layer_score_list(layers_with_md)

def print_sim_trk_sg_mtv_match_info(isim, event):
    layers_with_sg = []
    for isg in event.sim_sgIdx_isMTVmatch[isim]:
        layers_with_sg.append(tuple([event.sg_layer[isg][0],event.sg_layer[isg][2]]))
    print make_sg_layer_score_list(layers_with_sg)

def print_sim_trk_tl_mtv_match_info(isim, event):
    layers_with_tl = []
    for itl in event.sim_tlIdx_isMTVmatch[isim]:
        layers_with_tl.append(tuple([event.tl_layer[itl][0],event.tl_layer[itl][1],event.tl_layer[itl][2],event.tl_layer[itl][3]]))
    print make_tl_layer_score_list(layers_with_tl)

def print_sim_trk_tp_mtv_match_info(isim, event):
    layers_with_tp = []
    for itp in event.sim_tpIdx_isMTVmatch[isim]:
        layers_with_tp.append(tuple([event.tp_layer[itp][0],event.tp_layer[itp][2],event.tp_layer[itp][4]]))
    print make_tp_layer_score_list(layers_with_tp)

def print_sim_trk_tc_mtv_match_info(isim, event):
    layers_with_tc = []
    for itc in event.sim_tcIdx_isMTVmatch[isim]:
        layers = tuple(list(set(event.tc_layer[itc])))
        layers_with_tc.append(layers)
    print make_tc_layer_score_list(layers_with_tc)

def print_details(isim, event):
    for imd in event.sim_mdIdx_isMTVmatch[isim]: print event.md_layer[imd]
    for imd in event.sim_mdIdx_isMTVmatch[isim]: print event.md_hitIdx[imd]
    for isg in event.sim_sgIdx_isMTVmatch[isim]: print event.sg_layer[isg]
    for isg in event.sim_sgIdx_isMTVmatch[isim]: print event.sg_hitIdx[isg]
    for itl in event.sim_tlIdx_isMTVmatch[isim]: print event.tl_layer[itl]
    for itl in event.sim_tlIdx_isMTVmatch[isim]: print event.tl_hitIdx[itl]
    for itp in event.sim_tpIdx_isMTVmatch[isim]: print event.tp_layer[itp]
    for itp in event.sim_tpIdx_isMTVmatch[isim]: print event.tp_hitIdx[itp]
    for itc in event.sim_tcIdx_isMTVmatch[isim]: print event.tc_layer[itc]
    for itc in event.sim_tcIdx_isMTVmatch[isim]: print event.tc_hitIdx[itc]

def print_header():
    print "Reco  PT   Eta   Layers"
    print "____________________________"



def process_sim_trk_info(isim, event, status):
    layers = get_good_pair_layers(isim, event)
    layers = make_md_layer_score_list(layers)
    return layers

def process_sim_trk_md_mtv_match_info(isim, event):
    layers_with_md = []
    for imd in event.sim_mdIdx_isMTVmatch[isim]:
        layers_with_md.append(event.md_layer[imd][0])
    return make_md_layer_score_list(layers_with_md)





def is_not_recoable(layers):
    ls = layers.split()
    # if ls[1] == "x" or ls[4] == "x":
    if ls[4] == "x":
        return True
    else:
        nx = 0
        for i in ls:
            if i == "x":
                nx += 1
        if nx > 1:
            return True
    return False

if __name__ == "__main__":

    # f = r.TFile("/home/users/phchang/public_html/analysis/sdl/TrackLooper_/results/write_sdl_ntuple/pt0p5_2p0_20200907_Test_v1//fulleff_pt0p5_2p0.root")
    # f = r.TFile("/home/users/phchang/public_html/analysis/sdl/TrackLooper_/results/write_sdl_ntuple/pt0p5_2p0_20200910_Test_v2//fulleff_pt0p5_2p0.root")
    # f = r.TFile("/home/users/phchang/public_html/analysis/sdl/TrackLooper_/results/write_sdl_ntuple/pu200_w_truth_pdgid211_20200919_Test_v1_w_gap//fulleff_pu200_w_truth_pdgid211.root")
    # f = r.TFile("/home/users/phchang/public_html/analysis/sdl/TrackLooper_/results/write_sdl_ntuple/pu200_w_truth_pdgid211_20200919_Test_v2_w_gap//fulleff_pu200_w_truth_pdgid211.root")
    f = r.TFile("/home/users/phchang/public_html/analysis/sdl/TrackLooper_/results/write_sdl_ntuple/pion_20200925_Test_v2//fulleff_pion.root")
    tree = f.Get("tree")

    ndenom = 0
    nnumer = 0
    nineff = 0

    nsimfail = 0
    nmdfail = 0

    for ievent, event in enumerate(tree):

        for isim, (pt, eta, tcIdx_isMTVmatch) in enumerate(zip(event.sim_pt, event.sim_eta, event.sim_tcIdx_isMTVmatch)):

            pdgid = event.sim_pdgId[isim]
            bunch = event.sim_bunchCrossing[isim]
            dxy = event.sim_pca_dxy[isim]
            dz = event.sim_pca_dz[isim]

            if abs(pdgid) != 211:
                continue
            if bunch != 0:
                continue

            if abs(dxy) > 2.5:
                continue
            if abs(dz) > 30:
                continue

            # selections
            if pt < 1.5:
                continue
            if abs(eta) > 0.8:
                continue
            ndenom += 1

            if len(tcIdx_isMTVmatch) == 0:
                nineff += 1


                ##
                ## This is where we should take a look at what's going on
                ##

                # print "---------------------------------------------------"

                layers = process_sim_trk_info(isim, event, "Bad ")

                print ievent, isim
                print_sim_trk_info(isim, event, "Bad ")

                if is_not_recoable(layers):
                    nsimfail += 1

                layers_md = process_sim_trk_md_mtv_match_info(isim, event)

                if is_not_recoable(layers_md):
                    nmdfail += 1
                    # print_header()
                    # print_sim_trk_info(isim, event, "Bad ")

                if not is_not_recoable(layers) and not is_not_recoable(layers_md):
                    print "THIS"

                # print_header()
                # print_sim_trk_info(isim, event, "Bad ")
                # print_sim_trk_md_mtv_match_info(isim, event)
                # print_sim_trk_sg_mtv_match_info(isim, event)
                # print_sim_trk_tl_mtv_match_info(isim, event)
                # print_sim_trk_tp_mtv_match_info(isim, event)
                # print_sim_trk_tc_mtv_match_info(isim, event)

            else:
                nnumer += 1

                # print "---------------------------------------------------"

                # print_header()
                # print_sim_trk_info(isim, event, "Good")
                # print_sim_trk_md_mtv_match_info(isim, event)
                # print_sim_trk_sg_mtv_match_info(isim, event)
                # print_sim_trk_tl_mtv_match_info(isim, event)
                # print_sim_trk_tp_mtv_match_info(isim, event)
                # print_sim_trk_tc_mtv_match_info(isim, event)

        if ievent > 9:
            break

    print "nineff  ", nineff
    print "nsimfail", nsimfail
    print "nmdfail ", nmdfail
    print "nnumer  ", nnumer
    print "ndenom  ", ndenom
