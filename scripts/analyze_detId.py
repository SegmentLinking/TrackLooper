#!/bin/env python

def bitwise_and(b1, b2):

    if len(b1) != len(b2): print "Error: length does not match"
    if len(b1) != 29:      print "Error: b1 length not 29"
    if len(b2) != 29:      print "Error: b1 length not 29"

    newbin = []
    for i, j in zip(b1, b2):
        newbin.append(str(int(i) * int(j)))

    return "".join(newbin)

if __name__ == "__main__":


    f = open("detId_to_seqId.txt")

    for index, line in enumerate(f.readlines()):

        if index == 0: continue

        detId = line.strip().split()[0]
        seqId = line.strip().split()[1]

        # detId = 0-th index from "binary" 
        # seqId = ph2_layer, ph2_order, ph2_ring, ph2_rod, ph2_subdet, ph2_side, ph2_module, ph2_isLower
        # "binary"  idx : 1          2         3        4           5         6           7            8
        detIdIdx   = 0
        layerIdx   = 1
        orderIdx   = 2
        ringIdx    = 3
        rodIdx     = 4
        subdetIdx  = 5
        sideIdx    = 6
        moduleIdx  = 7
        isLowerIdx = 8

        binary = []
        binary.append("{0:b}".format(int(detId)))
        for item in seqId.split(","):
            binary.append("{0:b}".format(int(item)))

        # 29 bits total
        # 00 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28
        #  x  x  x  x  x  x  x  x  x  x  x  x  x  x  x  x  x  x  x  x  x  x  x  x  x  x  x  x  x
        #    -subdet-       -layer--          ------ph2_rod-------    -----ph2_module-----
        #             -side       --layer-       -ph2_ring--  0  1    -----ph2_module-----
        detId_binary = binary[detIdIdx]

        print " ".join(binary)

        # ph2_subdet 3 bits [1:1+3]
        if detId_binary[1:1+3] != "{:s}".format(binary[subdetIdx].rjust(3, '0')):
            print "Error: parsed ph2_subdet from detId does not match what is in tracking ntuple"
            # This also determines "order" which doesn't seem necessary

        # ph2_module 7 bits from 20 to 26th bits
        if detId_binary[20:20+7] != "{:s}".format(binary[moduleIdx].rjust(7, '0')):
            print "Error: parsed ph2_module from detId does not match what is in tracking ntuple"

        if binary[subdetIdx] == "101":
            # ph2_rod 7 bits from 12 to 18
            if detId_binary[12:12+7] != "{:s}".format(binary[rodIdx].rjust(7, '0')):
                print "Error: parsed ph2_rod from detId does not match what is in tracking ntuple", detId_binary[12:12+7], binary[rodIdx].rjust(7, '0')
            # ph2_layer 3 bits from 6 to 8
            if detId_binary[6:6+3] != "{:s}".format(binary[layerIdx].rjust(3, '0')):
                print "Error: parsed ph2_layer from detId does not match what is in tracking ntuple", detId_binary[6:6+3], binary[layerIdx].rjust(3, '0')
        elif binary[subdetIdx] == "100":
            # ph2_ring 4 bits from 12 to 18
            if detId_binary[13:13+4] != "{:s}".format(binary[ringIdx].rjust(4, '0')):
                print "Error: parsed ph2_ring from detId does not match what is in tracking ntuple", detId_binary[13:13+4], binary[ringIdx].rjust(4, '0')
            # ph2_layer 3 bits from 8 to 10
            if detId_binary[8:8+3] != "{:s}".format(binary[layerIdx].rjust(3, '0')):
                print "Error: parsed ph2_layer from detId does not match what is in tracking ntuple", detId_binary[8:8+3], binary[layerIdx].rjust(3, '0')
            # ph2_side 2 bits from 4 to 5
            if detId_binary[4:4+2] != "{:s}".format(binary[sideIdx].rjust(2, '0')):
                print "Error: parsed ph2_side from detId does not match what is in tracking ntuple", detId_binary[4:4+2], binary[sideIdx].rjust(2, '0')
        else:
            print "Error: unrecognized subdet", binary[subdetIdx]

        # ph2_isLower last bit
        if detId_binary[28] != binary[isLowerIdx]:
            print "Error: parsed ph2_isLower from detId does not match what is in tracking ntuple", detId_binary[28], binary[isLowerIdx]







