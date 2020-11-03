#!/bin/env python

#// Decoding DetId
#//
#// detId comes in 29 bits. There are two formats depending on which sub detector it is.
#//
#// 29 bits total
#//
#// left to right index (useful python, i.e. string[idx:jdx])
#// 00 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28
#//
#// right to left index (useful when C++ style, i.e. bit shifting)
#// 28 27 26 25 24 23 22 21 20 19 18 17 16 15 14 13 12 11 10 09 08 07 06 05 04 03 02 01 00
#//
#//  x  x  x  x  x  x  x  x  x  x  x  x  x  x  x  x  x  x  x  x  x  x  x  x  x  x  x  x  x
#//
#//    -subdet-       -layer-- -side    --------rod---------    -------module-------        # if subdet == 5
#//    -subdet- -side       --layer-       ----ring---          -------module-------        # if subdet == 4
#//
#//

def getDetId(
        subdet,
        side,
        layer,
        rod,
        ring,
        module,
        isLower):

    # barrel
    if subdet == 5:

        # left to right index (useful python, i.e. string[idx:jdx])
        # 00 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28
        #  x  x  x  x  x  x  x  x  x  x  x  x  x  x  x  x  x  x  x  x  x  x  x  x  x  x  x  x  x
        #    -subdet-       -layer-- -side    --------rod---------    -------module-------        # if subdet == 5
        return "1{0:03b}00{1:03b}{2:02b}0{3:07b}0{4:07b}{5:02b}".format(subdet, layer, side, rod, module, isLower+1)

    else:

        # left to right index (useful python, i.e. string[idx:jdx])
        # 00 01 02 03 04 05 06 07 08 09 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28
        #  x  x  x  x  x  x  x  x  x  x  x  x  x  x  x  x  x  x  x  x  x  x  x  x  x  x  x  x  x
        #    -subdet- -side       --layer-       ----ring---          -------module-------        # if subdet == 4
        return "1{0:03b}{1:02b}00{2:03b}00{3:04b}000{4:07b}{5:02b}".format(subdet, side, layer, ring, module, isLower+1)


if __name__ == "__main__":

    # print getDetId(5, 3, 1, 1, 0, 1, 1)
    # print getDetId(5, 3, 1, 1, 0, 1, 0)
    print int(getDetId(5, 3, 1, 1, 0, 1, 1), 2)
    print int(getDetId(5, 3, 1, 1, 0, 1, 0), 2)

    # ring 14 has 72 modules
    for i in xrange(1, 73):
        # print getDetId(4, 1, 2, 0, 14, i, 1)
        # print getDetId(4, 1, 2, 0, 14, i, 0)
        print int(getDetId(4, 1, 2, 0, 14, i, 1), 2)
        print int(getDetId(4, 1, 2, 0, 14, i, 0), 2)
        # print getDetId(4, 3, 2, 0, 14, i, 1)
        # print getDetId(4, 3, 2, 0, 14, i, 0)
        print int(getDetId(4, 3, 2, 0, 14, i, 1), 2)
        print int(getDetId(4, 3, 2, 0, 14, i, 0), 2)
