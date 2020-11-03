#!/usr/local/bin/python2.7

import os
import sys


if __name__ == '__main__':


    # # with open("/Users/namin/sandbox/itermtest/blah.txt", "a") as fhout:
    # #     fhout.write(sys.stdin.read().strip())
    # #     fhout.write("\n")
    # #     fhout.write("here")
    # #     fhout.write("\n")
    # print sys.argv[-1]

    # sys.exit(0)

    # # path = sys.stdin.readline().split("MY_"+"IMGCAT")[1].split()[0]
    # # print "ls {}".format(path)
    path = sys.argv[-1]
    if "/" in path:
        ext = path.rsplit(".",1)[-1]
        tmp = "/Users/namin/.temp.{}".format(ext)
        os.system("scp -q ucsd:{} {}".format(path,tmp))
        os.system("open -g -a Preview {}".format(tmp))
    sys.exit(0)
