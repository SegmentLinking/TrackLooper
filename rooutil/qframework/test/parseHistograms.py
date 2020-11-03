#!/bin/env python

def main():
    job = TQHistoMakerAnalysisJob()

    lines = [
        'TH1F("testhisto","",10,0,10) << ("some variable" : "some title");',
        'TH1F("binlabels","",10,0,10) << ("some variable" : "some title" : {"bin1","bin2","bin3"} );'
    ]
    
    for line in lines:
        if not "TH" in line:
            continue
            job.bookHistogram(line)
    job.printBooking("test")


if __name__ == "__main__":
    from QFramework import *
    main()
