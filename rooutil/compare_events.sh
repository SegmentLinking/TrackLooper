#!/bin/bash

help() {

    echo "Usage :"
    echo ""
    echo "    sh $0 FILE1 FILE2"
    echo ""
    echo "each file should contain a list of run:lumi:evt events"
    echo ""
    echo ""
    exit

}

if [ -z $1 ]; then help; fi
if [ -z $2 ]; then help; fi 

echo """
//
// Author : Hannsjorg Weber haweber@fnal.gov
//
// I just stole it :P
//


#include <iostream>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <math.h>
#include <vector>
#include <string>
#include <stdlib.h>
#include <stdio.h>
#include \"TMath.h\"

using namespace std;

struct PrintOut{
    int Run;
    int LS;
    unsigned int Evt;
    float leppt;
    float lepminiIso;
    int leppdgid;
    float met;
    float mt;
    int njets;
    int nvetoleps;
    int nbjets;
    int ngenleps;
    float DPhiWlep;
    float htssm;
    float Mlb_cb;
    float Mjjj;
    float mt2w;
    float Topness;
    float chi2;
    bool trackveto;
    bool tauveto;
};
bool SortPrintOut(PrintOut const& lhs, PrintOut const& rhs) {
    if (lhs.Run != rhs.Run)
        return lhs.Run < rhs.Run;
    if (lhs.LS != rhs.LS)
        return lhs.LS < rhs.LS;
    return lhs.Evt < rhs.Evt;
}

void CompareTwoEventListFiles(string fname1, string fname2, bool forcopypaste=false){
    
    cout << \"file1 \" << fname1 << endl;
    cout << \"file2 \" << fname2 << endl;
    
    PrintOut po1; po1.Run = -1;
    vector<PrintOut> f1;
    char buffer1[300];
    ifstream file1(fname1.c_str());
    while( file1.getline(buffer1, 300, '\n') ){
        sscanf(buffer1, \"\t%d:%d:%u\t\",
        &po1.Run, &po1.LS, &po1.Evt);
        if(po1.Run>=1) f1.push_back(po1);
    }
    PrintOut po2; po2.Run = -1;
    vector<PrintOut> f2;
    char buffer2[300];
    ifstream file2(fname2.c_str());
    while( file2.getline(buffer2, 300, '\n') ){
        sscanf(buffer2, \"\t%d:%d:%u\t\",
               &po2.Run, &po2.LS, &po2.Evt);
        if(po2.Run>=1) f2.push_back(po2);
    }

    std::sort(f1.begin(), f1.end(), &SortPrintOut);
    std::sort(f2.begin(), f2.end(), &SortPrintOut);
    
    vector<PrintOut> only1;
    vector<PrintOut> only2;
    vector<PrintOut> same;
    int lpt(0),lid(0), met(0), mt(0), nj(0), nb(0), ngl(0), dphi(0), htssm(0), mlb(0), mjjj(0), mt2w(0), top(0), chi2(0);

    for(unsigned int i = 0; i<f1.size();++i){
        int jind = -1;
        for(unsigned int j = 0; j<f2.size();++j){
            if(f1[i].Run!=f2[j].Run) continue;
            if(f1[i].LS !=f2[j].LS ) continue;
            if(f1[i].Evt!=f2[j].Evt) continue;
            jind = j; break;
        }
        if(jind==-1){
            only1.push_back(f1[i]);
        }
        else same.push_back(f1[i]);
    }
    for(unsigned int i = 0; i<f2.size();++i){
        int jind = -1;
        for(unsigned int j = 0; j<f1.size();++j){
            if(f2[i].Run!=f1[j].Run) continue;
            if(f2[i].LS !=f1[j].LS ) continue;
            if(f2[i].Evt!=f1[j].Evt) continue;
            jind = j; break;
        }
        if(jind==-1){
            only2.push_back(f2[i]);
        }
    }

    
    if(only1.size()>0) cout << \"These events are only in file1\" << endl;
    for(unsigned int i = 0; i<only1.size(); ++i){
        PrintOut p = only1[i];
        if(!forcopypaste) cout << p.Run << \":\" << p.LS << \":\" << p.Evt  << endl;
        else              cout << \"addeventtocheck(e, \" << setw(7) << p.Run << \", \" << setw(5) << p.LS << \", \" << setw(12) << p.Evt << \");\" << endl;
        //cout /*<< \" \" << fixed << setprecision(4)*/ << p.Run << \" \" << setw(4) << p.LS << \" \" << setw(6) << p.Evt  << endl;
    }
    if(only1.size()>0) cout << \"These events are only in file2\" << endl;
    for(unsigned int i = 0; i<only2.size(); ++i){
        PrintOut p = only2[i];
        if(!forcopypaste) cout << p.Run << \":\" << p.LS << \":\" << p.Evt  << endl;
        else              cout << \"addeventtocheck(e, \" << setw(7) << p.Run << \", \" << setw(5) << p.LS << \", \" << setw(12) << p.Evt << \");\" << endl;
        //cout /*<< \" \" << fixed << setprecision(4)*/ << p.Run << \" \" << setw(4) << p.LS << \" \" << setw(6) << p.Evt  << endl;

    }
    //if(same.size()>0) cout << \"These events are in both\" << endl;
    //for(unsigned int i = 0; i<same.size(); ++i){
    //    PrintOut p = same[i];
    //    cout /*<< \" \" << fixed << setprecision(4)*/ << p.Run << \" \" << setw(4) << p.LS << \" \" << setw(6) << p.Evt  << endl;
    //}
    cout << \"There are \" << f1.size() << \" events in file1 \" << fname1 << endl;
    cout << \"There are \" << f2.size() << \" events in file2 \" << fname2 << endl;
    cout << \"Out of these we have \" << same.size() << \" overlapping events\" << endl;
    cout << \"Out of these there are \" << only1.size() << \" events only in file1 \" << fname1 << endl;
    cout << \"Out of these there are \" << only2.size() << \" events only in file2 \" << fname2 << endl;
    cout << \"Check \" << f1.size() << \"+\" << f2.size() << \"=\" << f1.size()+f2.size() << \" should be equal to \" << \"2*\" << same.size() << \"+\" << only1.size() << \"+\" << only2.size() << \"=\" << (2*same.size())+only1.size()+only2.size() << endl;

    if(only1.size()+same.size()!=f1.size()){
        for(unsigned int i = 0; i<f1.size();++i){
            int jind = -1;
            for(unsigned int j = i+1; j<f1.size();++j){
                if(f1[i].Run!=f1[j].Run) continue;
                if(f1[i].LS !=f1[j].LS ) continue;
                if(f1[i].Evt!=f1[j].Evt) continue;
                jind = j; break;
            }
            if(jind!=-1){
                cout << \"Warning, file1 (\" << fname1 << \") has following event duplicated (check your eventlist creation): \" << f1[jind].Run << \":\" << f1[jind].LS << \":\" << f1[jind].Evt << endl;
            }
        }
    }
    if(only2.size()+same.size()!=f2.size()){
        for(unsigned int i = 0; i<f2.size();++i){
            int jind = -1;
            for(unsigned int j = i+1; j<f2.size();++j){
                if(f2[i].Run!=f2[j].Run) continue;
                if(f2[i].LS !=f2[j].LS ) continue;
                if(f2[i].Evt!=f2[j].Evt) continue;
                jind = j; break;
            }
            if(jind!=-1){
                cout << \"Warning, file2 (\" << fname2 << \") has following event duplicated (check your eventlist creation): \" << f2[jind].Run << \":\" << f2[jind].LS << \":\" << f2[jind].Evt << endl;
            }
        }
    }

}
""" > /tmp/CompareTwoEventListFiles.C

root -l -b -q /tmp/CompareTwoEventListFiles.C\(\"$1\",\"$2\"\)
