#include "TROOT.h"
#include "TTree.h"
#include "TFile.h"
#include "TBranch.h"
#include <iostream>
#include <stdio.h>
#include <vector>
#include <iterator>
using namespace std;

set<int> convertToSet(vector<int> v1)
{
  //Declaring the  set
  set<int> s;
    
  // Traverse the Vector
  for (int x : v1) { 
    // Insert each element
    // into the Set
    s.insert(x);
  }
 
 // Return the resultant Set
  return s;
}
set<int> addToSet(set<int> s,vector<int> v2)
{
  // Traverse the Vector
  for (int x : v2) { 
    // Insert each element
    // into the Set
    s.insert(x);
  }
 
 // Return the resultant Set
  return s;
}

int main(int argc, char** argv){
TFile *infile = TFile::Open("duplicate_removal/pu200_clean_T5_nonAnchor_v2.root");
TTree *t1= (TTree*)infile->Get("tree");

vector<float>* score_rphiA =0;
vector<float>* score_rphiNA =0;
vector<float>* score_rz =0;
vector<vector<int>>* hits = 0;
vector<vector<int>>* sim = 0;

vector<float> * sim_eta =0;
vector<int> * sim_q =0;
vector<int> * sim_event =0;
vector<int> * sim_bunch =0;
vector<int> * sim_parent =0;
vector<float> * simvtx_x =0;
vector<float> * simvtx_y =0;
vector<float> * simvtx_z =0;

t1->SetBranchAddress("t5_score_rphi", & score_rphiA);
t1->SetBranchAddress("t5_score_rphiz", & score_rphiNA);
t1->SetBranchAddress("t5_score_rz", & score_rz);
t1->SetBranchAddress("t5_hitIdxs", &hits);
t1->SetBranchAddress("t5_matched_simIdx", &sim);
t1->SetBranchAddress("sim_eta", &sim_eta);
t1->SetBranchAddress("sim_q", &sim_q);
t1->SetBranchAddress("sim_event", &sim_event);
t1->SetBranchAddress("sim_bunchCrossing", &sim_bunch);
t1->SetBranchAddress("sim_parentVtxIdx", &sim_parent);
t1->SetBranchAddress("simvtx_x", &simvtx_x);
t1->SetBranchAddress("simvtx_y", &simvtx_y);
t1->SetBranchAddress("simvtx_z", &simvtx_z);


struct T5Pair{
  int isim;
  int isFake;
  float rphiA;
  float rphiNA;
  float rphiSUM;
  float rz;
  int nMatched;
};

for( int entry = 0; entry < 30; entry++){
  printf("%d\n",entry);
  t1->GetEntry(entry);
  vector<T5Pair> allpairs;
  allpairs.reserve(10000);
  //vector<int> sim_sel;
  //sim_sel.reserve(5000);
  set<int> sim_sel;
  for( int simid =0; simid<sim_eta->size(); simid++){
    if(sim_bunch->at(simid) !=0){continue;}
    if(sim_event->at(simid) !=0){continue;}
    if(sim_q->at(simid) ==0){continue;}
    if(abs(sim_eta->at(simid)) >2.4 ){continue;}
    int VTX_parent = sim_parent->at(simid);
    if(simvtx_z->at(VTX_parent) >30){continue;}
    float simvtx_xi = simvtx_x->at(VTX_parent);
    float simvtx_yi = simvtx_y->at(VTX_parent);
    if(simvtx_xi*simvtx_xi + simvtx_yi*simvtx_yi > 6.25){continue;}
    //sim_sel.emplace_back(simid);
    sim_sel.insert(simid);
  }
  //sort(sim_sel.begin(),sim_sel.end());
  for(int itrk=0; itrk< score_rphiA->size(); itrk++){
    if(itrk %1000==0){printf("%d/%d\n",itrk,score_rphiA->size());}
    vector<int> ihits = hits->at(itrk);
    set<int> ihitSet = convertToSet(ihits);
    float score_rphiA_i = score_rphiA->at(itrk);
    float score_rphiNA_i = score_rphiNA->at(itrk);
    float score_rz_i = score_rz->at(itrk);
    vector<int> isim_vec = sim->at(itrk);
    bool isFake;
    int isim;
    if(isim_vec.size() ==0){isFake =true; isim = -1;}
    else{ isFake = false; isim = isim_vec.at(0);}



    for(int jtrk=0; jtrk< score_rphiA->size(); jtrk++){
      if(itrk==jtrk){continue;}
      vector<int> jhits = hits->at(jtrk);
      set<int> jhitSet = addToSet(ihitSet,jhits);
      int nMatched = 20-jhitSet.size(); // the set has no duplicates. so a set of hits from both tracks should have 20 elements minus 1 for each duplciate
      float scoreDiff_rphiA = score_rphiA_i - score_rphiA->at(jtrk); 
      float scoreDiff_rphiNA = score_rphiNA_i - score_rphiNA->at(jtrk); 
      float scoreDiff_rphiSUM = (score_rphiA_i+ score_rphiNA_i) - (score_rphiA->at(jtrk)+ score_rphiNA->at(jtrk)); 
      float scoreDiff_rz = score_rz_i - score_rz->at(jtrk); 
      if(nMatched >=7){
      //  printf("%d %d %d %f %f %f %f %d %d\n",itrk, jtrk,nMatched,scoreDiff_rphiA, scoreDiff_rphiNA, scoreDiff_rphiSUM, scoreDiff_rz,isFake,isim);
        T5Pair t5pair = {isim,isFake,scoreDiff_rphiA, scoreDiff_rphiNA, scoreDiff_rphiSUM, scoreDiff_rz,nMatched};
        allpairs.emplace_back(t5pair);
      }
    }
    //for(int ihit=0; ihit<10; ihit++){
    //  printf("%d %d %d\n",itrk, ihit, ihits.at(ihit));
    //}
  }
  
  
  set<int> cut1;
  set<int> allT5sim;
  for(auto pair : allpairs){
      allT5sim.insert(pair.isim);
     if(pair.rphiSUM >0){ 
      cut1.insert(pair.isim);
  //  printf("T5: %d\n",pair.isim);
     }
  }
  set<int> selT5sim;
  set_intersection(sim_sel.begin(), sim_sel.end(),allT5sim.begin(),allT5sim.end(),inserter(selT5sim,selT5sim.end()));

  set<int> finT5sim;
  set_intersection(sim_sel.begin(), sim_sel.end(),cut1.begin(),cut1.end(),inserter(finT5sim,finT5sim.end()));

  set<int> lost_sim;
  set_difference(selT5sim.begin(), selT5sim.end(),finT5sim.begin(),finT5sim.end(),inserter(lost_sim,lost_sim.end()));

  for (auto lost: lost_sim){
    printf("lost trk %d\n",lost);
  }
  //for (auto sim: sim_sel){
  //  printf("sims %d\n",sim);
  //}
  //printf("selected sims %d\n",sim_sel.size());
}
return 0;
}
