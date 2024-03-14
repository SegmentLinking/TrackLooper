#ifndef process_h
#define process_h

#include "SDL.h"
#include "rooutil.h"
#include "cxxopts.h"


// helper functions
void bookEfficiencySets(std::vector<EfficiencySetDefinition>& effset);
void bookEfficiencySet(EfficiencySetDefinition& effset);
void fillEfficiencySets(std::vector<EfficiencySetDefinition>& effset);
void fillEfficiencySet(int isimtrk, EfficiencySetDefinition& effset, bool excludepT5Found);
void bookFakeRateSets(std::vector<FakeRateSetDefinition>& FRset);
void bookFakeRateSet(FakeRateSetDefinition& FRset);
void fillFakeRateSets(std::vector<FakeRateSetDefinition>& FRset);
void fillFakeRateSet(int isimtrk, FakeRateSetDefinition& FRset);
void bookDuplicateRateSets(std::vector<DuplicateRateSetDefinition>& DLset);
void bookDuplicateRateSet(DuplicateRateSetDefinition& DLset);
void fillDuplicateRateSets(std::vector<DuplicateRateSetDefinition>& DLset);
void fillDuplicateRateSet(int isimtrk, DuplicateRateSetDefinition& DLset);
void createSDLVariables();
void setSDLVariables();
void printSDLVariables();
void printSDLVariablesForATrack(int isimtrk);


#endif
