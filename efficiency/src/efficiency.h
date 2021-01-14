#ifndef process_h
#define process_h

#include "SDL.h"
#include "rooutil.h"
#include "cxxopts.h"
#include "helper.h"


// helper functions
void bookEfficiencySets(std::vector<EfficiencySetDefinition>& effset);
void bookEfficiencySet(EfficiencySetDefinition& effset);
void fillEfficiencySets(std::vector<EfficiencySetDefinition>& effset);
void fillEfficiencySet(int isimtrk, EfficiencySetDefinition& effset);
void createSDLVariables();
void setSDLVariables();
void printSDLVariables();
void printSDLVariablesForATrack(int isimtrk);


#endif
