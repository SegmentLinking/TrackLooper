#ifndef build_module_map_h
#define build_module_map_h

#include <iostream>
#include "AnalysisConfig.h"
#include "SDL/Module.h"
#include <cppitertools/itertools.hpp>
#include "trkCore.h"

void build_module_map();
void printModuleConnections(std::ofstream& ostrm);

#endif
