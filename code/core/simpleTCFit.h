#ifndef simpleTCFit_h
#define simpleTCFit_h

#include <vector>
#include <tuple>
#include <math.h>

#include "trktree.h"
#include "MathUtil.h"

#include "TGraph.h"
#include "TF1.h"
#include "TCanvas.h"

std::tuple<float, float, float, float, float, int> simpleTCFit(
    std::vector<unsigned int> hit_idx, 
    std::vector<unsigned int> hit_type,
    int itc=0);

#endif
