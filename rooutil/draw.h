// I wish these were more customizable. But unforutnately, I do use some hardcoded values. So may not be completely ideal, but it is what it is.

#ifndef draw_h
#define draw_h

#include <vector>
#include <tuple>
#include <iostream>
#include <fstream>
#include <algorithm>
#include <utility>

#include "TString.h"

#include "json.h"
#include "stringutil.h"
#include "printutil.h"
#include "multidraw.h"
#include "fileutil.h"

using json = nlohmann::json;

namespace RooUtil
{
    class DrawExprTool
    {
        public:
            typedef std::tuple<std::vector<TString>, std::vector<TString>, std::vector<TString>> tripleVecTStr;
            typedef std::tuple<std::vector<TString>, std::vector<TString>> pairVecTStr;
            typedef std::tuple<TString, TString> pairTStr;

            json _j;

            void setJson(json j) { _j = j; }
            tripleVecTStr getDrawExprTriple();
            pairVecTStr getDrawExprPairs();
            pairVecTStr getPairVecTStr(json& j);
            pairTStr getPairTStr(json& j);
            pairTStr getPairTStrFromRegion(json& j, TString region, std::vector<int> exclude=std::vector<int>());
            pairTStr getPairTStrFromRegionFromExpr(json& j, TString expr);
            TString getExprFromRegion(json& j, TString expr);
            std::vector<TString> getFullDrawSelExprs(json& j);
            pairVecTStr getFullDrawSelExprsAndWgts(json& j);
            std::vector<TString> getFullDrawCmdExprs(json& j);
    };

    namespace DrawUtil
    {
        struct HistDef
        {
            TString name;
            TString var;
            TString bin;
            void print()
            {
                std::cout <<  " name: " << name <<  " var: " << var <<  " bin: " << bin <<  std::endl;
            }
        };
        struct Cut
        {
            TString reg;
            int idx;
            TString cut;
            TString wgt;
            void print()
            {
                std::cout <<  " reg: " << reg <<  " idx: " << idx <<  " cut: " << RooUtil::StringUtil::cleanparantheses(cut) <<  " wgt: " << wgt <<  std::endl;
            }
        };
        struct DrawExpr
        {
            TString cmd;
            TString cut;
            TString wgt;
            void print()
            {
                std::cout <<  " cmd: " << cmd <<  " cut: " << cut <<  " wgt: " << wgt <<  std::endl;
            }
        };
        typedef std::vector<HistDef> HistDefs;
        typedef std::vector<Cut> Cuts;
        typedef std::vector<DrawExpr> DrawExprs;
        // =========================================================================================================
        HistDefs getHistDefs(json& j);
        Cuts getCuts(json& j, std::vector<TString> a=std::vector<TString>());
        Cuts compileCuts(Cuts, std::vector<TString> a=std::vector<TString>());
        void printHistDefs(HistDefs histdefs);
        void printCuts(Cuts cuts);
        void printDrawExprs(DrawExprs exprs);
        std::map<TString, TH1*> drawHistograms(TChain* c, DrawExprs exprs);
        // =========================================================================================================
        DrawExprTool::tripleVecTStr getDrawExprs(json& j);
        std::map<TString, TH1*> drawHistograms(TChain*, json&, TString="", bool=false);
    }
}

#endif
