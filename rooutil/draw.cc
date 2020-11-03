#include "draw.h"

using namespace RooUtil::StringUtil;
using namespace RooUtil;

//==================================================================================================================================
//==================================================================================================================================
//==================================================================================================================================

//##################################################################################################################################
// Parses jsons like the following,
//
//    "histograms":
//    {
//        "MllSS" : { "var" : "MllSS", "bin" : "(25, 0, 250)", "options" : { "doall" : true } },
//        "nb"    : { "var" : "nb"   , "bin" : "( 7, 0,   7)", "options" : { "doall" : true } }
//    }
//
RooUtil::DrawUtil::HistDefs RooUtil::DrawUtil::getHistDefs(json& j)
{
    HistDefs ret;
    for (json::iterator it_hist = j.begin(); it_hist != j.end(); ++it_hist)
    {
        json& histj = it_hist.value();
        if (!histj.count("var"))
        {
            print("ERROR - Did not find 'var' field for this histogram definition");
            std::cout << std::setw(4) << histj << std::endl;
        }
        if (!histj.count("bin"))
        {
            print("ERROR - Did not find 'bin' field for this histogram definition");
            std::cout << std::setw(4) << histj << std::endl;
        }
        TString name = it_hist.key();
        TString var = it_hist.value()["var"];
        TString bin = it_hist.value()["bin"];
        bin = sjoin(bin, " ", "");
        HistDef hist;
        hist.name = name;
        hist.var = var;
        hist.bin = bin;
        ret.push_back(hist);
    }
    return ret;
}

//##################################################################################################################################
// Parses jsons like the following,
//
//    "Analysis" :
//    {
//        "SSPreselection":
//        {
//            "cuts":
//            {
//                "preselections" :
//                [
//                    "ntrk == 0 ^ weight*lepsf*trigsf*purewgt",  <--- Cuts are counted from here. cut0
//                    "pass_offline_trig>0",                                                       cut1
//                    "pass_online_trig>0",                                                        cut2 <-- drop (See below part)
//                    "n_tight_ss_lep==2",                                                         cut3 <-- drop
//                    "n_veto_ss_lep==3",                                                          cut4
//                    "nj>=2",                                                                     cut5
//                    "MllSS > 40."                                                                cut6
//                ]
//            }
//        },
//        "SSee":
//        {
//            "cuts":
//            {
//                "preselections":
//                [
//                    "#SSPreselection%2,3",               <---- NOTE the notations on #, %, and numbers.
//                    "lep_flav_prod_ss==121"                    # means "take from regions with the name matching what follows # up to the end or up to %.
//                ],                                             % means "I am going to exclude some cuts."
//                "selections":                                  , separated numbers mean "Drop cuts matching those indices."
//                [
//                    "nb==0"
//                ]
//            }
//        }
//    }
//
// There is a twist to this that a string pattern match and replace is supported.
// This is because depending on samples sometimes you want to apply slightly different cuts.
// So we provide {pattern} keywords to allow the users to provide a list of std::vector<TString> = {"pattern=blah", "pattern2=blah2"};
// If not all {pattern}'s are resolved it will print out warning but not quit. (In case, users wishes to handle the logic on their own.)
//
RooUtil::DrawUtil::Cuts RooUtil::DrawUtil::getCuts(json& j, std::vector<TString> modifiers)
{
    // Modifying the json based off of modifiers
    std::string origjson_str = j.dump();
    TString origjson = origjson_str;
    TString newjson = format(origjson, modifiers);
    json h = json::parse(newjson.Data());
    Cuts ret;
    RooUtil::DrawExprTool dt;
    dt.setJson(h);
    for (json::iterator it_reg = h.begin(); it_reg != h.end(); ++it_reg)
    {
        TString region = it_reg.key().c_str();
        json g(h[it_reg.key()]);
        if (g.count("skip") && g["skip"])
            continue;
        std::vector<TString> this_reg_draw_sel;
        std::vector<TString> this_reg_draw_wgt;
        std::tie(this_reg_draw_sel, this_reg_draw_wgt) = dt.getFullDrawSelExprsAndWgts(g["cuts"]);
        for (size_t ith = 0; ith < this_reg_draw_sel.size(); ++ith)
        {
            TString sel = this_reg_draw_sel[ith];
            TString wgt = this_reg_draw_wgt[ith];
            Cut cut;
            cut.reg = region;
            cut.idx = ith;
            cut.cut = sel;
            cut.wgt = wgt;
            ret.push_back(cut);
        }
    }
    return ret;
}

//##################################################################################################################################
// From a list of parsed cuts from the Json, replace the {} modifiers
RooUtil::DrawUtil::Cuts RooUtil::DrawUtil::compileCuts(RooUtil::DrawUtil::Cuts cuts, std::vector<TString> modifiers)
{
    Cuts all_cuts;
    for (auto& c : cuts)
    {
        TString reg = c.reg;
        int     idx = c.idx;
        TString cut = c.cut;
        TString wgt = c.wgt;
        reg = format(reg, modifiers);
        cut = format(cut, modifiers);
        wgt = format(wgt, modifiers);
        Cut cutobj;
        cutobj.reg = reg;
        cutobj.idx = idx;
        cutobj.cut = cut;
        cutobj.wgt = wgt;
        all_cuts.push_back(cutobj);
    }
    return all_cuts;
}

//##################################################################################################################################
// From a list of "DrawExpr" (just a struct with three TStrings) perform parallel ttree::draw.
std::map<TString, TH1*> RooUtil::DrawUtil::drawHistograms(TChain* c, RooUtil::DrawUtil::DrawExprs exprs)
{
    std::map<TString, TH1*> ret_hists;
    TMultiDrawTreePlayer* p = RooUtil::FileUtil::createTMulti(c);
    int nentries = c->GetEntries();
    // Sanity check book keeping. Users cannot book SAME HISTOGRAM TWICE.
    std::vector<TString> histnames;
    for (auto& expr : exprs)
    {
        TString cmd = expr.cmd;
        TString cut = expr.cut;
        TString wgt = expr.wgt;
        TString histname = split(split(cmd, ">>")[1], "(")[0];
        if (std::find(histnames.begin(), histnames.end(), histname) != histnames.end())
            error(Form("You have booked same histograms! Please fix this problem. TMultiDrawTreePlayer does not support queuing draw commands with same histogram names! offending histname=%s", histname.Data()));
        else
            histnames.push_back(histname);
        p->queueDraw( cmd.Data(), Form("(%s)*(%s)", cut.Data(), wgt.Data()), "goffe", nentries);
    }
    p->execute();
    for (auto& histname : histnames)
    {
        TH1* h = RooUtil::FileUtil::get(histname);
        if (h) ret_hists[histname] = h;
    }
    return ret_hists;
}

void RooUtil::DrawUtil::printHistDefs(HistDefs histdefs) { for (auto& h : histdefs) h.print(); }
void RooUtil::DrawUtil::printCuts(Cuts cuts) { for (auto& c : cuts) c.print(); }
void RooUtil::DrawUtil::printDrawExprs(DrawExprs exprs) { for (auto& e : exprs) e.print(); }
//==================================================================================================================================
//==================================================================================================================================
//==================================================================================================================================
//==================================================================================================================================

std::map<TString, TH1*> RooUtil::DrawUtil::drawHistograms(TChain* c, json& j, TString prefix, bool nowgt)
{
    std::map<TString, TH1*> ret_hists;
    TMultiDrawTreePlayer* p = RooUtil::FileUtil::createTMulti(c);
    std::vector<TString> cmds;
    std::vector<TString> sels;
    std::vector<TString> wgts;
    std::tie(cmds, sels, wgts) = getDrawExprs(j);
    int nentries = c->GetEntries();
    for (size_t ith = 0; ith < cmds.size(); ++ith)
    {
        TString cmd = Form(cmds[ith].Data(), prefix.Data());
        TString sel = sels[ith].Data();
        TString wgt = nowgt ? "1" : wgts[ith].Data();
        p->queueDraw(
                cmd.Data(),
                Form("(%s)*(%s)", sel.Data(), wgt.Data()),
                "goffe",
                nentries);
    }
    p->execute();
    for (size_t ith = 0; ith < cmds.size(); ++ith)
    {
        TString histname = Form(split(split(cmds[ith], ">>")[1], "(")[0], prefix.Data());
        TH1* h = RooUtil::FileUtil::get(histname);
        if (h)
            ret_hists[histname] = h;
    }
    return ret_hists;
}

RooUtil::DrawExprTool::tripleVecTStr RooUtil::DrawUtil::getDrawExprs(json& j)
{
    RooUtil::DrawExprTool dt;
    dt.setJson(j);
    return dt.getDrawExprTriple();
}

RooUtil::DrawExprTool::tripleVecTStr RooUtil::DrawExprTool::getDrawExprTriple()
{
    std::vector<TString> draw_cmd;
    std::vector<TString> draw_sel;
    std::vector<TString> draw_wgt;
    for (json::iterator it_reg = _j.begin(); it_reg != _j.end(); ++it_reg)
    {
        json g(_j[it_reg.key()]);
        TString reg = it_reg.key().c_str();
        if (!g.count("cuts"))
        {
            print("ERROR - Did not find any cuts field for this json");
            std::cout << std::setw(4) << g << std::endl;
        }
//        if (!g.count("histograms"))
//        {
//            warning("This json has no histograms defined. Seems unusual. Is this correct?");
//            std::cout << std::setw(4) << g << std::endl;
//        }
        std::vector<TString> this_reg_draw_cmd = getFullDrawCmdExprs(g["histograms"]);
        std::vector<TString> this_reg_draw_sel;
        std::vector<TString> this_reg_draw_wgt;
        std::tie(this_reg_draw_sel, this_reg_draw_wgt) = getFullDrawSelExprsAndWgts(g["cuts"]);
        for (auto& cmd : this_reg_draw_cmd)
        {
            for (size_t isel = 0; isel < this_reg_draw_sel.size(); ++isel)
            {
                TString cmd_w_name = Form(cmd.Data(), (TString("%s_") + Form("%s_cut%zu_", reg.Data(), isel)).Data());
                TString sel = this_reg_draw_sel[isel];
                TString wgt = this_reg_draw_wgt[isel];
                draw_cmd.push_back(cmd_w_name);
                draw_sel.push_back(sel);
                draw_wgt.push_back(wgt);
            }
        }
    }
    return std::make_tuple(draw_cmd, draw_sel, draw_wgt);
}

RooUtil::DrawExprTool::pairVecTStr RooUtil::DrawExprTool::getDrawExprPairs()
{
    std::vector<TString> draw_cmd;
    std::vector<TString> draw_sel;
    for (json::iterator it_reg = _j.begin(); it_reg != _j.end(); ++it_reg)
    {
        json g(_j[it_reg.key()]);
        TString reg = it_reg.key().c_str();
        if (!g.count("cuts"))
        {
            print("ERROR - Did not find any cuts field for this json");
            std::cout << std::setw(4) << g << std::endl;
        }
        if (!g.count("histograms"))
        {
            warning("This json has no histograms defined. Seems unusual. Is this correct?");
            std::cout << std::setw(4) << g << std::endl;
        }
        std::vector<TString> this_reg_draw_cmd = getFullDrawCmdExprs(g["histograms"]);
        std::vector<TString> this_reg_draw_sel = getFullDrawSelExprs(g["cuts"]);
        for (auto& cmd : this_reg_draw_cmd)
        {
            for (size_t isel = 0; isel < this_reg_draw_sel.size(); ++isel)
            {
                TString cmd_w_name = Form(cmd.Data(), (TString("%s_") + Form("%s_cut%zu_", reg.Data(), isel)).Data());
                TString sel = this_reg_draw_sel[isel];
                draw_cmd.push_back(cmd_w_name);
                draw_sel.push_back(sel);
            }
        }
    }
    return std::make_tuple(draw_cmd, draw_sel);
}

RooUtil::DrawExprTool::pairVecTStr RooUtil::DrawExprTool::getPairVecTStr(json& j)
{
    std::vector<TString> cuts;
    std::vector<TString> wgts;
    std::vector<TString> ps_strs = j;
    for (auto& str : ps_strs)
    {
        TString tmpstr = str;
        if (str.Contains("#"))
            tmpstr = getExprFromRegion(_j, str);
        std::vector<TString> str_items = split(tmpstr, "^");
        if (str_items.size() != 1 && str_items.size() != 2)
            error(Form("failed to parse selection string = %s", tmpstr.Data()));
        TString cut = str_items[0];
        TString wgt = str_items.size() > 1 ? str_items[1] : "1";
        cuts.push_back(cut);
        wgts.push_back(wgt);
    }
    return std::make_tuple(cuts, wgts);
}

RooUtil::DrawExprTool::pairTStr RooUtil::DrawExprTool::getPairTStr(json& j)
{
    std::vector<TString> cuts;
    std::vector<TString> wgts;
    std::tie(cuts, wgts) = getPairVecTStr(j);
    return std::make_tuple(formexpr(cuts), formexpr(wgts));
}

RooUtil::DrawExprTool::pairTStr RooUtil::DrawExprTool::getPairTStrFromRegion(json& j, TString region, std::vector<int> exclude)
{
    json& regj = j[region.Data()];
    if (!regj.count("cuts"))
    {
        print("ERROR - Did not find any cuts field for this json");
        std::cout << std::setw(4) << regj << std::endl;
    }
    json& cutj = regj["cuts"];
    std::vector<TString> tmp_all_selections;
    if (cutj.count("preselections"))
    {
        std::vector<TString> v = cutj["preselections"];
        tmp_all_selections.insert(tmp_all_selections.end(), v.begin(), v.end());
    }
    if (cutj.count("selections"))
    {
        std::vector<TString> v = cutj["selections"];
        tmp_all_selections.insert(tmp_all_selections.end(), v.begin(), v.end());
    }

    std::vector<TString> all_selections;
    for (size_t idx = 0; idx < tmp_all_selections.size(); ++idx)
    {
        if (std::find(exclude.begin(), exclude.end(), idx) != exclude.end())
            continue;
        all_selections.push_back(tmp_all_selections.at(idx));
    }

    json tmp(all_selections);
    return getPairTStr(tmp);
}

RooUtil::DrawExprTool::pairTStr RooUtil::DrawExprTool::getPairTStrFromRegionFromExpr(json& j, TString expr)
{
    expr.ReplaceAll("#", "");
    std::vector<TString> expr_tokens = split(expr, "%");
    if (expr_tokens.size() != 1 && expr_tokens.size() != 2)
        error(Form("failed to parse selection string = %s", expr.Data()));
    TString region = expr_tokens[0];
    TString exclude_str = expr_tokens.size() > 1 ? expr_tokens[1] : "";
    std::vector<int> exclude;
    for (auto& elem : split(exclude_str, ","))
    {
        if (!elem.IsNull())
            exclude.push_back(elem.Atoi());
    }
    return getPairTStrFromRegion(j, region, exclude);
}

TString RooUtil::DrawExprTool::getExprFromRegion(json& j, TString expr)
{
    pairTStr p = getPairTStrFromRegionFromExpr(j, expr);
    return Form("%s ^ %s", std::get<0>(p).Data(), std::get<1>(p).Data());
}

RooUtil::DrawExprTool::pairVecTStr RooUtil::DrawExprTool::getFullDrawSelExprsAndWgts(json& j)
{
    // Parse the "selections" data in a given region json.
    std::vector<TString> individ_selections;
    std::vector<TString> individ_selections_weights;
    if (j.count("selections"))
        std::tie(individ_selections, individ_selections_weights) = getPairVecTStr(j["selections"]);

    // Parse the "preselections" data in a given region json.
    TString preselection = "1";
    TString preselection_weight = "1";
    if (j.count("preselections"))
        std::tie(preselection, preselection_weight) = getPairTStr(j["preselections"]);

    // Pre-pend the preselections to the selections
    individ_selections.insert(individ_selections.begin(), preselection);
    individ_selections_weights.insert(individ_selections_weights.begin(), preselection_weight);

    // Now collapse the selections into a list of selections
    std::vector<TString> selections;
    std::vector<TString> selections_weights;
    for (size_t isel = 0; isel < individ_selections.size(); ++isel)
    {
        std::vector<TString> cutexpr(individ_selections.begin(), individ_selections.begin() + isel + 1);
        std::vector<TString> wgtexpr(individ_selections_weights.begin(), individ_selections_weights.begin() + isel + 1);
        selections.push_back(formexpr(cutexpr));
        selections_weights.push_back(formexpr(wgtexpr));
    }
    return std::make_tuple(selections, selections_weights);
}

std::vector<TString> RooUtil::DrawExprTool::getFullDrawSelExprs(json& j)
{
    std::vector<TString> draw_sel;

    // Parse the "selections" data in a given region json.
    std::vector<TString> individ_selections;
    std::vector<TString> individ_selections_weights;
    if (j.count("selections"))
        std::tie(individ_selections, individ_selections_weights) = getPairVecTStr(j["selections"]);

    // Parse the "preselections" data in a given region json.
    TString preselection = "1";
    TString preselection_weight = "1";
    if (j.count("preselections"))
        std::tie(preselection, preselection_weight) = getPairTStr(j["preselections"]);

    // Pre-pend the preselections to the selections
    individ_selections.insert(individ_selections.begin(), preselection);
    individ_selections_weights.insert(individ_selections_weights.begin(), preselection_weight);

    // Now collapse the selections into a list of selections
    std::vector<TString> selections;
    std::vector<TString> selections_weights;
    for (size_t isel = 0; isel < individ_selections.size(); ++isel)
    {
        std::vector<TString> cutexpr(individ_selections.begin(), individ_selections.begin() + isel + 1);
        std::vector<TString> wgtexpr(individ_selections_weights.begin(), individ_selections_weights.begin() + isel + 1);
        TString full_draw_sel_expr = Form("(%s)*(%s)", formexpr(cutexpr).Data(), formexpr(wgtexpr).Data());
        full_draw_sel_expr = cleanparantheses(full_draw_sel_expr);
        draw_sel.push_back(full_draw_sel_expr);
    }

    return draw_sel;
}

std::vector<TString> RooUtil::DrawExprTool::getFullDrawCmdExprs(json& j)
{
    std::vector<TString> draw_cmd;
    draw_cmd.push_back("0>>%scount(1,0,1)");
    for (json::iterator it_hist = j.begin(); it_hist != j.end(); ++it_hist)
    {
        json& histj = it_hist.value();
        if (!histj.count("var"))
        {
            print("ERROR - Did not find 'var' field for this histogram definition");
            std::cout << std::setw(4) << histj << std::endl;
        }
        if (!histj.count("bin"))
        {
            print("ERROR - Did not find 'bin' field for this histogram definition");
            std::cout << std::setw(4) << histj << std::endl;
        }
        TString name = it_hist.key();
        TString var = it_hist.value()["var"];
        TString bin = it_hist.value()["bin"];
        TString cmd = Form("%s>>", var.Data()) + TString("%s") + Form("%s", name.Data()) + Form("%s", bin.Data());
        cmd = sjoin(cmd, " ", "");
        draw_cmd.push_back(cmd);
    }
    return draw_cmd;
}

