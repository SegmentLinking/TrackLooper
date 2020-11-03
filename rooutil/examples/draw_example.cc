#include "draw.h"
#include "json.h"
#include <iostream>
#include <fstream>

int main()
{
    std::ifstream i(".draw_example.json");
    json j;
    i >> j;

    std::vector<TString> draw_cmd;
    std::vector<TString> draw_sel;
    RooUtil::DrawExprTool dt;
    dt.setJson(j["Analysis1"]);
    std::cout << std::setw(4) << j["Analysis1"] << std::endl;
    std::tie(draw_cmd, draw_sel) = dt.getDrawExprPairs();

    for (size_t idraw = 0; idraw < draw_cmd.size(); ++idraw) std::cout << draw_cmd[idraw] << " " << draw_sel[idraw] << std::endl;
}
