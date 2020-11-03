#include "tmvautil.h"
#include "stringutil.h"

using namespace std;

//_________________________________________________________________________________________________
TMVA::Reader* RooUtil::TMVAUtil::createReader(TString methodType, TString xmlpath)
{

    TMVA::Reader* reader = new TMVA::Reader("!Silent:!Color");

    // hadoopmap
    ifstream infile(xmlpath.Data());

    if (infile.good())
    {
        ifstream xmlfile;
        xmlfile.open(xmlpath.Data());

        std::string line;
//        int nvar = -1;

        while ( std::getline( xmlfile, line ) ) 
        {

//            if (TString(line.c_str()).Contains("NVar"))
//            {
//                nvar = RooUtil::StringUtil::split(TString(line.c_str()),"\"")[1].Atoi();
//            }

            if (TString(line.c_str()).Contains("Variable VarIndex"))
            {
                RooUtil::StringUtil::vecTString vs = RooUtil::StringUtil::split(TString(line.c_str()), " ");
                TString varname;
                TString vartype;
                for (auto& s : vs)
                {
                    if (s.Contains("Expression"))
                    {
                        varname = RooUtil::StringUtil::split(s, "=")[1].ReplaceAll("\"","");
                    }
                    if (s.Contains("Type"))
                    {
                        vartype = RooUtil::StringUtil::split(s, "=")[1].ReplaceAll("\"","");
                    }
                }
                reader->AddVariable(varname, (float*) 0x0);
            }

        }
    }

    reader->BookMVA(methodType, xmlpath);

    return reader;
}

//_________________________________________________________________________________________________
std::vector<float> RooUtil::TMVAUtil::getInputValues(TMVA::Reader* reader, RooUtil::TTreeX& tx)
{

    std::vector<float> rtn;
    for (auto& vi : reader->DataInfo().GetVariableInfos())
    {
        TString name = vi.GetExpression();
        if (vi.GetVarType() == 'F') rtn.push_back(tx.getBranch<float>(name));
        if (vi.GetVarType() == 'I') rtn.push_back(tx.getBranch<int>(name));
    }

    return rtn;
}

//_________________________________________________________________________________________________
RooUtil::TMVAUtil::ReaderX::ReaderX(TString methodType_, TString xmlpath)
{
    reader = RooUtil::TMVAUtil::createReader(methodType_, xmlpath);
    methodType = methodType_;
}

//_________________________________________________________________________________________________
float RooUtil::TMVAUtil::ReaderX::eval(RooUtil::TTreeX& tx)
{
    return reader->EvaluateMVA(RooUtil::TMVAUtil::getInputValues(this->reader, tx), this->methodType);
}
