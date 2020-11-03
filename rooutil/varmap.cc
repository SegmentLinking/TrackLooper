//  .
// ..: P. Chang, philip@physics.ucsd.edu

#include "varmap.h"

///////////////////////////////////////////////////////////////////////////////////////////////////
//
//
// Variable Map class
//
//
///////////////////////////////////////////////////////////////////////////////////////////////////

//_________________________________________________________________________________________________
RooUtil::VarMap::VarMap()
{
}

//_________________________________________________________________________________________________
RooUtil::VarMap::VarMap( TString filename, TString delim, int nkeys )
{
    load(filename, delim, nkeys);
}

//_________________________________________________________________________________________________
RooUtil::VarMap::~VarMap() {}

//_________________________________________________________________________________________________
void RooUtil::VarMap::load( TString filename, TString delim, int nkeys )
{
    varmap_.clear();
    ifstream ifile;
    ifile.open( filename.Data() );
    std::string line;
    filename_ = filename;

    while ( std::getline( ifile, line ) )
    {
        TString rawline = line;
        std::vector<TString> list = RooUtil::StringUtil::split(rawline, delim);
        // If it has # at the front skip the event
        if (list.size() > 0)
            if (list[0].Contains("#"))
                continue;
        std::vector<int> keys;
        for (unsigned int ii = 0; (int) ii < nkeys; ++ii)
            keys.push_back( list[ii].Atoi() );
        std::vector<float> data;
        for (unsigned int ii = nkeys; ii < list.size(); ++ii)
            data.push_back( list[ii].Atof() );
        varmap_[keys] = data;
    }
}

//_________________________________________________________________________________________________
std::vector<float> RooUtil::VarMap::get( std::vector<int> key )
{
    try
    {
        return varmap_.at(key);
    }
    catch (const std::out_of_range& oor)
    {
        std::cout << "Key not found in map: " << filename_ << std::endl;
        std::cout << "keys:" << std::endl;
        for (auto& k : key)
            std::cout << k <<  std::endl;
        exit(-1);
    }
}
