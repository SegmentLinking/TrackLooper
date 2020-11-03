//  .
// ..: P. Chang, philip@physics.ucsd.edu

#include "cuttool.h"

//____________________________________________________________________________________________________________
bool RooUtil::CutTool::passesCut(TString name, std::function<bool()> cut, std::vector<TString> nm)
{
    // The N-m cut string vector has non-zero elements
    // So check whether the one being evaluated is a part of it and if so, skip this (i.e. return true)
    if (nm.size() != 0)
    {
        if (std::find(nm.begin(), nm.end(), name) != nm.end())
            return true;
    }

    // Try inserting the value to see whether a key already exists
    std::pair<std::map<TString, bool>::iterator, bool> ret;
    ret = cache.insert( {name, 0} );

    // Never evaluated, so evaluate
    if ( ret.second == true )
        cache[name] = cut();

    // return the stored value or the one just evaluated
    return cache[name];
}

//____________________________________________________________________________________________________________
bool RooUtil::CutTool::passesCut(TString name)
{
    return cache.at(name);
}
