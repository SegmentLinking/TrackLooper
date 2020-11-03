#include "dorky.h"
#include <algorithm>
#include <set>

namespace duplicate_removal
{

    DorkyEventIdentifier::DorkyEventIdentifier(unsigned long int r, unsigned long int e, unsigned long int l)
        : run(r),
        event(e),
        lumi_section(l)
    { }

    bool DorkyEventIdentifier::operator < (const DorkyEventIdentifier &other) const
    {
        if (run != other.run)
        { return run < other.run; }
        if (event != other.event)
        { return event < other.event; }
        if (lumi_section != other.lumi_section)
        { return lumi_section < other.lumi_section; }
        return false;
    }

    bool DorkyEventIdentifier::operator == (const DorkyEventIdentifier &other) const
    {
        if (run != other.run)
        { return false; }
        if (event != other.event)
        { return false; }
        if (lumi_section != other.lumi_section)
        { return false; }
        return true;
    }

    std::set<DorkyEventIdentifier> already_seen;
    bool is_duplicate(const DorkyEventIdentifier &id)
    {
        std::pair<std::set<DorkyEventIdentifier>::const_iterator, bool> ret = already_seen.insert(id);
        return !ret.second;
    }

    void clear_list()
    {
        already_seen.clear();
    }

}
