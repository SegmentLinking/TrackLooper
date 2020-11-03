// -*- C++ -*-

#ifndef TOOLS_H
#define TOOLS_H

#include <set>

namespace duplicate_removal
{
    struct DorkyEventIdentifier
    {
        // this is a workaround for not having unique event id's in MC
        DorkyEventIdentifier(unsigned long int r, unsigned long int e, unsigned long int l);
        unsigned long int run, event, lumi_section;
        bool operator < (const DorkyEventIdentifier &) const;
        bool operator == (const DorkyEventIdentifier &) const;
    };
    extern std::set<DorkyEventIdentifier> already_seen;
    bool is_duplicate(const DorkyEventIdentifier &id);
    void clear_list();
}

#endif
