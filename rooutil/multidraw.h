// @(#)root/treeplayer:$Id$
// Author: Rene Brun   12/01/96

/*************************************************************************
 * Copyright (C) 1995-2000, Rene Brun and Fons Rademakers.               *
 * All rights reserved.                                                  *
 *                                                                       *
 * For the licensing terms see $ROOTSYS/LICENSE.                         *
 * For the list of contributors see $ROOTSYS/README/CREDITS.             *
 *************************************************************************/

#ifndef ROOT_TMultiDrawTreePlayer
#define ROOT_TMultiDrawTreePlayer


//////////////////////////////////////////////////////////////////////////
//                                                                      //
// TMultiDrawTreePlayer                                                 //
//                                                                      //
// A TTree object is a list of TBranch.                                 //
//   To Create a TTree object one must:                                 //
//    - Create the TTree header via the TTree constructor               //
//    - Call the TBranch constructor for every branch.                  //
//                                                                      //
//   To Fill this object, use member function Fill with no parameters.  //
//     The Fill function loops on all defined TBranch.                  //
//                                                                      //
//////////////////////////////////////////////////////////////////////////

#ifndef ROOT_TTreePlayer
#include "TTreePlayer.h"
#endif

#include "multiselect.h"

#include <TTree.h>
#include <TVirtualTreePlayer.h>

#include <memory>
#include <unordered_map>
#include <iostream>
#include <unistd.h>
#include <chrono>
#include <ctime>
#include <numeric>

class TVirtualIndex;

struct DrawData {
    std::shared_ptr<TSelectorDraw> selector;
    std::shared_ptr<TList>         input;

    Long64_t    firstentry;
    Long64_t    nentries;
    std::string options;
};

class NotifyProxier: public TObject {
    public:
        NotifyProxier(const std::vector<DrawData>& draws):
            m_draws(draws) {
                // Empty
            }

        virtual Bool_t Notify() override {
            bool ret = true;
            for (auto& draw: m_draws) {
                ret &= draw.selector->Notify();
            }

            return ret;
        }

    private:
        const std::vector<DrawData>& m_draws;
};

class TMultiDrawTreePlayer: public TTreePlayer {

private:
   TMultiDrawTreePlayer(const TMultiDrawTreePlayer &);
   TMultiDrawTreePlayer& operator=(const TMultiDrawTreePlayer &);

protected:
   std::vector<DrawData> m_draws;

public:
   TMultiDrawTreePlayer();
   virtual ~TMultiDrawTreePlayer();

   virtual bool queueDraw(const char* varexp, const char* selection, Option_t *option = "", Long64_t nentries = 1000000000, Long64_t firstentry = 0);
   virtual bool execute();

};

#endif

