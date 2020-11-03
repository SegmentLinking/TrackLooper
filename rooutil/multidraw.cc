#include "multidraw.h"
#include "printutil.h"

#include "TROOT.h"
#include "TSystem.h"
#include "TFile.h"
#include "TEntryList.h"
#include "TPolyMarker3D.h"
#include "TText.h"
#include "TVirtualPad.h"
#include "TTreeFormula.h"
#include "TStyle.h"
#include "TH1.h"
#include "TVirtualMonitoring.h"
#include "TTreeCache.h"

#include <limits>

//ClassImp(TMultiDrawTreePlayer)

TMultiDrawTreePlayer::TMultiDrawTreePlayer():
    TTreePlayer() {
    // Empty
}

TMultiDrawTreePlayer::~TMultiDrawTreePlayer() {
    // Empty
}

bool TMultiDrawTreePlayer::queueDraw(const char* varexp0, const char* selection, Option_t *option, Long64_t nentries, Long64_t firstentry) {
   // Draw expression varexp for specified entries that matches the selection.
   // Returns -1 in case of error or number of selected events in case of succss.
   //
   // See the documentation of TTree::Draw for the complete details.

   if (fTree->GetEntriesFriend() == 0) return false;

   // Let's see if we have a filename as arguments instead of
   // a TTreeFormula expression.

   TString possibleFilename = varexp0;
   Ssiz_t dot_pos = possibleFilename.Last('.');
   if ( dot_pos != kNPOS
       && possibleFilename.Index("Alt$")<0 && possibleFilename.Index("Entries$")<0
       && possibleFilename.Index("Length$")<0  && possibleFilename.Index("Entry$")<0
       && possibleFilename.Index("LocalEntry$")<0
       && possibleFilename.Index("Min$")<0 && possibleFilename.Index("Max$")<0
       && possibleFilename.Index("MinIf$")<0 && possibleFilename.Index("MaxIf$")<0
       && possibleFilename.Index("Iteration$")<0 && possibleFilename.Index("Sum$")<0
       && possibleFilename.Index(">")<0 && possibleFilename.Index("<")<0
       && gSystem->IsFileInIncludePath(possibleFilename.Data())) {

      if (selection && strlen(selection) && !gSystem->IsFileInIncludePath(selection)) {
         Error("DrawSelect",
               "Drawing using a C++ file currently requires that both the expression and the selection are files\n\t\"%s\" is not a file",
               selection);
         return false;
      }
      return DrawScript("generatedSel",varexp0,selection,option,nentries,firstentry);

   } else {
      possibleFilename = selection;
      if (possibleFilename.Index("Alt$")<0 && possibleFilename.Index("Entries$")<0
          && possibleFilename.Index("Length$")<0  && possibleFilename.Index("Entry$")<0
          && possibleFilename.Index("LocalEntry$")<0
          && possibleFilename.Index("Min$")<0 && possibleFilename.Index("Max$")<0
          && possibleFilename.Index("MinIf$")<0 && possibleFilename.Index("MaxIf$")<0
          && possibleFilename.Index("Iteration$")<0 && possibleFilename.Index("Sum$")<0
          && possibleFilename.Index(">")<0 && possibleFilename.Index("<")<0
          && gSystem->IsFileInIncludePath(possibleFilename.Data())) {

         Error("DrawSelect",
               "Drawing using a C++ file currently requires that both the expression and the selection are files\n\t\"%s\" is not a file",
               varexp0);
         return false;
      }
   }

   Long64_t oldEstimate  = fTree->GetEstimate();
   TEventList *evlist  = fTree->GetEventList();
   TEntryList *elist = fTree->GetEntryList();
   if (evlist && elist){
      elist->SetBit(kCanDelete, kTRUE);
   }

   DrawData data;
   data.input.reset(new TList());
   data.input->Add(new TNamed("varexp",""));
   data.input->Add(new TNamed("selection",""));

   TNamed *cvarexp    = (TNamed*) data.input->FindObject("varexp");
   TNamed *cselection = (TNamed*) data.input->FindObject("selection");

   if (cvarexp) cvarexp->SetTitle(varexp0);
   if (cselection) cselection->SetTitle(selection);


   TString opt = option;
   opt.ToLower();
   Bool_t optpara   = kFALSE;
   Bool_t optcandle = kFALSE;
   Bool_t optgl5d   = kFALSE;
   Bool_t optnorm   = kFALSE;
   if (opt.Contains("norm")) {optnorm = kTRUE; opt.ReplaceAll("norm",""); opt.ReplaceAll(" ","");}
   if (opt.Contains("para")) optpara = kTRUE;
   if (opt.Contains("candle")) optcandle = kTRUE;
   if (opt.Contains("gl5d")) optgl5d = kTRUE;
   Bool_t pgl = gStyle->GetCanvasPreferGL();
   if (optgl5d) {
      fTree->SetEstimate(fTree->GetEntries());
      if (!gPad) {
         if (pgl == kFALSE) gStyle->SetCanvasPreferGL(kTRUE);
         gROOT->ProcessLineFast("new TCanvas();");
      }
   }


   // Do not process more than fMaxEntryLoop entries
   if (nentries > fTree->GetMaxEntryLoop()) nentries = fTree->GetMaxEntryLoop();

   data.nentries = GetEntriesToProcess(firstentry, nentries);
   data.firstentry = firstentry;
   data.options = option;

   data.selector.reset(new TSelectorMultiDraw());
   data.selector->SetInputList(data.input.get());

   m_draws.push_back(data);

   return true;

   // invoke the selector
   Long64_t nrows = Process(fSelector,option,nentries,firstentry);
   fSelectedRows = nrows;
   fDimension = fSelector->GetDimension();

   //*-* an Event List
   if (fDimension <= 0) {
      fTree->SetEstimate(oldEstimate);
      if (fSelector->GetCleanElist()) {
         // We are in the case where the input list was reset!
         fTree->SetEntryList(elist);
         delete fSelector->GetObject();
      }
      return true;
   }

   // Draw generated histogram
   Long64_t drawflag = fSelector->GetDrawFlag();
   Int_t action   = fSelector->GetAction();
   Bool_t draw = kFALSE;
   if (!drawflag && !opt.Contains("goff")) draw = kTRUE;
   if (!optcandle && !optpara) fHistogram = (TH1*)fSelector->GetObject();
   if (optnorm) {
      Double_t sumh= fHistogram->GetSumOfWeights();
      if (sumh != 0) fHistogram->Scale(1./sumh);
   }

   //if (!nrows && draw && drawflag && !opt.Contains("same")) {
   //   if (gPad) gPad->Clear();
   //   return 0;
   //}
   if (drawflag) {
      if (gPad) {
         gPad->DrawFrame(-1.,-1.,1.,1.);
         TText *text_empty = new TText(0.,0.,"Empty");
         text_empty->SetTextAlign(22);
         text_empty->SetTextFont(42);
         text_empty->SetTextSize(0.1);
         text_empty->SetTextColor(1);
         text_empty->Draw();
      } else {
         Warning("DrawSelect", "The selected TTree subset is empty.");
      }
   }

   //*-*- 1-D distribution
   if (fDimension == 1) {
      if (fSelector->GetVar1()->IsInteger()) fHistogram->LabelsDeflate("X");
      if (draw) fHistogram->Draw(opt.Data());

   //*-*- 2-D distribution
   } else if (fDimension == 2 && !(optpara||optcandle)) {
      if (fSelector->GetVar1()->IsInteger()) fHistogram->LabelsDeflate("Y");
      if (fSelector->GetVar2()->IsInteger()) fHistogram->LabelsDeflate("X");
      if (action == 4) {
         if (draw) fHistogram->Draw(opt.Data());
      } else {
         Bool_t graph = kFALSE;
         Int_t l = opt.Length();
         if (l == 0 || opt == "same") graph = kTRUE;
         if (opt.Contains("p")     || opt.Contains("*")    || opt.Contains("l"))    graph = kTRUE;
         if (opt.Contains("surf")  || opt.Contains("lego") || opt.Contains("cont")) graph = kFALSE;
         if (opt.Contains("col")   || opt.Contains("hist") || opt.Contains("scat")) graph = kFALSE;
         if (!graph) {
            if (draw) fHistogram->Draw(opt.Data());
         } else {
            if (fSelector->GetOldHistogram() && draw) fHistogram->Draw(opt.Data());
         }
      }
   //*-*- 3-D distribution
   } else if (fDimension == 3 && !(optpara||optcandle)) {
      if (fSelector->GetVar1()->IsInteger()) fHistogram->LabelsDeflate("Z");
      if (fSelector->GetVar2()->IsInteger()) fHistogram->LabelsDeflate("Y");
      if (fSelector->GetVar3()->IsInteger()) fHistogram->LabelsDeflate("X");
      if (action == 23) {
         if (draw) fHistogram->Draw(opt.Data());
      } else if (action == 33) {
         if (draw) {
            if (opt.Contains("z")) fHistogram->Draw("func z");
            else                   fHistogram->Draw("func");
         }
      } else {
         Int_t noscat = opt.Length();
         if (opt.Contains("same")) noscat -= 4;
         if (noscat) {
            if (draw) fHistogram->Draw(opt.Data());
         } else {
            if (fSelector->GetOldHistogram() && draw) fHistogram->Draw(opt.Data());
         }
      }
   //*-*- 4-D distribution
   } else if (fDimension == 4 && !(optpara||optcandle)) {
      if (fSelector->GetVar1()->IsInteger()) fHistogram->LabelsDeflate("Z");
      if (fSelector->GetVar2()->IsInteger()) fHistogram->LabelsDeflate("Y");
      if (fSelector->GetVar3()->IsInteger()) fHistogram->LabelsDeflate("X");
      if (draw) fHistogram->Draw(opt.Data());
      Int_t ncolors  = gStyle->GetNumberOfColors();
      TObjArray *pms = (TObjArray*)fHistogram->GetListOfFunctions()->FindObject("polymarkers");
      for (Int_t col=0;col<ncolors;col++) {
         if (!pms) continue;
         TPolyMarker3D *pm3d = (TPolyMarker3D*)pms->UncheckedAt(col);
         if (draw) pm3d->Draw();
      }
   //*-*- Parallel Coordinates or Candle chart.
   } else if (optpara || optcandle) {
      if (draw) {
         TObject* para = fSelector->GetObject();
         fTree->Draw(">>enlist",selection,"entrylist",nentries,firstentry);
         TObject *enlist = gDirectory->FindObject("enlist");
         gROOT->ProcessLine(Form("TParallelCoord::SetEntryList((TParallelCoord*)0x%lx,(TEntryList*)0x%lx)",
                                     (ULong_t)para, (ULong_t)enlist));
      }
   //*-*- 5d with gl
   } else if (optgl5d) {
      gROOT->ProcessLineFast(Form("(new TGL5DDataSet((TTree *)0x%lx))->Draw(\"%s\");", (ULong_t)fTree, opt.Data()));
      gStyle->SetCanvasPreferGL(pgl);
   }

   if (fHistogram) fHistogram->SetCanExtend(TH1::kNoAxis);
   return true;

}


bool TMultiDrawTreePlayer::execute() {
    if (m_draws.empty())
        return false;

    // Process this tree executing the code in the specified selector.
    // The return value is -1 in case of error and TSelector::GetStatus() in
    // in case of success.
    //
    //   The TSelector class has the following member functions:
    //
    //    Begin():        called every time a loop on the tree starts,
    //                    a convenient place to create your histograms.
    //    SlaveBegin():   called after Begin(), when on PROOF called only on the
    //                    slave servers.
    //    Process():      called for each event, in this function you decide what
    //                    to read and fill your histograms.
    //    SlaveTerminate: called at the end of the loop on the tree, when on PROOF
    //                    called only on the slave servers.
    //    Terminate():    called at the end of the loop on the tree,
    //                    a convenient place to draw/fit your histograms.
    //
    //  If the Tree (Chain) has an associated EventList, the loop is on the nentries
    //  of the EventList, starting at firstentry, otherwise the loop is on the
    //  specified Tree entries.

    //TDirectory::TContext ctxt(0);
    
    Long64_t firstentry = std::numeric_limits<Long64_t>::max();
    Long64_t lastentry = 0;

    // Get first and lastentry
    for (auto& data: m_draws) {
        firstentry = std::min(firstentry, data.firstentry);
        lastentry = std::max(lastentry, data.firstentry + data.nentries);

        data.selector->SetOption(data.options.c_str());

        data.selector->Begin(fTree);       //<===call user initialization function
        data.selector->SlaveBegin(fTree);  //<===call user initialization function
        if (data.selector->Version() >= 2)
            data.selector->Init(fTree);
        data.selector->Notify();
    }

    Long64_t nentries = lastentry - firstentry;

    nentries = GetEntriesToProcess(firstentry, nentries);

    NotifyProxier notifyProxier(m_draws);
    fTree->SetNotify(&notifyProxier);

    if (gMonitoringWriter)
        gMonitoringWriter->SendProcessingStatus("STARTED",kTRUE);

    Long64_t readbytesatstart = 0;
    readbytesatstart = TFile::GetFileBytesRead();

    //set the file cache
    TTreeCache *tpf = 0;
    TFile *curfile = fTree->GetCurrentFile();
    if (curfile && fTree->GetCacheSize() > 0) {
        tpf = (TTreeCache*)curfile->GetCacheRead(fTree);
        if (tpf)
            tpf->SetEntryRange(firstentry,firstentry+nentries);
        else {
            fTree->SetCacheSize(fTree->GetCacheSize());
            tpf = (TTreeCache*)curfile->GetCacheRead(fTree);
            if (tpf) tpf->SetEntryRange(firstentry,firstentry+nentries);
        }
    }

    //Create a timer to get control in the entry loop(s)
    TProcessEventTimer *timer = 0;
    Int_t interval = fTree->GetTimerInterval();
    if (!gROOT->IsBatch() && interval)
        timer = new TProcessEventTimer(interval);

    //loop on entries (elist or all entries)
    Long64_t entry, entryNumber, localEntry;

    // force the first monitoring info
    if (gMonitoringWriter)
        gMonitoringWriter->SendProcessingProgress(0,0,kTRUE);

    //trying to set the first tree, because in the Draw function
    //the tree corresponding to firstentry has already been loaded,
    //so it is not set in the entry list
    for (auto& data: m_draws) {
        fSelectorUpdate = data.selector.get();
        UpdateFormulaLeaves();
    }

    std::chrono::time_point<std::chrono::system_clock> t_first = std::chrono::system_clock::now();
    std::chrono::time_point<std::chrono::system_clock> t_old = std::chrono::system_clock::now();
    std::vector<double> deq;
    int period = nentries/3000;
    int smoothing = 40;

    RooUtil::print( "Start EventLooping using TMultiDrawTreePlayer" );
    RooUtil::start();
    TBenchmark* bmark = new TBenchmark();
    bmark->Start("benchmark");
    RooUtil::print( Form("Total events to loop over = %lld", nentries) );

    for (entry=firstentry;entry<firstentry+nentries;entry++) {
        entryNumber = fTree->GetEntryNumber(entry);
        if (entryNumber < 0) break;
        if (timer && timer->ProcessEvents()) break;
        if (gROOT->IsInterrupted()) break;
        localEntry = fTree->LoadTree(entryNumber);
        if (localEntry < 0) break;

        // if (entryNumber % 10000 == 0) std::cout << entryNumber << std::endl;

        if(period > 0 && entryNumber%period == 0) {
            auto now = std::chrono::system_clock::now();
            double dt = ((std::chrono::duration<double>)(now - t_old)).count();
            t_old = now;
            if (deq.size() >= (unsigned int) smoothing) deq.erase(deq.begin());
            deq.push_back(dt);
            double avgdt = std::accumulate(deq.begin(),deq.end(),0.)/deq.size();
            float prate = (float)period/avgdt;
            float peta = (nentries-entryNumber)/prate;
            if (isatty(1)) {
                float pct = (float)entryNumber/(nentries*0.01);
                if( ( nentries - entryNumber ) <= period ) {
                    pct = 100.0;
                    double dt_total = ((std::chrono::duration<double>)(now - t_first)).count();
                    printf("\015\033[32m ---> \033[1m\033[31m%4.1f%% \033[34m [Avg rate: %.2f kHz, Time elapsed: %.0f s] \033[0m\033[32m  <---\033[0m\015 ", pct, nentries/(1000.*dt_total), dt_total);
                } else {
                    printf("\015\033[32m ---> \033[1m\033[31m%4.1f%% \033[34m [%.2f kHz, ETA: %.0f s] \033[0m\033[32m  <---\033[0m\015 ", pct, prate/1000.0, peta);
                }
                if( ( nentries - entryNumber ) > period ) fflush(stdout);
                else std::cout << std::endl;

            }
        }

        bool abort = false;
        bool skipToNextFile = false;
        for (auto& data: m_draws) {
            bool process = (data.selector->GetAbort() != TSelector::kAbortProcess &&
                    (data.selector->Version() != 0 || data.selector->GetStatus() != -1)) ? true : false;

            if ((entry < data.firstentry) || (entry >= (data.firstentry + data.nentries)))
                process = false;

            if (! process)
                continue;

            bool useCutFill = data.selector->Version() == 0;

            if(useCutFill) {
                if (data.selector->ProcessCut(localEntry))
                    data.selector->ProcessFill(localEntry); //<==call user analysis function
            } else {
                data.selector->Process(localEntry);        //<==call user analysis function
            }

            if (data.selector->GetAbort() == TSelector::kAbortProcess)
                abort = true;

            if (data.selector->GetAbort() == TSelector::kAbortFile) {
                skipToNextFile = true;
                data.selector->ResetAbort();
            }
        }

        if (gMonitoringWriter)
            gMonitoringWriter->SendProcessingProgress((entry-firstentry),TFile::GetFileBytesRead()-readbytesatstart,kTRUE);

        if (abort)
            break;

        if (skipToNextFile) {
            // Skip to the next file.
            entry += fTree->GetTree()->GetEntries() - localEntry;
            skipToNextFile = false;
        }
    }

    RooUtil::end();
     
    // return
    using namespace std;
    bmark->Stop("benchmark");
    cout << endl;
    cout << "------------------------------" << endl;
    cout << "CPU  Time:	" << Form( "%.01f", bmark->GetCpuTime("benchmark")  ) << endl;
    cout << "Real Time:	" << Form( "%.01f", bmark->GetRealTime("benchmark") ) << endl;
    cout << endl;
    delete bmark;

    delete timer;
    //we must reset the cache
    {
        TFile *curfile2 = fTree->GetCurrentFile();
        if (curfile2 && fTree->GetCacheSize() > 0) {
            tpf = (TTreeCache*)curfile2->GetCacheRead(fTree);
            if (tpf) tpf->SetEntryRange(0,0);
        }
    }

    bool res = true;
    for (auto& data: m_draws) {
        bool process = (data.selector->GetAbort() != TSelector::kAbortProcess &&
                (data.selector->Version() != 0 || data.selector->GetStatus() != -1)) ? true : false;

        if (! process)
            continue;

        data.selector->SlaveTerminate();   //<==call user termination function
        data.selector->Terminate();        //<==call user termination function
        res &= (data.selector->GetStatus() != 0);
    }

    fTree->SetNotify(0); // Detach the selector from the tree.
    fSelectorUpdate = 0;
    if (gMonitoringWriter)
        gMonitoringWriter->SendProcessingStatus("DONE");

    m_draws.clear();

    return res;
}
