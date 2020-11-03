#include "QFramework/TQHistComparer.h"
#include "QFramework/TQLibrary.h"
#include "QFramework/TQIterator.h"
#include "QFramework/TQPlotter.h"
#include "TFile.h"
#include "TCanvas.h"
#include "THStack.h"
#include <algorithm>

////////////////////////////////////////////////////////////////////////////////////////////////
//
// TQHistComparer:
//
// TODO: write documentation
//
////////////////////////////////////////////////////////////////////////////////////////////////

ClassImp(TQHistComparer)

TQHistComparer::TQHistComparer(TQSampleFolder* sf) : TQPresenter(sf) {
  // constructor using a sample folder
  fHists = new TObjArray();
  fSummaryHists = new TObjArray();
  fDistNames = new TObjArray();
}

TQHistComparer::TQHistComparer(TQSampleDataReader* reader) : TQPresenter(reader) {
  // constructor using a sample data reader
  fHists = new TObjArray();
  fDistNames = new TObjArray();
}

bool TQHistComparer::resetDistributions()
{
  this->fDistNames->Clear();
  return true;
}

void TQHistComparer::addDistribution(TString name, TString title)
{
  if (title == "")
    title = name;
  TQNamedTaggable* dist = new TQNamedTaggable(name);
  dist->setTagString(".title",title);
  dist->setTagString(".name",name);
  this->fDistNames->Add(dist);
}


bool TQHistComparer::writeToFile(const TString& filename, const TString& filename_summary, const TString& tags) {
  // write the plot to a file
  TQTaggable tmp(tags);
  return this->writeToFile(filename, filename_summary, tmp);
}

bool TQHistComparer::writeToFile(const TString& filename, const TString& filename_summary, TQTaggable& tags){
  // write the plot to a file
  tags.importTags(this);
  if (this->fProcesses->GetEntries() != 2)
    {
      ERRORclass("TQHistComparer needs exactly 2 processes but found %i !",this->fProcesses->GetEntries());
      return false;
    }
  if (this->fDistNames->GetEntries() < 1)
    {
      ERRORclass("TQHistComparer needs at least one distribution");
      return false;
    }
  if (this->fCuts->GetEntries() < 1)
    {
      ERRORclass("TQHistComparer needs at least one cut");
      return false;
    }
  TH1 * h1, *h2;
  h1 = h2 = 0;
  TQTaggableIterator cuts(this->fCuts);
  TQTaggableIterator processes(this->fProcesses);
  TQNamedTaggable *process = 0;
  int i = 0;
  TQTaggableIterator dists(this->fDistNames);
  int bin;
  std::vector<float> *vals = new std::vector<float>;
  while(cuts.hasNext()){
    bin=0;
    TQNamedTaggable* cut = cuts.readNext();
    TString name = cut->getName();
    TH1D *h = new TH1D(name,name,this->fDistNames->GetEntries(),-0.5,fDistNames->GetEntries()-0.5);
    TH1D *h_sum = 0;
    dists.reset();

    while(dists.hasNext()){
      ++bin;
      TQNamedTaggable *dist = dists.readNext();
      //@tag: [.title] Distribution tag used to set the lable of the corresponding bin. Default: "none"
      h->GetXaxis()->SetBinLabel(bin,dist->getTagStringDefault(".title","none"));
      i=0;
      processes.reset();
      while (processes.hasNext()){
        process = processes.readNext();
        if (i==0)
          h1 = this->fReader->getHistogram(process->getName(),cut->getTagStringDefault(".name","non")+"/"+dist->getTagStringDefault(".name","non"));
        else
          h2 = this->fReader->getHistogram(process->getName(),cut->getTagStringDefault(".name","non")+"/"+dist->getTagStringDefault(".name","non"));
        ++i;
      }
      if (h1==0 || h2 == 0)
        {
          ERRORclass("Couldn't find both %s histograms for cut %s ", dist->getName().Data(),cut->getName().Data());
          return false;
        }
      double val = -999;
      if(tags.getTagBoolDefault("useChi2",false))
        val = h1->Chi2Test(h2,"UW CHI2/NDF");
      else
        val = h1->KolmogorovTest(h2);
      h->SetBinContent(bin,val);
      vals->push_back(val);

 
      INFOclass("Cut: %s , distribution: %s - %f",cut->getName().Data(),dist->getName().Data(), val);
    }
    if(tags.getTagBoolDefault("useChi2",false))
      h_sum = new TH1D(name+"_summary",name+"_summary",this->fDistNames->GetEntries()/2,0,*(std::max_element(vals->begin(),vals->end()))*1.1);
    else
      h_sum = new TH1D(name+"_summary",name+"_summary",this->fDistNames->GetEntries()/2,0,1);
    for (unsigned int i=0; i<vals->size(); ++i)
      h_sum->Fill(vals->at(i));
    vals->clear();
    fSummaryHists->Add(h_sum);
    fHists->Add(h);
  }
  TQTH1Iterator itr(fHists->MakeIterator(kIterForward),true);
  TFile * of = 0;
  TCanvas *c1 = new TCanvas("c1","c1",this->fDistNames->GetEntries()*60,600);
  THStack *s = new THStack("s","stack");
  if (filename.Contains("root"))
    {
      of = new TFile(filename,"RECREATE");
    }
  i = 0;
  while(itr.hasNext() != 0)
    {
      TH1 *h = itr.readNext();
      h->SetLineColor(kRed+i*4);
      h->SetLineWidth(2);
      if (of)
        h->Write();
      else
        {
          TQPlotter::setStyleAtlas();
          s->Add(h);
        }
      ++i;
    }
  if (of)
    of->Close();
  else
    {
      s->Draw("nostack");
      if(tags.getTagBoolDefault("useChi2",false))
        s->GetHistogram()->GetYaxis()->SetTitle("#chi^{2}/ndf");
      else
        s->GetHistogram()->GetYaxis()->SetTitle("KS");
      s->GetHistogram()->GetYaxis()->SetTitleOffset(0.7);
      s->Draw("nostacksame");
      this->makeLegend(tags,fHists)->Draw("same");
      c1->SaveAs(filename);
    }

  TQTH1Iterator itr2(fSummaryHists->MakeIterator(kIterForward),true);
  TCanvas *c2 = new TCanvas("c2","c2",800,600);
  THStack *s2 = new THStack("s2","stack");
  i = 0;
  while(itr2.hasNext() != 0)
    {
      TH1 *h = itr2.readNext();
      h->SetLineColor(kRed+i*4);
      h->SetLineWidth(2);
      if (of)
        h->Write();
      else
        {
          TQPlotter::setStyleAtlas();
          s2->Add(h);
        }
      ++i;
    }
  if (of)
    of->Close();
  else
    {
      s2->Draw("nostack");
      if(tags.getTagBoolDefault("useChi2",false))
        s2->GetHistogram()->GetXaxis()->SetTitle("#chi^{2}/ndf");
      else
        s2->GetHistogram()->GetXaxis()->SetTitle("KS");
      s2->GetHistogram()->GetYaxis()->SetTitle("Entries");
      s2->Draw("nostacksame");
      this->makeLegend(tags,fHists)->Draw("same");
      c2->SaveAs(filename_summary);
    }
  return true;
}
 
TLegend * TQHistComparer::makeLegend(TQTaggable& tags, TObjArray* histos){
  int nLegendCols = tags.getTagIntegerDefault ("style.nLegendCols",1);
  double legendHeight = tags.getTagDoubleDefault ("style.legendHeight",1. );

  /* the nominal coordinates of the legend */
  double x1 = tags.getTagDoubleDefault("legend.xMin",0.9);
  double y1 = tags.getTagDoubleDefault("legend.yMin",0.70) - tags.getTagDoubleDefault("geometry.main.additionalTopMargin",0.);
  double x2 = tags.getTagDoubleDefault("legend.xMax",0.50);
  double y2 = tags.getTagDoubleDefault("legend.yMax",0.92) - tags.getTagDoubleDefault("geometry.main.additionalTopMargin",0.);

  /* ===== scale the hight of the legend depending on the number of entries ===== */
  /* calculate the number of entries */
  int nEntries = histos->GetEntries();

  /* calculate the height of the legend */
  int nLegendRows = (int)nEntries / nLegendCols + ((nEntries % nLegendCols) > 0 ? 1 : 0);
  legendHeight *= (y2 - y1) * (double)nLegendRows / 5.;

  /* set the height of the legend */
  y1 = y2 - legendHeight;
  /* create the legend and set some attributes */
  double tmpx1 = x1; 
  double tmpx2 = x2; 
  double tmpy1 = y1; 
  double tmpy2 = y2; 
 
  TLegend* legend = new TLegend(tmpx1, tmpy1, tmpx2, tmpy2);
  legend->SetBorderSize(0);
  legend->SetNColumns(nLegendCols);
  double textsize = tags.getTagDoubleDefault("legend.textSize",0.032);
  legend->SetTextSize(textsize);
 
  TQTH1Iterator itr(fHists->MakeIterator(kIterForward),true);
  while(itr.hasNext() != 0)
    {
      TH1 *h = itr.readNext();
      legend->AddEntry(h,h->GetName(),"l");
    }
  return legend;
}
