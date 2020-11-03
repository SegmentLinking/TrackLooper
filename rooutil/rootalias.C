void print(TH1* h)
{
    THashList* labels = h->GetXaxis()->GetLabels();
    if (labels)
    {

//        TH1.Print Name  = test, Entries= 0, Total sum= 0
//        fSumw[0]=0, x=-0.5
//        fSumw[1]=0, x=0.5
//        fSumw[2]=0, x=1.5
//        fSumw[3]=0, x=2.5
//        fSumw[4]=0, x=3.5
//        fSumw[5]=0, x=4.5
//        fSumw[6]=0, x=5.5

        std::cout << " TH1.Print Name = " << h->GetName() << ", Entries= " << h->GetEntries() << ", Total sum= " << h->Integral() << std::endl;
        for (unsigned int i = 0; i < h->GetNbinsX()+2; ++i)
        {
            float bc = h->GetBinContent(i);
            float be = h->GetBinError(i);
            TString bl = h->GetXaxis()->GetBinLabel(i);
            std::cout << " fSumw[" << i << "]=" << bc << ", error=" << be << ", binLabel=" << bl << std::endl;
        }
    }
    else
    {
        h->Print("all");
    }
}
