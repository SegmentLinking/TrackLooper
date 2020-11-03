//  .
// ..: P. Chang, philip@physics.ucsd.edu

#include "autohist.h"

///////////////////////////////////////////////////////////////////////////////////////////////////
//
//
// Auto histogram maker
//
//
///////////////////////////////////////////////////////////////////////////////////////////////////

//__________________________________________________________________________________________________
RooUtil::AutoHist::AutoHist()
{
    resolution = 1000;
}

//__________________________________________________________________________________________________
RooUtil::AutoHist::~AutoHist()
{
    if ( histdb.size() > 0 )
        save( "autohist_output.root" );
}

//__________________________________________________________________________________________________
void RooUtil::AutoHist::save( TString ofilename, TString option )
{
    TFile* ofile = new TFile( ofilename, option );
    RooUtil::print( "AutoHist::save() saving histograms to " + ofilename );
    ofile->cd();

    for ( auto& pair_tstr_th1 : histdb )
    {
        TString name = pair_tstr_th1.first;
        TH1* hist = pair_tstr_th1.second;

        if ( hist )
        {
            hist->Write();
            delete hist;
        }
    }

    histdb.clear();
}

//__________________________________________________________________________________________________
void RooUtil::AutoHist::save( TFile* ofile )
{
    RooUtil::print( Form( "AutoHist::save() saving histograms to %s", ofile->GetName() ) );
    ofile->cd();

    for ( auto& pair_tstr_th1 : histdb )
    {
        TString name = pair_tstr_th1.first;
        TH1* hist = pair_tstr_th1.second;

        if ( hist )
        {
            hist->Write();
            delete hist;
        }
    }

    histdb.clear();
}

//__________________________________________________________________________________________________
void RooUtil::AutoHist::fill( double xval, STRING name, double wgt )
{
    TH1* hist = 0;
    std::pair<MAP<STRING, TH1*>::iterator, bool> ret;
    ret = histdb.insert( {name, hist} );

    if ( ret.second == false )
        fill( xval, ( *ret.first ).second, wgt );
    else
    {
        hist = createHist( xval, name, wgt );
        histdb[name] = hist;
    }
}

//__________________________________________________________________________________________________
void RooUtil::AutoHist::fill(
        double xval, STRING name, double wgt, int nbin, double min, double max,
        std::vector<TString> binlabels )
{
    TH1* hist = 0;
    std::pair<MAP<STRING, TH1*>::iterator, bool> ret;
    ret = histdb.insert( {name, hist} );

    if ( ret.second == false )
        fill( xval, ( *ret.first ).second, wgt, true );
    else
    {
        hist = createFixedBinHist( xval, name, wgt, nbin, min, max );

        for ( unsigned int ibin = 0; ibin < binlabels.size(); ++ibin )
            hist->GetXaxis()->SetBinLabel( ibin + 1, binlabels[ibin] );

        histdb[name] = hist;
    }
}

//__________________________________________________________________________________________________
void RooUtil::AutoHist::fill( double xval, STRING name, double wgt, int nbin, double* bins )
{
    TH1* hist = 0;
    std::pair<MAP<STRING, TH1*>::iterator, bool> ret;
    ret = histdb.insert( {name, hist} );

    if ( ret.second == false )
        fill( xval, ( *ret.first ).second, wgt, true );
    else
    {
        hist = createFixedBinHist( xval, name, wgt, nbin, bins );
        histdb[name] = hist;
    }
}

//__________________________________________________________________________________________________
void RooUtil::AutoHist::fill(
        double xval, double yval, STRING name, double wgt,
        int nbx, double xm, double xM,
        int nby, double ym, double yM,
        std::vector<TString> binlabels
        )
{
    TH1* hist = 0;
    std::pair<MAP<STRING, TH1*>::iterator, bool> ret;
    ret = histdb.insert( {name, hist} );

    if ( ret.second == false )
        ( ( TH2D* )( *ret.first ).second )->Fill( xval, yval, wgt );
    else
    {
        hist = createFixedBinHist( xval, yval, name, wgt, nbx, xm, xM, nby, ym, yM );

        for ( unsigned int ibin = 0; ibin < binlabels.size(); ++ibin )
            hist->GetXaxis()->SetBinLabel( ibin + 1, binlabels[ibin] );

        histdb[name] = hist;
    }
}

//__________________________________________________________________________________________________
void RooUtil::AutoHist::fill(
        double xval, double yval, STRING name, double wgt,
        int nbx, double* bx,
        int nby, double* by )
{
    TH1* hist = 0;
    std::pair<MAP<STRING, TH1*>::iterator, bool> ret;
    ret = histdb.insert( {name, hist} );

    if ( ret.second == false )
        ( ( TH2D* )( *ret.first ).second )->Fill( xval, yval, wgt );
    else
    {
        hist = createFixedBinHist( xval, yval, name, wgt, nbx, bx, nby, by );
        histdb[name] = hist;
    }
}

//__________________________________________________________________________________________________
int RooUtil::AutoHist::getRes( double range )
{
    if ( range > 1000 )
        return 10;
    else if ( range > 500 )
        return 100;
    else if ( range > 250 )
        return 1000;
    else if ( range > 1 )
        return 1000;
    else
        return 10000;
}

//__________________________________________________________________________________________________
int RooUtil::AutoHist::getRes( TH1* h )
{
    double max = h->GetXaxis()->GetXmax();
    double min = h->GetXaxis()->GetXmin();
    int nbin = h->GetNbinsX();
    return nbin / ( max - min );
}

//__________________________________________________________________________________________________
TH1* RooUtil::AutoHist::createHist( double xval, TString name, double wgt, bool alreadyneg, int forceres )
{
    // Create a histogram with the xval as the characteristic value.
    // It takes xval, then multiply by 2 find the closest integer,
    // and give histogram of nbin where each intger has 1000 bins.
    // If it extends to negative value, blow it up
    int bound = 2 * std::max( ( ( int ) fabs( xval ) ), 1 );
    bool extendneg = ( alreadyneg || xval < 0 ) ? true : false;
    double min = extendneg ? -bound : 0;
    double max = bound;
    int res = forceres > 0 ? forceres : getRes( max - min );
    int nbin = ( max - min ) * res;
    TH1D* h = new TH1D( name, name, nbin, min, max );
    h->SetDirectory( 0 );
    h->Sumw2();
    h->Fill( xval, wgt );
    return h;
}

//__________________________________________________________________________________________________
TH1* RooUtil::AutoHist::createFixedBinHist( double xval, TString name, double wgt, int n, double m, double M )
{
    TH1D* h = new TH1D( name, name, n, m, M );
    h->SetDirectory( 0 );
    h->Sumw2();
    h->Fill( xval, wgt );
    return h;
}

//__________________________________________________________________________________________________
TH1* RooUtil::AutoHist::createFixedBinHist( double xval, TString name, double wgt, int n, double* bin )
{
    TH1D* h = new TH1D( name, name, n, bin );
    h->SetDirectory( 0 );
    h->Sumw2();
    h->Fill( xval, wgt );
    return h;
}

//__________________________________________________________________________________________________
TH1* RooUtil::AutoHist::createFixedBinHist(
        double xval, double yval, TString name, double wgt,
        int xn, double xm, double xM,
        int yn, double ym, double yM )
{
    TH2D* h = new TH2D( name, name, xn, xm, xM, yn, ym, yM );
    h->SetDirectory( 0 );
    h->Sumw2();
    h->Fill( xval, yval, wgt );
    return h;
}

//__________________________________________________________________________________________________
TH1* RooUtil::AutoHist::createFixedBinHist(
        double xval, double yval, TString name, double wgt,
        int xn, double* xb,
        int yn, double* yb )
{
    TH2D* h = new TH2D( name, name, xn, xb, yn, yb );
    h->SetDirectory( 0 );
    h->Sumw2();
    h->Fill( xval, yval, wgt );
    return h;
}

//__________________________________________________________________________________________________
void RooUtil::AutoHist::fill( double xval, TH1*& h, double wgt, bool norebinning )
{
    if ( ( xval >= h->GetXaxis()->GetXmax() || xval <= h->GetXaxis()->GetXmin() ) && !norebinning )
    {
        TH1D* hist = ( TH1D* ) createHist( xval, h->GetName(), wgt, h->GetXaxis()->GetXmin() < 0 );
        transfer( hist, h );
        delete h;
        h = hist;
    }
    else
        h->Fill( xval, wgt );
}

//__________________________________________________________________________________________________
void RooUtil::AutoHist::transfer( TH1* tohist, TH1* fromhist )
{
    for ( int ibin = 0; ibin <= fromhist->GetNbinsX(); ++ibin )
    {
        double wgt = fromhist->GetBinContent( ibin );
        double err = fromhist->GetBinError( ibin );
        double val = fromhist->GetBinCenter( ibin );

        if ( fabs( wgt ) )
        {
            int jbin = tohist->FindBin( val );
            double curerror = tohist->GetBinError( jbin );
            tohist->Fill( val, wgt );
            tohist->SetBinError( jbin, sqrt( curerror * curerror + err * err ) );
        }
    }
}

//__________________________________________________________________________________________________
TH1* RooUtil::AutoHist::get( STRING name )
{
    try
    {
        return histdb.at( name );
    }
    catch ( const std::out_of_range& oor )
    {
        RooUtil::print( "Warning! No histogram named " + name + " found!" );
        return 0;
    }
}

//__________________________________________________________________________________________________
void RooUtil::AutoHist::print()
{
    RooUtil::print("Printing histdb ...");
    for (auto& i : histdb)
    {
        TString msg = Form("  %s", i.second->GetName());
        RooUtil::print(msg.Data());
    }
}

//__________________________________________________________________________________________________
TH1* RooUtil::AutoHist::crop( TH1* orighist, int nbin, double min, double max )
{
    // Checks whether this is uniform binning
    // https://root-forum.cern.ch/t/taxis-getxbins/19598/2
    if ( ( *( orighist->GetXaxis()->GetXbins() ) ).GetSize() == 0 )
    {
        int    orignbin = orighist->GetNbinsX();
        double origmin = orighist->GetXaxis()->GetXmin();
        double origmax = orighist->GetXaxis()->GetXmax();
        double origbinsize = ( origmax - origmin ) / orignbin;

        if ( std::remainder( fabs( origmin - max ), origbinsize ) < 1e-9
                && std::remainder( fabs( origmin - min ), origbinsize ) < 1e-9 )
        {
            TH1D* rtnhist = new TH1D( orighist->GetName(), orighist->GetName(), nbin, min, max );
            rtnhist->Sumw2();
            rtnhist->SetDirectory( 0 );
            transfer( rtnhist, orighist );
            return rtnhist;
        }
        // Not good, I can't crop it.
        else
            RooUtil::error( "You are trying to crop a histogram to a binning where you can't!", __FUNCTION__ );
    }

    return 0;
}

//__________________________________________________________________________________________________
TH1* RooUtil::AutoHist::hadd( TH1* hist1, TH1* hist2 )
{
    double min1 = hist1->GetXaxis()->GetXmin();
    double max1 = hist1->GetXaxis()->GetXmax();
    double min2 = hist2->GetXaxis()->GetXmin();
    double max2 = hist2->GetXaxis()->GetXmax();

    if ( min1 <= min2 && max1 >= max2 )
    {
        transfer( hist1, hist2 );
        return hist1;
    }
    else if ( min2 <= min1 && max2 >= max1 )
    {
        transfer( hist2, hist1 );
        return hist2;
    }
    else if ( max1 >= max2 && min1 >= min2 )
    {
        TH1* newhist = createHist( max1, hist1->GetName(), 0, true, getRes( hist1 ) );
        transfer( newhist, hist1 );
        transfer( newhist, hist2 );
        return newhist;
    }
    else if ( max2 >= max1 && min2 >= min1 )
    {
        TH1* newhist = createHist( max2, hist2->GetName(), 0, true, getRes( hist2 ) );
        transfer( newhist, hist1 );
        transfer( newhist, hist2 );
        return newhist;
    }
    else
    {
        RooUtil::error( "it should never reach here! send email to philip@physics.ucsd.edu", __FUNCTION__ );
        return 0;
    }
}
