import os
import sys
from errors import E
import ROOT as r

r.gROOT.SetBatch(True)

main_canvas = None
main_legend = None
nsame = 0

oldinit = r.TH1F.__init__
class MyTH1F(r.TH1F):
    def __init__(self, *args):
        oldinit(self, *args)

    def classify_me(self):
        return self.__class__(self)

    """
    Adding and subtracting returns the old TH1F, so
    overload and wrap in this new class to save users
    from doing it themselves
    """
    oldadd = r.TH1F.__add__
    def __add__(self,other):
        return self.__class__(self.oldadd(other))
    oldsub = r.TH1F.__sub__
    def __sub__(self,other):
        return self.__class__(self.oldsub(other))
    oldrmul = r.TH1F.__rmul__
    def __rmul__(self,other):
        return self.__class__(self.oldrmul(other))
    oldmul = r.TH1F.__mul__
    def __mul__(self,other):
        return self.__class__(self.oldmul(other))

    def IntegralBetween(self, xmin, xmax):
        """
        Return single value for integral between `xmin` and `xmax`
        """
        axis = self.GetXaxis()
        bmin = axis.FindBin(xmin)
        bmax = axis.FindBin(xmax)
        return self.Integral(bmin,bmax)

    def IntegralAndErrorBetween(self, xmin, xmax):
        """
        Return error object for integral between `xmin` and `xmax`
        """
        axis = self.GetXaxis()
        bmin = axis.FindBin(xmin)
        bmax = axis.FindBin(xmax)
        error = r.Double()
        value = self.IntegralAndError(bmin,bmax,error)
        return E(value,error)

    def GetBinValueErrors(self):
        """
        Return error objects for all bins (including under/overflow)
        """
        values = [self.GetBinContent(ibin) for ibin in range(self.GetNbinsX()+2)]
        errors = [self.GetBinError(ibin)   for ibin in range(self.GetNbinsX()+2)]
        return map(lambda x: E(*x),zip(values, errors))

    def FillFromList(self, xvals, weights=None):
        """
        Partially stolen from root_numpy implementation.
        Using a for loop with TH1::Fill() is slow in Python,
        so use numpy to convert list/array/numpy array 
        `xvals` to C-style array, and then FillN.
        `weights` are optional
        """
        import numpy as np
        xvals = np.asarray(xvals, dtype=np.double)
        two_d = False
        # if yvals is not None:
        #     two_d = True
        #     yvals = np.asarray(yvals, dtype=np.double)
        if weights is None:
            weights = np.ones(len(xvals))
        else:
            weights = np.asarray(weights, dtype=np.double)
        if not two_d:
            self.FillN(len(xvals),xvals,weights)
        else:
            self.FillN(len(xvals),xvals,yvals,weights)

    def MakePretty(self, color=r.kRed-2):
        self.SetLineColor(color)
        self.SetMarkerColor(color)
        self.SetLineWidth(2)
        self.SetFillColorAlpha(self.GetLineColor(),0.5)

    def Show(self, drawopt="HISTE", logy=False):
        """
        Quick peek at a histogram using imgcat.
        If it's not the first time making a TCanvas,
        then just reuse the old one.
        Also, count up how many histograms are on the
        same canvas based off of the draw option,
        so that we can color them differently
        """
        global main_canvas, nsame, main_legend
        if not main_canvas:
            c1 = r.TCanvas("c1","c1",600,400)
            c1.SetCanvasSize(600,400)
            main_canvas = c1
        else:
            c1 = main_canvas
        if "same" in drawopt:
            nsame += 1
        else:
            nsame = 0
        self.MakePretty([r.kAzure-2, r.kRed-2, r.kBlack, r.kOrange+2][nsame % 4])
        c1.SetLogy(logy)
        self.Draw(drawopt)
        if not main_legend or not "same" in drawopt:
            main_legend = r.TLegend(0.14, 0.67, 0.14+0.3, 0.67+0.2)
        main_legend.AddEntry(self)
        main_legend.Draw("same")
        c1.SaveAs("temp.png")
        os.system("which -s ic && ic temp.png || imgcat temp.png")

r.TH1F = MyTH1F

if __name__ == "__main__":

    import numpy as np

    h1 = r.TH1F("h1","not a regular hist",10,0,10)
    h1.FillRandom("expo",50000)

    print h1.IntegralBetween(0.,2.)

    # now fill it with my own values
    rands = np.random.normal(5.,0.1, 100000)
    h1.FillFromList(rands)
    print h1.GetEntries()
    binvalerrs = h1.GetBinValueErrors() 
    print binvalerrs
    print sum(binvalerrs)
    h1.Show()

