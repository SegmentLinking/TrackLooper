
import numpy as np

def clopper_pearson(passed,total,level=0.6827):
    """
    matching TEfficiency::ClopperPearson()
    """
    import scipy.stats
    alpha = 0.5*(1.-level)
    low = scipy.stats.beta.ppf(alpha, passed, total-passed+1)
    high = scipy.stats.beta.ppf(1 - alpha, passed+1, total-passed)
    return low, high

class Hist1D(object):

    def __init__(self, obj=None, **kwargs):
        tstr = str(type(obj))

        self._counts = None
        self._edges = None
        self._errors = None
        self._errors_up = None    # only handled when dividing with binomial errors
        self._errors_down = None  # only handled when dividing with binomial errors
        if "ROOT." in tstr:
            self.init_root(obj,**kwargs)
        elif "ndarray" in tstr:
            self.init_numpy(obj,**kwargs)

    def init_numpy(self, obj, **kwargs):
        if "errors" in kwargs:
            self._errors = kwargs["errors"]
            del kwargs["errors"]

        self._counts, self._edges = np.histogram(obj,**kwargs)
        self._counts = self._counts.astype(np.float64)

        # poisson defaults if not specified
        if self._errors is None:
            self._errors = np.sqrt(self._counts)
        self._errors = self._errors.astype(np.float64)

    def init_root(self, obj, **kwargs):
        low_edges = np.array([1.0*obj.GetBinLowEdge(ibin) for ibin in range(obj.GetNbinsX()+1)])
        bin_widths = np.array([1.0*obj.GetBinWidth(ibin) for ibin in range(obj.GetNbinsX()+1)])
        self._counts = np.array([1.0*obj.GetBinContent(ibin) for ibin in range(1,obj.GetNbinsX()+1)],dtype=np.float64)
        self._errors = np.array([1.0*obj.GetBinError(ibin) for ibin in range(1,obj.GetNbinsX()+1)],dtype=np.float64)
        self._edges = low_edges + bin_widths

    def get_errors(self):
        return self._errors

    def get_errors_up(self):
        return self._errors_up

    def get_errors_down(self):
        return self._errors_down

    def get_counts(self):
        return self._counts

    def get_counts_errors(self):
        return self._counts, self._errors

    def get_edges(self):
        return self._edges

    def get_bin_centers(self):
        return 0.5*(self._edges[1:]+self._edges[:-1])

    def get_bin_widths(self):
        return self._edges[1:]-self._edges[:-1]

    def get_integral(self):
        return np.sum(self._counts)

    def _check_consistency(self, other):
        if len(self._edges) != len(other._edges):
            raise Exception("These histograms cannot be combined due to different binning")

    def __add__(self, other):
        self._check_consistency(other)
        hnew = Hist1D()
        hnew._counts = self._counts + other._counts
        hnew._errors = (self._errors**2. + other._errors**2.)**0.5
        hnew._edges = self._edges
        return hnew

    def __sub__(self, other):
        self._check_consistency(other)
        hnew = Hist1D()
        hnew._counts = self._counts - other._counts
        hnew._errors = (self._errors**2. + other._errors**2.)**0.5
        hnew._edges = self._edges
        return hnew

    def __div__(self, other):
        return self.divide(other)

    def divide(self, other, binomial=False):
        self._check_consistency(other)
        hnew = Hist1D()
        hnew._edges = self._edges
        if not binomial:
            with np.errstate(divide="ignore",invalid="ignore"):
                hnew._counts = self._counts / other._counts
                hnew._errors = (
                        (self._errors/other._counts)**2.0 +
                        (other._errors*self._counts/(other._counts)**2.0)**2.0
                        )**0.5
        else:
            hnew._errors_down, hnew._errors_up = clopper_pearson(self._counts,other._counts)
            hnew._counts = self._counts/other._counts
            hnew._errors = 0.*hnew._counts
            # these are actually the positions for down and up, but we want the errors
            # wrt to the central value
            with np.errstate(divide="ignore",invalid="ignore"):
                hnew._errors_up = hnew._errors_up - hnew._counts
                hnew._errors_down = hnew._counts - hnew._errors_down
        return hnew

    def __mul__(self, fact):
        if type(fact) in [float,int]:
            self._counts *= fact
            self._errors *= fact**0.5
            return self
        else:
            raise Exception("Can't multiply histogram by non-scalar")

    __rmul__ = __mul__

    def __repr__(self):
        use_ascii = False
        if use_ascii: sep = "+-"
        else: sep = u"\u00B1".encode("utf-8")
        # trick: want to use numpy's smart formatting (truncating,...) of arrays
        # so we convert value,error into a complex number and format that 1D array :)
        formatter = {"complex_kind": lambda x:"%.2f {} %.2f".format(sep) % (np.real(x),np.imag(x))}
        a2s = np.array2string(self._counts+self._errors*1j,formatter=formatter, suppress_small=True, separator="   ")
        return "<Hist1D:\n{}\n>".format(a2s)

if __name__ == "__main__":
    np.random.seed(42)

    # make a root histogram with 10k gaussians, and convert it into a Hist1D object
    nbins = 20
    N = 10000
    import ROOT as r
    hroot = r.TH1F("h1","h1",nbins,-3,3)
    hroot.FillRandom("gaus",N)
    h1 = Hist1D(hroot)

    # make a Hist1D object out of a numpy array of 10k gaussians, with same binning
    h2 = Hist1D(np.random.normal(0,1,N),bins=np.linspace(-3,3,nbins+1))

    print "Nice repr... h1/h2:"
    print h1/h2

    print "_" * 40

    gaus1 = np.random.normal(0,1,N)
    gaus2 = gaus1[gaus1 > np.random.normal(0,1,N)]
    g1 = Hist1D(gaus1,bins=np.linspace(-3,3,nbins+1))
    g2 = Hist1D(gaus2,bins=np.linspace(-3,3,nbins+1))
    print g1
    print g2
    g2 = g2.divide(g1,binomial=True)
    print g2
    print g2.get_errors_up()
    print g2.get_errors_down()

