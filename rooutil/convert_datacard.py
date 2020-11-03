#!/bin/env python

import sys
import ROOT as r

"""
This class converts a datacard that is utilizing histograms to fit multi-bin SR analysis into a single bin text-based data cards.
e.g.
dc = DataCardConverter("datacard.txt", 3)
                                       ^ choose 3rd bin of multi-bin SR histogram
"""

class DataCardConverter:

    """ Converts histogram based datacards to txt based datacards."""

    def __init__(self, datacard_path, bin_number=1, bin_name="SR"):
        """ The path is given when the converter is initialized."""
        self.datacard_path = datacard_path
        self.bin_number = bin_number # This is the bin number for the yield histograms
        self.bin_name = bin_name # This is the label used to label the channel

        self.open_datacard_file()
        self.get_datacard_lines()
        self.get_shape_systematic_configuration_line()
        self.retrieve_histogram_root_file_path()
        self.get_systematic_format()
        self.open_hist_rootfile()
        self.get_process_names_and_indices()
        self.process_nominal_rates()
        self.get_shape_systematic_lines()
        self.process_systematics()
        #self.print_contents()

    def get_str(self):
        return self.get_header() + self.get_rates_table() + self.get_systematic_table()

    def open_datacard_file(self):
        self.f = open(self.datacard_path)

    def get_datacard_lines(self):
        self.datacard_lines = [ x.strip() for x in self.f.readlines() ]

    def get_shape_systematic_configuration_line(self):
        """ Line that looks like "shapes * * hist.root ... " contains the configuration of the shape systematics."""
        for line in self.datacard_lines:
            if "shapes" in line:
                self.shape_config_line = line
                break

    def retrieve_histogram_root_file_path(self):
        """ The shape systematic configuration line contains the path to the rootfile histograms."""
        self.hist_rootfile_path = self.shape_config_line.split()[3]

    def get_systematic_format(self):
        """ The histogram naming convention for shape systematic histograms are defined in the shape systematic configuration line."""
        tmp = self.shape_config_line.split()[5]
        if tmp != "$PROCESS_$SYSTEMATIC":
            raise ValueError("DataCardConverter currently does not know how to deal with shape systematic histogram naming convention other than $PROCESS_$SYSTEMATIC. (user had set to {})".format(tmp))
        self.shape_systematic_format = "{PROCESS}_{SYSTEMATIC}"

    def open_hist_rootfile(self):
        self.hf = r.TFile(self.hist_rootfile_path)

    def get_process_names_and_indices(self):
        """ There are two lines that begin with "process". one of them has indices, while the other has the label for the process."""
        """ Use the fact that the line with indices always start like "process    0   1   2  .... "."""
        self.lines = []
        for line in self.datacard_lines:
            if "process" in line.split():
                self.lines.append(line)
        which_line_has_labels  = 1 if self.lines[0].split()[1] == "0" else 0
        which_line_has_indices = 0 if self.lines[0].split()[1] == "0" else 1

        """ After parsing create 3 objects: a list of process_names, a list of indices, a map of index to process name."""
        self.process_names = self.lines[which_line_has_labels].split()[1:]
        self.process_indices = [ int(x) for x in self.lines[which_line_has_indices].split()[1:] ]
        self.process_names_indices_map = {}
        for name, i in zip(self.process_names, self.process_indices):
            self.process_names_indices_map[i] = name

    def process_nominal_rates(self):
        """ Go through nominal histograms and obtain the main rates"""
        self.rates = {}
        for proc in self.process_names:
            self.rates[proc] = self.get_rate(proc)

        """ Get data rates too"""
        self.rates["data_obs"] = self.get_rate("data_obs")

    def get_rate(self, proc, syst=""):
        """ From the histograms, retrieve the yields."""
        if syst:
            h = self.hf.Get(self.shape_systematic_format.format(PROCESS=proc, SYSTEMATIC=syst))
        else:
            h = self.hf.Get(proc)
        return h.GetBinContent(self.bin_number)

    def get_frac_error(self, proc, syst):
        """ From the nominal and systematic histograms compute the fractional error for the datacard"""
        rate_nom = self.get_rate(proc)
        rate_sys = self.get_rate(proc, syst)
        frac_err = rate_sys / rate_nom if rate_nom != 0 else 1
        if frac_err <= 1e-3:
            frac_err = 1e-3
        return frac_err

    def get_shape_systematic_lines(self):
        """ Identify lines that refer to systematics and save them"""
        self.shape_systematic_lines = []
        for line in self.datacard_lines:
            if len(line.split()) > 1:
                if "shape" in line.split()[1]:
                    self.shape_systematic_lines.append(line)

    def process_systematics(self):
        """ Loop over the systematics"""
        self.systs = {}

        """ Loop over the shape systematic lines and create place holders for systematic values."""
        """ Nested map to hold all the systematic values (map - map - map) where it will be key'd as [proc][syst]["Up"] or [proc][syst]["Down"] """
        for line in self.shape_systematic_lines:
            syst_name = line.split()[0]
            self.systs[syst_name] = {}
            for index in self.process_names_indices_map:
                proc = self.process_names_indices_map[index]
                self.systs[syst_name][proc] = {"Up" : 1., "Down" : 1.}

        """ Now loop over the shape systematic lines and process them."""
        """ Each line will have something like "systname   shape   1   -    -   1   -  -  " """
        """ 1 means the multiplier to the systematic is 1, and - means no systematic for that process."""
        for line in self.shape_systematic_lines:
            for index, multiplier in enumerate(line.split()[2:]): # The multiplier will most likely will be 1 or -. could have other multiplier values but unlikely.
                proc = self.process_names_indices_map[index]
                syst = line.split()[0]
                if multiplier == "-": # if the multiplier is set to "-" that means this process does not have any shape systematics applied
                    self.systs[syst][proc]["Up"] = "-"
                    self.systs[syst][proc]["Down"] = "-"
                else:
                    multiplier = float(multiplier)
                    if multiplier != 1:
                        raise ValueError("DataCardConverter does not know what to do When the value set for the shape systematic is not equal to 1 or - (see your datacard for {})".format(line))
                    self.systs[syst][proc]["Up"] = self.get_frac_error(proc, syst+"Up")
                    self.systs[syst][proc]["Down"] = self.get_frac_error(proc, syst+"Down")

    def get_delimiter(self):
        delimiter = "-" * 38
        for index in self.process_names_indices_map:
            delimiter += "-" * 17
        return delimiter

    def get_header(self):
        header = ""
        header += "imax 1  number of channels\n"
        header += "jmax *  number of backgrounds ('*' = automatic)\n"
        header += "kmax *  number of nuisance parameters (sources of systematical uncertainties)\n"
        header += "{}\n".format(self.get_delimiter())
        header += "bin            {}\n".format(self.bin_name)
        header += "observation    {}\n".format(self.rates["data_obs"])
        header += "{}\n".format(self.get_delimiter())
        return header

    def get_rates_table(self):
        """ Print the top portion of the table where the rates are listed (and the process name)"""
        """ loop over process and print the rates"""
        proc_bin_line   = "{:38s}".format("bin")
        proc_index_line = "{:38s}".format("process")
        proc_label_line = "{:38s}".format("process")
        proc_rates_line = "{:38s}".format("rate")
        for index in self.process_names_indices_map:
            proc_name = self.process_names_indices_map[index]
            proc_bin_line   += "{:17s}".format(self.bin_name)
            proc_index_line += "{:17s}".format(str(index))
            proc_label_line += "{:17s}".format(str(proc_name))
            proc_rates_line += "{:.5f}{:10s}".format(self.rates[proc_name],"")
        rtn  = proc_bin_line + "\n"
        rtn += proc_index_line + "\n"
        rtn += proc_label_line + "\n"
        rtn += proc_rates_line + "\n"
        rtn += self.get_delimiter() + "\n"
        return rtn

    def get_systematic_table(self):
        """ Print the main systematic error table portion"""
        """ loop over systematics and process and print the table"""
        rtnstr = ""
        for syst in sorted(self.systs.keys()):
            tmpstr = "{:23s}lnN            ".format(syst)
            for index in self.process_names_indices_map:
                proc = self.process_names_indices_map[index]
                if self.systs[syst][proc]["Up"] == "-":
                    tmpstr += "{:17s}".format("-")
                else:
                    tmpstr += "{:.5f}/{:.5f}  ".format(self.systs[syst][proc]["Down"], self.systs[syst][proc]["Up"])
            rtnstr += tmpstr + "\n"
        return rtnstr

    def print_contents(self):
        print self.f
        for line in self.datacard_lines:
            print line
        print self.shape_config_line
        print self.hist_rootfile_path
        print self.shape_systematic_format
        print self.hf
        for proc in self.process_names:
            print proc
        for proc in self.process_names:
            print proc, self.rates[proc]
        for syst_line in self.shape_systematic_lines:
            print syst_line
        for proc in self.systs:
            for syst in self.systs[proc]:
                if isinstance(self.systs[proc][syst]["Up"], str):
                    print proc, syst, "{:s}".format(self.systs[proc][syst]["Up"])+"/"+"{:s}".format(self.systs[proc][syst]["Down"])
                else:
                    print proc, syst, "{:.5f}".format(self.systs[proc][syst]["Up"])+"/"+"{:.5f}".format(self.systs[proc][syst]["Down"])

if __name__ == "__main__":

    datacard_path = sys.argv[1]

    dc = DataCardConverter(datacard_path, 3)
    print dc.get_str()

