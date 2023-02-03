#!/bin/env python

import os
import glob
import sys

#___________________________________________________________________________________________________
def write_pages_v2(directory, objecttypes):

    os.system("mkdir -p {directory}".format(directory=directory))

    ##############################################################################################################################

    # eff
    pdgids = [0, 11, 13, 211, 321]
    sels = ["base", "loweta", "xtr", "vtr"]
    variables = [
            "pt",
            "ptzoom",
            "ptlow",
            "ptlowzoom",
            "ptmtv",
            "ptmtvzoom",
            "eta",
            "etazoom",
            "etacoarse",
            "etacoarsezoom",
            "phi",
            "phizoom",
            "phicoarse",
            "phicoarsezoom",
            "dxy",
            "dxycoarse",
            "dxycoarsezoom",
            "dz",
            "dzcoarse",
            "dzcoarsezoom",
            ]
    breakdowns = ["_breakdown"]
    charges = [0, -1, 1]

    plotdir = "../mtv"
    plot_width = 250
    plot_large_width = 600

    index_md = open("{directory}/index.md".format(directory=directory), "w")
    index_md.write("# LST Performance\n\n")

    ##############################################################################################################################

    metric = "eff"
    for objecttype in objecttypes:
        index_md.write("## {}\n\n".format(objecttype))
        index_md.write("### Efficiencies\n\n")
        for breakdown in breakdowns:
            for selection in sels:
                index_md.write("#### For {selectionstr}\n\n".format(selectionstr=get_selectionstr(selection)))
                for charge in charges:
                    summary_file_name = "{objecttype}_{metric}_{selection}_{charge}".format(objecttype=objecttype,metric=metric, selection=selection, charge=charge)
                    summary_markdown = open("{directory}/{summary_file_name}.md".format(directory=directory, summary_file_name=summary_file_name), "w")
                    TOC = {}
                    SectionID = 0
                    page_name = "Summary Plots of LST {objecttype} Efficiency for {selectionstr} with Charge={chargestr}".format(objecttype=objecttype, selectionstr=get_selectionstr(selection), chargestr=get_chargestr(charge))
                    index_md.write("[Charge={chargestr}]({summary_file_name}.html)\n\n".format(selectionstr=get_selectionstr(selection), chargestr=get_chargestr(charge), summary_file_name=summary_file_name))
                    for pdgid in pdgids:
                        SectionID += 1
                        pdgidstr = get_pdgidstr(pdgid)
                        SectionTitle = "{SectionID} {objecttype} {pdgidstr} {metricstr}".format(SectionID=SectionID, objecttype=objecttype, pdgidstr=pdgidstr, metricstr=get_metricstr(metric))
                        summary_markdown.write("\n\n## <a name=\"{SectionID}\"></a> {SectionTitle}\n\n [[back to top](#top)]\n\n".format(SectionTitle=SectionTitle, SectionID=SectionID))
                        TOC["#{SectionID}".format(SectionID=SectionID)] = SectionTitle
                        for variable in variables:
                            name = "{objecttype}_{selection}_{pdgid}_{charge}_{metric}_{variable}".format(objecttype=objecttype, selection=selection, pdgid=pdgid, charge=charge, metric=metric, variable=variable)
                            html = "{name}.html".format(name=name)
                            md = "{name}.md".format(name=name)
                            f = open("{directory}/{md}".format(directory=directory, md=md), "w")
                            f.write("# {objecttype} {metricstr} vs. {variable}\n\n[[back to main](./)]\n\n".format(objecttype=objecttype, metricstr=get_metricstr(metric), variable=variable))
                            f.write("\n\n")
                            f.write("## Ratio\n\n[![Ratio]({plotdir}/var/{name}.png){{ width={plot_large_width}px }}]({plotdir}/var/{name}.pdf)\n\n".format(plotdir=plotdir, name=name, plot_large_width=plot_large_width))
                            if len(breakdown) != 0:
                                for i in range(5):
                                    f.write("## Denominator {i}\n\n[![Denominator]({plotdir}/den/{name}_den{i}.png){{ width={plot_large_width}px }}]({plotdir}/den/{name}_den{i}.pdf)\n\n".format(plotdir=plotdir, name=name, plot_large_width=plot_large_width, i=i))
                                    f.write("## Numerator {i}\n\n[![Numerator]({plotdir}/num/{name}_num{i}.png){{ width={plot_large_width}px }}]({plotdir}/num/{name}_num{i}.pdf)\n\n".format(plotdir=plotdir, name=name, plot_large_width=plot_large_width, i=i))
                                for i in range(4):
                                    f.write("## Double Ratio {i}\n\n[![Double Ratio]({plotdir}/ratio/{name}_ratio{i}.png){{ width={plot_large_width}px }}]({plotdir}/ratio/{name}_ratio{i}.pdf)\n\n".format(plotdir=plotdir, name=name, plot_large_width=plot_large_width, i=i))
                            else:
                                f.write("## Numerator\n\n[![Numerator]({plotdir}/num/{name}_num0.png){{ width={plot_large_width}px }}]({plotdir}/num/{name}_num0.pdf)\n\n".format(plotdir=plotdir, name=name, plot_large_width=plot_large_width, i=i))
                            f.close()
                            summary_markdown.write("[![]({plotdir}/var/{name}.png){{ width={plot_width}px }}]({html})\n".format(plotdir=plotdir, name=name, plot_width=plot_width, html=html))

                    summary_markdown.close()

                    # Reopen and add TOC at the top
                    tmp = open("{directory}/{summary_file_name}.md".format(directory=directory, summary_file_name=summary_file_name))
                    lines = tmp.readlines()
                    tmp.close()

                    header_lines = []
                    header_lines.append("[[back to main](./)]\n\n")
                    header_lines.append("# <a name=\"top\"></a> {page_name}\n".format(page_name=page_name))
                    header_lines.append("\n")
                    for key in sorted(TOC.keys()):
                        header_lines.append("[{SectionTitle}]({key})<br/>".format(SectionTitle=TOC[key], key=key))

                    newlines = header_lines + ["\n\n"] + lines

                    summary_markdown = open("{directory}/{summary_file_name}.md".format(directory=directory, summary_file_name=summary_file_name), "w")
                    for line in newlines:
                        summary_markdown.write(line)

                    summary_markdown.close()


    ##############################################################################################################################

    recometrics = ["fakerate", "duplrate"]

    for metric in recometrics:

        index_md.write("### {metricstr}\n\n".format(metricstr=get_metricstr(metric)))

        variables = [
                "pt",
                "ptzoom",
                "ptlow",
                "ptlowzoom",
                "ptmtv",
                "ptmtvzoom",
                "eta",
                "etazoom",
                "etacoarse",
                "etacoarsezoom",
                "phi",
                "phizoom",
                "phicoarse",
                "phicoarsezoom",
                ]

        for breakdown in breakdowns:
            summary_file_name = "{metric}".format(metric=metric)
            summary_markdown = open("{directory}/{summary_file_name}.md".format(directory=directory, summary_file_name=summary_file_name), "w")
            TOC = {}
            SectionID = 0
            page_name = "Summary Plots of LST {metricstr}".format(metricstr=get_metricstr(metric))
            index_md.write("[{metricstr}]({summary_file_name}.html)\n\n".format(metricstr=get_metricstr(metric), summary_file_name=summary_file_name))
            for objecttype in objecttypes:
                SectionID += 1
                pdgidstr = get_pdgidstr(pdgid)
                SectionTitle = "{SectionID} {objecttype} {metricstr}".format(SectionID=SectionID, objecttype=objecttype, metricstr=get_metricstr(metric))
                summary_markdown.write("\n\n## <a name=\"{SectionID}\"></a> {SectionTitle}\n\n [[back to top](#top)]\n\n".format(SectionTitle=SectionTitle, SectionID=SectionID))
                TOC["#{SectionID}".format(SectionID=SectionID)] = SectionTitle
                for variable in variables:
                    name = "{objecttype}_{metric}_{variable}".format(objecttype=objecttype, metric=metric, variable=variable)
                    html = "{name}.html".format(name=name)
                    md = "{name}.md".format(name=name)
                    f = open("{directory}/{md}".format(directory=directory, md=md), "w")
                    f.write("# {objecttype} {metricstr} vs. {variable}\n\n[[back to main](./)]\n\n".format(objecttype=objecttype, metricstr=get_metricstr(metric), variable=variable))
                    f.write("\n\n")
                    f.write("## Ratio\n\n[![Ratio]({plotdir}/var/{name}.png){{ width={plot_large_width}px }}]({plotdir}/var/{name}.pdf)\n\n".format(plotdir=plotdir, name=name, plot_large_width=plot_large_width))
                    if len(breakdown) != 0:
                        for i in range(5):
                            f.write("## Denominator {i}\n\n[![Denominator]({plotdir}/den/{name}_den{i}.png){{ width={plot_large_width}px }}]({plotdir}/den/{name}_den{i}.pdf)\n\n".format(plotdir=plotdir, name=name, plot_large_width=plot_large_width, i=i))
                            f.write("## Numerator {i}\n\n[![Numerator]({plotdir}/num/{name}_num{i}.png){{ width={plot_large_width}px }}]({plotdir}/num/{name}_num{i}.pdf)\n\n".format(plotdir=plotdir, name=name, plot_large_width=plot_large_width, i=i))
                        for i in range(4):
                            f.write("## Double Ratio {i}\n\n[![Double Ratio]({plotdir}/ratio/{name}_ratio{i}.png){{ width={plot_large_width}px }}]({plotdir}/ratio/{name}_ratio{i}.pdf)\n\n".format(plotdir=plotdir, name=name, plot_large_width=plot_large_width, i=i))
                    else:
                        f.write("## Numerator\n\n[![Numerator]({plotdir}/num/{name}_num0.png){{ width={plot_large_width}px }}]({plotdir}/num/{name}_num0.pdf)\n\n".format(plotdir=plotdir, name=name, plot_large_width=plot_large_width, i=i))
                    f.close()
                    summary_markdown.write("[![]({plotdir}/var/{name}.png){{ width={plot_width}px }}]({html})\n".format(plotdir=plotdir, name=name, plot_width=plot_width, html=html))

            summary_markdown.close()

            # Reopen and add TOC at the top
            tmp = open("{directory}/{summary_file_name}.md".format(directory=directory, summary_file_name=summary_file_name))
            lines = tmp.readlines()
            tmp.close()

            header_lines = []
            header_lines.append("[[back to main](./)]\n\n")
            header_lines.append("# <a name=\"top\"></a> {page_name}\n".format(page_name=page_name))
            header_lines.append("\n")
            for key in sorted(TOC.keys()):
                header_lines.append("[{SectionTitle}]({key})<br/>".format(SectionTitle=TOC[key], key=key))

            newlines = header_lines + ["\n\n"] + lines

            summary_markdown = open("{directory}/{summary_file_name}.md".format(directory=directory, summary_file_name=summary_file_name), "w")
            for line in newlines:
                summary_markdown.write(line)

            summary_markdown.close()

#___________________________________________________________________________________________________
def get_pdgidstr(pdgid):
    if pdgid == 0:
        pdgidstr = "All"
    if pdgid == 11:
        pdgidstr = "Electron"
    if pdgid == 13:
        pdgidstr = "Muon"
    if pdgid == 211:
        pdgidstr = "Pion"
    if pdgid == 321:
        pdgidstr = "Kaon"
    return pdgidstr

#___________________________________________________________________________________________________
def get_chargestr(charge):
    if charge == 0:
        chargestr = "Both"
    if charge == 1:
        chargestr = "Positive"
    if charge == -1:
        chargestr = "Negative"
    return chargestr

#___________________________________________________________________________________________________
def get_selectionstr(selection):
    if selection == "base":
        selectionstr = "|eta| < 4.5"
    if selection == "loweta":
        selectionstr = "|eta| < 2.4"
    if selection == "xtr":
        selectionstr = "1.1 < |eta| < 2.7"
    if selection == "vtr":
        selectionstr = "not (1.1 < |eta| < 2.7) and |eta| < 2.4"
    return selectionstr

#___________________________________________________________________________________________________
def get_metricstr(metric):
    if metric == "eff":
        metricstr = "Efficiency"
    if metric == "fakerate":
        metricstr = "Fake Rate"
    if metric == "duplrate":
        metricstr = "Duplicate Rate"
    return metricstr

#___________________________________________________________________________________________________
def write_footnote(ff):
    ff.write("""
``` {=html}
<style>
body { min-width: 100% !important; }
</style>
```
""")

if __name__ == "__main__":

    write_pages_v2("summary", ["TC", "pT5_lower", "pT3_lower", "T5_lower"])
    write_pages_v2("compare", ["TC", "pT5", "pT3", "T5", "pLS", "pT5_lower", "pT3_lower", "T5_lower"])
