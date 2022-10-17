#!/bin/env python

import os
import glob

def write_footnote(ff):
    ff.write("""
``` {=html}
<style>
body { min-width: 100% !important; }
</style>
```
""")

def write_pages(
            plot_width,
            plot_large_width,
            plotdir,
            directory,
            pdgids,
            Sections,
            Plots
        ):


    if not os.path.exists(directory):
        os.makedirs(directory)

    summary_markdown = open("{}/index.md".format(directory), "w")
    TOC={}
    SectionID = 0
    for ObjectType in Sections:
        SectionID += 1
        SectionSubID = 0
        for Metric in Plots:
            for pdgid in pdgids:
                print(ObjectType, Metric, pdgid)
                # PDGID variations are only for efficiency plots
                if Metric != "Efficiency" and pdgid != 0:
                    continue
                prefix = Plots[Metric][0]
                xvars = Plots[Metric][1]
                SectionSubID += 1
                Stacked = ""
                ObjectTypeWSuffix = ObjectType + ("_{}".format(pdgid) if Metric == "Efficiency" else "")
                objectTypeShortName = ObjectTypeWSuffix
                if "Stack" in ObjectTypeWSuffix:
                    objectTypeShortName = ObjectTypeWSuffix.replace("Stack", "")
                    prefix = prefix + "_stack"
                    Stacked = " Breakdown"
                if pdgid == 0:
                    pdgidstr = "All"
                if pdgid == 11:
                    pdgidstr = "Electron"
                if pdgid == 13:
                    pdgidstr = "Muon"
                if pdgid == 211:
                    pdgidstr = "Pion"
                objectTypeTitleName = objectTypeShortName.split("_")[0] + (" {}".format(pdgidstr) if Metric == "Efficiency" else "")
                SectionTitle = "{SectionID}.{SectionSubID} {objectTypeTitleName} {Metric}{Stacked}".format(SectionID=SectionID, SectionSubID=SectionSubID, objectTypeTitleName=objectTypeTitleName, Metric=Metric, Stacked=Stacked)
                summary_markdown.write("## <a name=\"{SectionID}.{SectionSubID}\"></a> {SectionTitle}\n\n [[back to top](#top)]\n".format(SectionTitle=SectionTitle, SectionID=SectionID, SectionSubID=SectionSubID))
                TOC["#{SectionID}.{SectionSubID}".format(SectionID=SectionID, SectionSubID=SectionSubID)] = SectionTitle
                summary_markdown.write("\n")
                for xvar in xvars:
                    smallhtml = "{ObjectTypeWSuffix}_{prefix}_{xvar}.html".format(ObjectTypeWSuffix=ObjectTypeWSuffix, prefix=prefix, xvar=xvar)
                    smallmd = "{ObjectTypeWSuffix}_{prefix}_{xvar}.md".format(ObjectTypeWSuffix=ObjectTypeWSuffix, prefix=prefix, xvar=xvar)
                    summary_markdown.write("[![]({plotdir}/var/{objectTypeShortName}_{prefix}_{xvar}.png){{ width={plot_width}px }}]({smallhtml})\n".format(plotdir=plotdir, objectTypeShortName=objectTypeShortName, prefix=prefix, xvar=xvar, plot_width=plot_width, smallhtml=smallhtml))
                    f = open("{directory}/{smallmd}".format(directory=directory, smallmd=smallmd), "w")
                    f.write("# {objectTypeTitleName} {Metric} vs. {xvar}\n\n[[back to main](./)]\n\n".format(objectTypeTitleName=objectTypeTitleName, Metric=Metric, xvar=xvar))
                    f.write("\n\n")
                    f.write("## Ratio\n\n[![Ratio]({plotdir}/var/{objectTypeShortName}_{prefix}_{xvar}.png){{ width={plot_large_width}px }}]({plotdir}/var/{objectTypeShortName}_{prefix}_{xvar}.pdf)\n\n".format(plotdir=plotdir, objectTypeShortName=objectTypeShortName, prefix=prefix, xvar=xvar, plot_large_width=plot_large_width))
                    f.write("## Numerator\n\n[![Numerator]({plotdir}/num/{objectTypeShortName}_{prefix}_{xvar}_num0.png){{ width={plot_large_width}px }}]({plotdir}/num/{objectTypeShortName}_{prefix}_{xvar}_num0.pdf)\n\n".format(plotdir=plotdir, objectTypeShortName=objectTypeShortName, prefix=prefix, xvar=xvar, plot_large_width=plot_large_width))
                    f.write("## Denominator\n\n[![Denominator]({plotdir}/den/{objectTypeShortName}_{prefix}_{xvar}_den.png){{ width={plot_large_width}px }}]({plotdir}/den/{objectTypeShortName}_{prefix}_{xvar}_den.pdf)\n\n".format(plotdir=plotdir, objectTypeShortName=objectTypeShortName, prefix=prefix, xvar=xvar, plot_large_width=plot_large_width))
                    write_footnote(f)
                    f.close()
                summary_markdown.write("\n")

    write_footnote(summary_markdown)

    summary_markdown.close()


    # Reopen and add TOC at the top
    tmp = open("{directory}/index.md".format(directory=directory))
    lines = tmp.readlines()
    tmp.close()

    header_lines = []
    header_lines.append("# <a name=\"top\"></a> Summary Plots of LST Performance\n")
    header_lines.append("\n")
    for key in sorted(TOC.keys()):
        header_lines.append("[{SectionTitle}]({key})<br/>".format(SectionTitle=TOC[key], key=key))

    newlines = header_lines + ["\n\n"] + lines

    summary_markdown = open("{}/index.md".format(directory), "w")
    for line in newlines:
        summary_markdown.write(line)

    summary_markdown.close()



if __name__ == "__main__":

    plot_width = 450
    plot_large_width = 600
    plotdir = "../mtv"

    directory = "summary"
    pdgids = [0, 11, 13, 211]
    Sections = ["TC", "TCStack", "pT5", "pT3", "T5", "pLS"]
    Plots = {
            "Efficiency":
                [
                    "eff",
                    [
                        "pt",
                        "ptzoom",
                        "ptlow",
                        "ptlowzoom",
                        "ptmtv",
                        "ptmtvzoom",
                        "etacoarse",
                        "etacoarsezoom",
                        "eta",
                        "etazoom",
                        "dxycoarse",
                        "dzcoarse",
                        "dxy",
                        "dz",
                        "phi",
                    ],
                ],
            "Fake Rate":
                [
                    "fakerate",
                    [
                        "pt",
                        "ptzoom",
                        "ptlow",
                        "ptlowzoom",
                        "ptmtv",
                        "ptmtvzoom",
                        "etacoarse",
                        "etacoarsezoom",
                        "eta",
                        "etazoom",
                        "phi",
                    ],
                ],
            "Duplicate Rate":
                [
                    "duplrate",
                    [
                        "pt",
                        "ptzoom",
                        "ptlow",
                        "ptlowzoom",
                        "ptmtv",
                        "ptmtvzoom",
                        "etacoarse",
                        "etacoarsezoom",
                        "eta",
                        "etazoom",
                        "phi",
                    ],
                ],
            }

    write_pages(
            plot_width,
            plot_large_width,
            plotdir,
            directory,
            pdgids,
            Sections,
            Plots,
            )

    # compare
    Sections.remove("TCStack")
    pdgids = [0]
    directory = "compare"
    write_pages(
            plot_width,
            plot_large_width,
            plotdir,
            directory,
            [0],
            Sections,
            Plots,
            )

