#!/bin/env python

import os
import glob

plot_width = 450
plot_large_width = 600
plotdir = "../mtv"

Sections = ["TC", "TCStack", "pT5", "pT3", "T5", "pLS"]
Plots = {
        "Efficiency":
            [
                "eff",
                [
                    "pt",
                    "ptzoom",
                    "etacoarse",
                    "etacoarsezoom",
                    "dxycoarse",
                    "dzcoarse",
                    "phi",
                ],
            ],
        "Fake Rate":
            [
                "fakerate",
                [
                    "pt",
                    "ptzoom",
                    "etacoarse",
                    "etacoarsezoom",
                    "phi",
                ],
            ],
        "Duplicate Rate":
            [
                "duplrate",
                [
                    "pt",
                    "ptzoom",
                    "etacoarse",
                    "etacoarsezoom",
                    "phi",
                ],
            ],
        }

def write_footnote(ff):
    ff.write("""
``` {=html}
<style>
body { min-width: 100% !important; }
</style>
```
""")


directory = "summary"
if not os.path.exists(directory):
    os.makedirs(directory)

summary_markdown = open("summary/index.md", "w")
TOC={}
SectionID = 0
for ObjectType in Sections:
    SectionID += 1
    SectionSubID = 0
    for Metric in Plots:
        prefix = Plots[Metric][0]
        xvars = Plots[Metric][1]
        SectionSubID += 1
        objectTypeShortName = ObjectType
        Stacked = ""
        if "Stack" in ObjectType:
            objectTypeShortName = ObjectType.replace("Stack", "")
            prefix = prefix + "_stack"
            Stacked = " Stacked"
        SectionTitle = "{SectionID}.{SectionSubID} {objectTypeShortName} {Metric}{Stacked}".format(SectionID=SectionID, SectionSubID=SectionSubID, objectTypeShortName=objectTypeShortName, Metric=Metric, Stacked=Stacked)
        summary_markdown.write("## <a name=\"{SectionID}.{SectionSubID}\"></a> {SectionTitle}\n".format(SectionTitle=SectionTitle, SectionID=SectionID, SectionSubID=SectionSubID))
        TOC["#{SectionID}.{SectionSubID}".format(SectionID=SectionID, SectionSubID=SectionSubID)] = SectionTitle
        summary_markdown.write("\n")
        for xvar in xvars:
            smallhtml = "{ObjectType}_{prefix}_{xvar}.html".format(ObjectType=ObjectType, prefix=prefix, xvar=xvar)
            smallmd = "{ObjectType}_{prefix}_{xvar}.md".format(ObjectType=ObjectType, prefix=prefix, xvar=xvar)
            summary_markdown.write("[![]({plotdir}/var/{objectTypeShortName}_{prefix}_{xvar}.png){{ width={plot_width}px }}]({smallhtml})\n".format(plotdir=plotdir, objectTypeShortName=objectTypeShortName, prefix=prefix, xvar=xvar, plot_width=plot_width, smallhtml=smallhtml))
            f = open("summary/{smallmd}".format(smallmd=smallmd), "w")
            f.write("# {objectTypeShortName} {Metric} vs. {xvar}\n\n".format(objectTypeShortName=objectTypeShortName, Metric=Metric, xvar=xvar))
            f.write("## Ratio\n\n[![Ratio]({plotdir}/var/{objectTypeShortName}_{prefix}_{xvar}.png){{ width={plot_large_width}px }}]({plotdir}/var/{objectTypeShortName}_{prefix}_{xvar}.pdf)\n\n".format(plotdir=plotdir, objectTypeShortName=objectTypeShortName, prefix=prefix, xvar=xvar, plot_large_width=plot_large_width))
            f.write("## Numerator\n\n[![Numerator]({plotdir}/num/{objectTypeShortName}_{prefix}_{xvar}_num.png){{ width={plot_large_width}px }}]({plotdir}/num/{objectTypeShortName}_{prefix}_{xvar}_num.pdf)\n\n".format(plotdir=plotdir, objectTypeShortName=objectTypeShortName, prefix=prefix, xvar=xvar, plot_large_width=plot_large_width))
            f.write("## Denominator\n\n[![Denominator]({plotdir}/den/{objectTypeShortName}_{prefix}_{xvar}_den.png){{ width={plot_large_width}px }}]({plotdir}/den/{objectTypeShortName}_{prefix}_{xvar}_den.pdf)\n\n".format(plotdir=plotdir, objectTypeShortName=objectTypeShortName, prefix=prefix, xvar=xvar, plot_large_width=plot_large_width))
            write_footnote(f)
            f.close()
        summary_markdown.write("\n")

write_footnote(summary_markdown)

summary_markdown.close()


# Reopen and add TOC at the top
tmp = open("summary/index.md")
lines = tmp.readlines()
tmp.close()

header_lines = []
header_lines.append("# Summary Plots of LST Performance\n")
header_lines.append("\n")
for key in sorted(TOC.keys()):
    header_lines.append("[{SectionTitle}]({key})<br/>".format(SectionTitle=TOC[key], key=key))

newlines = header_lines + ["\n\n"] + lines

summary_markdown = open("summary/index.md", "w")
for line in newlines:
    summary_markdown.write(line)

summary_markdown.close()



