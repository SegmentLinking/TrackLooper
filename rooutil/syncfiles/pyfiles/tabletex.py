#!/usr/bin/python

import math, sys, os, commands
from itertools import groupby

beginStr = """
\\documentclass{article}
\\usepackage{multirow}
\\usepackage{slashed}
\\newcommand{\\met}{\\slashed{E}_\\mathrm{T}}
\\newcommand{\\mt}{m_\\mathrm{T}}
\\newcommand{\\pt}{p_\\mathrm{T}}
\\newcommand{\\mtmin}{m_{T}^\\mathrm{min}}
\\newcommand{\\Ht}{H_\\mathrm{T}}
\\renewcommand{\\arraystretch}{1.2}
\\begin{document}
\\pagenumbering{gobble}% remove (eat) page numbers
\\begin{table}[h]
\\centering """

endStr = """
\\end{table}
\\end{document}
"""

def listToRanges(a,addOne=True):
    # turns [1,2,4,5,9] into ['1-2','4-5','9-9'] for use with cline
    ranges = []

    # if addOne, we add one (to account for 0-indexing)
    if(addOne): a = [e+1 for e in a]

    for k, iterable in groupby(enumerate(sorted(a)), lambda x: x[1]-x[0]):
         rng = list(iterable)
         if len(rng) == 1: s = str(rng[0][1])+"-"+str(rng[0][1])
         else: s = "%s-%s" % (rng[0][1], rng[-1][1])
         ranges.append(s)
    return ranges


def makeTableTeX(lines, complete=True):
    # clean lines and get maximum number of columns
    rows = []
    maxcols = -1
    for line in lines:
        line = line.strip()
        rows.append(line)
        maxcols=max(maxcols,len(line.split("|")))
    maxrows = len(rows)

    cells = {} # indexed by row,column. key is [raw content, latex code , meta]

    # make matrix
    for ir in range(maxrows):
        for ic in range(maxcols):
            cells[ir,ic] = ["","",1] # 1 means underline and 0 means no (used for cline)

    # fill matrix
    sectionRows = [] # rows with double hlines
    for irow,row in enumerate(rows):
        if(len(row) < 2):
            sectionRows.append(irow)
            continue

        for icol,col in enumerate(row.split("|")):
            cells[irow,icol][0] = col.strip()

    # loop over matrix
    for irow in range(maxrows):
        if(irow in sectionRows): continue

        for icol in range(maxcols):
            content, latex, underline = cells[irow,icol]
            if(len(latex) > 0):
                continue # we've already handled this cell then

            latex = "& " + content
            cells[irow,icol] = [content, latex, underline]

            if(content.startswith("mrc")):
                nrows = min(int(content.split(" ")[1]), maxrows-irow)
                ncols = min(int(content.split(" ")[2]), maxcols-icol)
                text = " ".join(content.split(" ")[3:])
                for ir in range(nrows):
                    for ic in range(ncols):
                        if(ir == 0 and ic == 0): cells[irow+ir,icol+ic][1] = "& \\multicolumn{%i}{|c|}{\\multirow{%i}{*}{%s}}" % (ncols,nrows,text)
                        elif(ic == 0): cells[irow+ir,icol+ic][1] = "& \\multicolumn{%i}{|c|}{}" % (ncols)
                        else: cells[irow+ir,icol+ic][1] = " "

                        if(ir != nrows-1): cells[irow+ir,icol+ic][2] = 0

    output = ""
    # start printing tex
    if(complete): output += beginStr + "\n"
    output += "  \\begin{tabular}{|"+"c|" * maxcols+"}" + "\n"
    output += "  \\hline" + "\n"

    # print matrix
    for irow in range(maxrows):
        output += "    "
        if(irow in sectionRows):
            output += " \\hline\\hline" + "\n"
            continue

        underlines = []
        for icol in range(maxcols):
            content, latex, underline = cells[irow,icol]
            if(icol == 0): latex = latex.replace("&","")
            if underline: underlines.append(icol)

            output += latex + " "
        output += "\\\\ "
        for r in listToRanges(underlines): output += "\\cline{%s} " % r
        output += "\n"

    output += "  \\end{tabular}\n"
    if(complete): output += endStr

    return output

def makePDF(content,fname):
    basename = ".".join(fname.split(".")[:-1])
    basedir = "/".join(fname.split("/")[:-1])
    fh = open(basename+".tex","w")
    fh.write(content)
    fh.close()

    status,out = commands.getstatusoutput("pdflatex -interaction=nonstopmode -output-directory=%s %s" % (basedir, basename+".tex"))
    # print out
    if(" Error" in out):
        print "[TM] ERROR: Tried to compile, but failed. Last few lines of printout below."
        print "_"*40
        print "\n".join(out.split("\n")[-30:])
    else:
        status,out = commands.getstatusoutput("pdfcrop %s %s" % (basename+".pdf", basename+".pdf"))
        print "[TM] Created %s" % (basename+".pdf")

def getString(fname, complete=True):
    # complete=True returns full blown compileable document
    # complete=False just returns the tabular part for embedding
    fh = open(fname,"r")
    content = makeTableTeX(fh.readlines(), complete)
    fh.close()
    return content

def makeTable(fname):
    content = getString(fname)
    makePDF(content, fname)


if __name__=='__main__':
    if(sys.stdin.isatty()):
        fname = "output.txt"
        if(len(sys.argv) > 1):
            fname = sys.argv[-1]

        fh = open(fname,"r")
        content = makeTableTeX(fh.readlines())
        fh.close()

        makePDF(content, fname)
    else:
        lines = []
        for item in sys.stdin: lines.append(item)
        content = makeTableTeX(lines)

        print content


        if(len(lines) < 1):
            print "Pipe in some stuff, doofus."
            sys.exit(1)



