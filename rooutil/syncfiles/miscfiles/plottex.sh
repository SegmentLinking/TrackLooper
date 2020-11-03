#!/bin/bash

# see http://root.cern.ch/drupal/content/saving-canvas-tex
# need to do c1->Print("hist.tex"). works in root 5.34+
if [ $# -lt 1 ]; then
    echo "usage: plottex <hist.tex>"
    return 1
fi

# hist.tex to temp_hist.tex
echo "$1 ==> temp_$1"

touch temp_$1
echo "\\documentclass{article}" >> temp_$1
echo "\\usepackage{tikz}" >> temp_$1
echo "\\usetikzlibrary{patterns}" >> temp_$1
echo "\\usetikzlibrary{plotmarks}" >> temp_$1
echo "\\begin{document}" >> temp_$1
echo "\\pagenumbering{gobble}" >> temp_$1
echo "\\par" >> temp_$1
echo "\\begin{figure}[htbp]" >> temp_$1
echo "\\scalebox{0.7}{\input{$1}}" >> temp_$1
echo "\\end{figure}" >> temp_$1
echo "\\end{document}" >> temp_$1

# temp_hist.tex to hist.pdf
echo "temp_$1 ==> temp_${1%%.tex}.pdf"
pdflatex temp_$1 >& /dev/null
mv temp_${1%%.tex}.pdf ${1%%.tex}.pdf
rm temp_${1%%.tex}.aux
rm temp_${1%%.tex}.tex
rm temp_${1%%.tex}.log
