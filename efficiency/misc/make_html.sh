#!/bin/bash

cd summary/
for File in $(ls *.md); do
    /home/users/phchang/local/bin/pandoc -f markdown -t html -o ${File/.md/.html} ${File}
done
cd ../

cd compare/
for File in $(ls *.md); do
    /home/users/phchang/local/bin/pandoc -f markdown -t html -o ${File/.md/.html} ${File}
done
cd ../
