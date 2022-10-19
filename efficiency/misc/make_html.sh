#!/bin/bash

cd summary/
rm -f .jobs.txt
for File in $(ls *.md); do
    echo "/home/users/phchang/local/bin/pandoc -f markdown -t html -o ${File/.md/.html} ${File}" >> .jobs.txt
done
xargs.sh .jobs.txt
rm -f .jobs.txt
cd ../

cd compare/
rm -f .jobs.txt
for File in $(ls *.md); do
    echo "/home/users/phchang/local/bin/pandoc -f markdown -t html -o ${File/.md/.html} ${File}" >> .jobs.txt
done
xargs.sh .jobs.txt
rm -f .jobs.txt
cd ../
