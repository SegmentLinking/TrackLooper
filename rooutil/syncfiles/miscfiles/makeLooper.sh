#!/bin/bash

macro="temp_macro.C"
input=""
tree="Events"
histlist="hists.txt"
all="all.sh"
merged="merged.pdf"
branches="evt_event"

if [ $# -lt 1 ]; then
    echo "makeLooper -i inputFile.root -t [treeName] -b \"branch1 branch2 branch3 ...\""
    exit
    # return 1
fi

OPTIND=1
while getopts "i:t:b:" opt; do
  case $opt in
    i) input=$OPTARG;;
    t) tree=$OPTARG;;
    b) branches=$OPTARG;;
  esac
done
shift $((OPTIND-1))

if [ -z $input ]; then
    echo "You forgot the input file. Usage is:"
    echo "   makeLooper -i inputFile.root -t [treeName] -b \"branch1 branch2 branch3 ...\""
    exit
    # return 1
fi

### GET FILES
echo ">>> Make sure you have NOT done cmsenv yet!"
echo ""
echo ">>> Getting makeCMS3ClassFiles and dataMCplotMaker"
curl -O https://raw.githubusercontent.com/cmstas/Software/master/makeCMS3ClassFiles/makeCMS3ClassFiles.C > /dev/null 2>&1
curl -O https://raw.githubusercontent.com/cmstas/Software/master/dataMCplotMaker/dataMCplotMaker.cc > /dev/null 2>&1
curl -O https://raw.githubusercontent.com/cmstas/Software/master/dataMCplotMaker/dataMCplotMaker.h > /dev/null 2>&1
curl -O https://raw.githubusercontent.com/cmstas/Software/master/dataMCplotMaker/PlotMakingTools.h > /dev/null 2>&1

### HIST BOOKING
echo ">>> Making class files and figuring out histogram ranges"
echo "{" > $macro
echo "    gErrorIgnoreLevel=kError;" >> $macro
echo "    TChain *ch = new TChain(\"$tree\");" >> $macro
echo "    ch->Add(\"$input\");" >> $macro

for branch in $branches; do
    nicename=`echo $branch | sed 's/evt_//'`
    echo ">>>    Branch: $branch (aka $nicename)"

    echo "    ch->Draw(\"$branch>>h1D_$branch\");" >> $macro
    echo "    std::cout << \"$branch $nicename \" " >> $macro
    echo "              << \"TH1F *h1D_${nicename}_file = new TH1F(\\\"h1D_$nicename\\\"+filename,\\\"\\\", \" " >> $macro
    echo "              << h1D_$branch->GetXaxis()->GetNbins() << \",\" " >> $macro
    echo "              << h1D_$branch->GetXaxis()->GetXmin()  << \",\" " >> $macro
    echo "              << h1D_$branch->GetXaxis()->GetXmax() << \");\" << std::endl;" >> $macro

done

echo "    gROOT->ProcessLine(\".L makeCMS3ClassFiles.C++\");" >> $macro
echo "    gROOT->ProcessLine(\"makeCMS3ClassFiles(\\\"$input\\\",\\\"$tree\\\")\");" >> $macro
echo "}" >> $macro

root -b -q $macro | grep "TH1F" > $histlist

echo ">>> Updating doAll.C and ScanChain.C"

echo "" > temp_vec.txt
echo "" > temp_fill.txt
echo "" > temp_file.txt
echo "" > temp_draw.txt

echo "std::vector<std::string> titles;" >> temp_vec.txt
echo "titles.push_back(\"first\");" >> temp_vec.txt
echo "" >> temp_vec.txt

echo "TH1F* null = new TH1F(\"\",\"\",1,0,1);" >> temp_draw.txt
echo "std::string com = \"--noDivisionLabel --showPercentage --outputName pdfs/\"; " >> temp_draw.txt
echo "std::string spec = \"\"; " >> temp_draw.txt

while read line; do
    branch=$(echo $line | cut -d ' ' -f1)
    nicename=$(echo $line | cut -d ' ' -f2)
    hist=$(echo $line | cut -d '*' -f2 | cut -d ' ' -f1)
    vecname=h1D_${nicename}_vec
    
    # make vectors and histos
    echo "std::vector<TH1F*> $vecname;" >> temp_vec.txt

    echo "$(echo $line | cut -d ' ' -f3-)" >> temp_file.txt
    echo "$vecname.push_back($hist);" >> temp_file.txt

    # fill histo
    echo "$hist->Fill($branch());" >> temp_fill.txt

    # draw them
    echo "dataMCplotMaker(null,$vecname,titles,\"$nicename\",spec,com+\"$hist.pdf --isLinear --xAxisOverride [GeV] \");" >> temp_draw.txt
done < hists.txt


# load dataMCplotMaker before scanchain in doAll.C
awk '/ScanChain.C/ { print "  gROOT->ProcessLine(\".L dataMCplotMaker.cc+\");"; print; next }1' doAll.C > temp_doAll.C && mv temp_doAll.C doAll.C

# include dataMCplotMaker after CMS3 in ScanChain.C
awk '/CMS3.cc/ { print; print "#include \"dataMCplotMaker.h\""; next }1' ScanChain.C > temp_ScanChain.C && mv temp_ScanChain.C ScanChain.C
# setup filename
awk '/SetCacheSize/ { print; print "    TString filename(currentFile->GetTitle());"; next }1' ScanChain.C > temp_ScanChain.C && mv temp_ScanChain.C ScanChain.C
# put filling statements after Analysis Code marker
sed -e '/Analysis Code/r temp_fill.txt' ScanChain.C > temp_ScanChain.C && mv temp_ScanChain.C ScanChain.C
# put drawing statements after samplehist is drawn
sed -e '/->Draw/r temp_draw.txt' ScanChain.C > temp_ScanChain.C && mv temp_ScanChain.C ScanChain.C
# put vec declarations after "currentFile = 0"
sed -e '/currentFile = 0/r temp_vec.txt' ScanChain.C > temp_ScanChain.C && mv temp_ScanChain.C ScanChain.C
# put histogram creation after filename
sed -e '/TString filename/r temp_file.txt' ScanChain.C > temp_ScanChain.C && mv temp_ScanChain.C ScanChain.C
# delete junk stuff from ScanChain.C
sed -e '/samplehisto/d' -e '/bmark/d' -e '/rootdir/d' ScanChain.C > temp_ScanChain.C && mv temp_ScanChain.C ScanChain.C
# delete stuff which causes segfaults when pushing pointers to TH1Fs into a vector (why?)
sed '/file->Close()/d' ScanChain.C > temp_ScanChain.C && mv temp_ScanChain.C ScanChain.C
sed '/delete file/d' ScanChain.C > temp_ScanChain.C && mv temp_ScanChain.C ScanChain.C



# cleanup
echo ">>> Cleaning up"
rm $macro $histlist temp_*.txt makeCMS3ClassFiles.C *.d *.so

mkdir -p "pdfs"

echo ">>> Writing all.sh"

echo "rm -f pdfs/*.pdf" > $all
echo "root -b -q doAll.C" >> $all
echo "gs -dBATCH -dNOPAUSE -q -sDEVICE=pdfwrite -sOutputFile=$merged pdfs/*.pdf" >> $all
echo "cp $merged ~/public_html/dump/" >> $all
echo "echo \"uaf-6.t2.ucsd.edu/~namin/dump/$merged\"" >> $all

echo ">>> Getting a CMSSW environment"
source /code/osgcode/cmssoft/cms/cmsset_default.sh
scramv1 project CMSSW CMSSW_7_4_6_patch6 # cmsrel
cd CMSSW_7_4_6_patch6
eval `scram runtime -sh` # cmsenv
cd ..

echo ">>> Do \". all.sh\" now"
