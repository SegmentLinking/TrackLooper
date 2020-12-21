
if [ -z $1 ];
then
    echo "Must provide the path to the SDL ntuple"
    exit
fi

if [ -z $2 ];
then
    echo "Must provide the pt bounds setup"
    exit
fi

if [ -z $3 ];
then
    echo "Setting pdgid to 13"
    PDGID=13
else
    PDGID=$3
fi

rm .jobs.txt
NJOBS=16

rm -rf outputs/
mkdir -p outputs/

for i in $(seq 0 $((NJOBS-1))); do
    echo "./doAnalysis -i ${1} -p ${2} -g ${PDGID} -t tree -o outputs/output_${i}.root -j ${NJOBS} -I ${i} > outputs/output_${i}.log 2>&1" >> .jobs.txt
done

xargs.sh .jobs.txt

hadd -f efficiency.root outputs/*.root 
