
if [ -z $1 ];
then
    echo "Must provide that path to the SDL ntuple"
    exit
fi

rm .jobs.txt
NJOBS=16

rm -rf outputs/
mkdir -p outputs/

for i in $(seq 0 $((NJOBS-1))); do
    echo "./doAnalysis -i ${1} -p ${2} -t tree -o outputs/output_${i}.root -j ${NJOBS} -I ${i} > outputs/output_${i}.log 2>&1" >> .jobs.txt
done

xargs.sh .jobs.txt

hadd -f efficiency.root outputs/*.root 
