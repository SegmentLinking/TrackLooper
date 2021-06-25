rm -rf cpu_outputs
rm -f .cpu_jobs.txt
mkdir -p cpu_outputs
NJOBS=50
NJOBSm1=$((NJOBS - 1))
for i in $(seq 0 ${NJOBSm1}); do
    echo "sdl -i PU200 -c -l -n 100 -j ${NJOBS} -I ${i} -o cpu_outputs/PU200_cpu_${i}.root > cpu_outputs/PU200_cpu_${i}.log 2>&1" >> .cpu_jobs.txt
done
xargs.sh -n ${NJOBS} .cpu_jobs.txt
hadd -f cpu_outputs/PU200_cpu.root cpu_outputs/PU200_cpu_*.root > cpu_outputs/PU200_cpu.log 2>&1
