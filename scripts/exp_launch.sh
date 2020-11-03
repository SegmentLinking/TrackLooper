echo "nvprof ./bin/doAnalysis -i /nfs-7/userdata/bsathian/SDL_trackingNtuple/ttbar_highPU/trackingNtuple_with_PUinfo_500_evts.root -t trackingNtuple/tree -n 1 -d -v 1 1>nvprof_output.txt 2>&1"

nvprof ./bin/doAnalysis -i /nfs-7/userdata/bsathian/SDL_trackingNtuple/ttbar_highPU/trackingNtuple_with_PUinfo_500_evts.root -t trackingNtuple/tree -n 1 -d -v 1 1>nvprof_output.txt 2>&1

echo "nvprof --print-gpu-trace ./bin/doAnalysis -i /nfs-7/userdata/bsathian/SDL_trackingNtuple/ttbar_highPU/trackingNtuple_with_PUinfo_500_evts.root -t trackingNtuple/tree -n 1 -d -v 1 1>gpu_output.txt 2>&1"

nvprof --print-gpu-trace ./bin/doAnalysis -i /nfs-7/userdata/bsathian/SDL_trackingNtuple/ttbar_highPU/trackingNtuple_with_PUinfo_500_evts.root -t trackingNtuple/tree -n 1 -d -v 1 1>gpu_output.txt 2>&1

echo "nvprof --print-api-trace ./bin/doAnalysis -i /nfs-7/userdata/bsathian/SDL_trackingNtuple/ttbar_highPU/trackingNtuple_with_PUinfo_500_evts.root -t trackingNtuple/tree -n 1 -d -v 1 1>api_output.txt 2>&1"

nvprof --print-api-trace ./bin/doAnalysis -i /nfs-7/userdata/bsathian/SDL_trackingNtuple/ttbar_highPU/trackingNtuple_with_PUinfo_500_evts.root -t trackingNtuple/tree -n 1 -d -v 1 1>api_output.txt 2>&1

