
#
# stuff to make
#

CCSOURCES=$(filter-out LST.cc, $(wildcard *.cc))
CCOBJECTS_CPU=$(CCSOURCES:.cc=_cpu.o)
CCOBJECTS_CUDA=$(CCSOURCES:.cc=_cuda.o)

LSTSOURCES=LST.cc
LSTOBJECTS_CPU=$(LSTSOURCES:.cc=_cpu.o)
LSTOBJECTS_CUDA=$(LSTSOURCES:.cc=_cuda.o)

LIB_CUDA=libsdl_cuda.so
LIB_CPU=libsdl_cpu.so

ifeq ($(BACKEND), cpu)
  LIB_CUDA=
else ifeq ($(BACKEND), cuda)
  LIB_CPU=
endif

LIBS=$(LIB_CUDA) $(LIB_CPU)

#
# flags to keep track of
#

CXX                  = g++
CXXFLAGS_CPU         = -march=native -mtune=native -Ofast -fno-reciprocal-math -fopenmp-simd -g -Wall -Wshadow -Woverloaded-virtual -fPIC -fopenmp -I..
CXXFLAGS_CUDA        = -O3 -g --compiler-options -Wall --compiler-options -Wshadow --compiler-options -Woverloaded-virtual --compiler-options -fPIC --compiler-options -fopenmp -dc -lineinfo --ptxas-options=-v --cudart shared -arch=compute_70 --use_fast_math --default-stream per-thread -I..
ALPAKAINCLUDE        = -I${ALPAKA_ROOT}/include -I/${BOOST_ROOT}/include -std=c++17
ALPAKASERIAL         = -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
ALPAKACUDA           = -DALPAKA_ACC_GPU_CUDA_ENABLED -DALPAKA_ACC_GPU_CUDA_ONLY --expt-relaxed-constexpr
ROOTCFLAGS           = -pthread -m64 -I/cvmfs/cms.cern.ch/slc7_amd64_gcc900/cms/cmssw/CMSSW_11_2_0_pre5/external/slc7_amd64_gcc900/bin/../../../../../../../slc7_amd64_gcc900/lcg/root/6.20.06-ghbfee3/include
PRINTFLAG            = -DT4FromT3 #-DWarnings
DUPLICATES           = -DDUP_pLS -DDUP_T5 -DDUP_pT5 -DDUP_pT3 -DCrossclean_T5 -DCrossclean_pT3 #-DFP16_Base
CACHEFLAG            =
CACHEFLAG_FLAGS      = -DCACHE_ALLOC
T5CUTFLAGS           = $(T5DNNFLAG) $(T5RZCHI2FLAG) $(T5RPHICHI2FLAG)

LD_CPU               = g++
SOFLAGS_CPU          = -g -shared -fPIC
ALPAKABACKEND_CPU    = $(ALPAKASERIAL)
COMPILE_CMD_CPU      = $(LD_CPU) -c

LD_CUDA              = nvcc
SOFLAGS_CUDA         = -g -shared --compiler-options -fPIC --cudart shared -arch=compute_70 -code=sm_72
ALPAKABACKEND_CUDA   = $(ALPAKACUDA)
COMPILE_CMD_CUDA     = $(LD_CUDA) -x cu

CUTVALUEFLAG =
CUTVALUEFLAG_FLAGS = -DCUT_VALUE_DEBUG

LST_cpu.o: LST.cc
	 $(CXX) -c $(CXXFLAGS_CPU) $(ROOTLIBS) $(PRINTFLAG) $(CACHEFLAG) $(DUPLICATES) $(ROOTCFLAGS) $(ALPAKAINCLUDE) $(ALPAKASERIAL) $< -o $@

LST_cuda.o: LST.cc
	 $(CXX) -c $(CXXFLAGS_CPU) $(ROOTLIBS) $(PRINTFLAG) $(CACHEFLAG) $(DUPLICATES) $(ROOTCFLAGS) $(ALPAKAINCLUDE) $(ALPAKASERIAL) $< -o $@

%_cpu.o: %.cc
	$(COMPILE_CMD_CPU) $(CXXFLAGS_CPU) $(PRINTFLAG) $(CACHEFLAG) $(CUTVALUEFLAG) $(T5CUTFLAGS) $(NOPLSDUPCLEANFLAG) $(PTCUTFLAG) $(DUPLICATES) $(ALPAKAINCLUDE) $(ALPAKABACKEND_CPU) $< -o $@

%_cuda.o: %.cc
	$(COMPILE_CMD_CUDA) $(CXXFLAGS_CUDA) $(PRINTFLAG) $(CACHEFLAG) $(CUTVALUEFLAG) $(T5CUTFLAGS) $(NOPLSDUPCLEANFLAG) $(PTCUTFLAG) $(DUPLICATES) $(ALPAKAINCLUDE) $(ALPAKABACKEND_CUDA) $< -o $@

$(LIB_CUDA): $(CCOBJECTS_CUDA) $(LSTOBJECTS_CUDA)
	$(LD_CUDA) $(SOFLAGS_CUDA) $^ -o $@
	mkdir -p cuda
	ln -sf ../$@ cuda/$(@:_cuda.so=.so)

$(LIB_CPU): $(CCOBJECTS_CPU) $(LSTOBJECTS_CPU)
	$(LD_CPU) $(SOFLAGS_CPU) $^ -o $@
	mkdir -p cpu
	ln -sf ../$@ cpu/$(@:_cpu.so=.so)

explicit: $(LIBS)

explicit_cache: CACHEFLAG += $(CACHEFLAG_FLAGS)
explicit_cache: $(LIBS)

explicit_cache_cutvalue: CUTVALUEFLAG = $(CUTVALUEFLAG_FLAGS)
explicit_cache_cutvalue: CACHEFLAG += $(CACHEFLAG_FLAGS)
explicit_cache_cutvalue: $(LIBS)

clean:
	rm -f *.opp
	rm -f *.o
	rm -f *.d
	rm -f *.so
	rm -rf cpu/
	rm -rf cuda/
