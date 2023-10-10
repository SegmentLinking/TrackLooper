
#
# stuff to make
#

CCSOURCES=$(filter-out LST.cc, $(wildcard *.cc))
CCOBJECTS=$(CCSOURCES:.cc=_$(BACKEND).o)

LSTSOURCES=LST.cc
LSTOBJECTS=$(LSTSOURCES:.cc=_cpu.o)

LIB=libsdl.so

#
# flags to keep track of
#

CXX                  = g++
CXXFLAGS             = -g -Wall -Wshadow -Woverloaded-virtual -fPIC -fopenmp -I..
CXXFLAGS_CUDA        = -g --compiler-options -Wall --compiler-options -Wshadow --compiler-options -Woverloaded-virtual --compiler-options -fPIC --compiler-options -fopenmp -dc -lineinfo --ptxas-options=-v --cudart shared -arch=compute_70 --use_fast_math --default-stream per-thread -I..
ALPAKAINCLUDE        = -I${ALPAKA_ROOT}/include -I/${BOOST_ROOT}/include -std=c++17
ALPAKASERIAL         = -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
ALPAKACUDA           = -DALPAKA_ACC_GPU_CUDA_ENABLED -DALPAKA_ACC_GPU_CUDA_ONLY --expt-relaxed-constexpr
ROOTCFLAGS           = -pthread -m64 -I/cvmfs/cms.cern.ch/el8_amd64_gcc11/cms/cmssw/CMSSW_13_0_0_pre4/external/el8_amd64_gcc11/bin/../../../../../../../el8_amd64_gcc11/lcg/root/6.26.07-3c8fea8a0ce2aca35570ac22afed05d0/include
PRINTFLAG            = -DT4FromT3 #-DWarnings
DUPLICATES           = -DDUP_pLS -DDUP_T5 -DDUP_pT5 -DDUP_pT3 -DCrossclean_T5 -DCrossclean_pT3 #-DFP16_Base
MEMFLAG              =
CACHEFLAG            =
MEMFLAG_FLAGS        =
CACHEFLAG_FLAGS      = -DCACHE_ALLOC
T5CUTFLAGS           = $(T5DNNFLAG) $(T5RZCHI2FLAG) $(T5RPHICHI2FLAG)

ifeq ($(BACKEND), cpu)
  LD                 = g++
  SOFLAGS            = -g -shared -fPIC
  ALPAKABACKEND      = $(ALPAKASERIAL)
  COMPILE_CMD        = $(LD) -c -O2
  ACTUAL_CXXFLAGS    = $(CXXFLAGS)
else
  LD                 = nvcc
  SOFLAGS            = -g -shared --compiler-options -fPIC --cudart shared -arch=compute_70 -code=sm_72
  ALPAKABACKEND      = $(ALPAKACUDA)
  COMPILE_CMD        = $(LD) -x cu
  ACTUAL_CXXFLAGS    = $(CXXFLAGS_CUDA)
endif

CUTVALUEFLAG =
CUTVALUEFLAG_FLAGS = -DCUT_VALUE_DEBUG

%_$(BACKEND).o : %.cc
	$(COMPILE_CMD) $(ACTUAL_CXXFLAGS) $(LDFLAGS) $(ROOTLIBS) $(MEMFLAG) $(PRINTFLAG) $(CACHEFLAG) $(CUTVALUEFLAG) $(T5CUTFLAGS) $(NOPLSDUPCLEANFLAG) $(DUPLICATES) $(ALPAKAINCLUDE) $(ALPAKABACKEND) $< -o $@

LST_cpu.o : LST.cc
	$(CXX) -c -O2 $(CXXFLAGS) $(LDFLAGS) $(ROOTLIBS) $(MEMFLAG) $(PRINTFLAG) $(CACHEFLAG) $(DUPLICATES) $(ROOTCFLAGS) $(ALPAKAINCLUDE) $(ALPAKASERIAL) $< -o $@

$(LIB):$(CCOBJECTS) $(LSTOBJECTS)
	$(LD) $(SOFLAGS) $^ -o $@

explicit: MEMFLAG += $(MEMFLAG_FLAGS)
explicit: $(LIB)

explicit_cache: MEMFLAG += $(MEMFLAG_FLAGS)
explicit_cache: CACHEFLAG += $(CACHEFLAG_FLAGS)
explicit_cache: $(LIB)

explicit_cache_cutvalue: CUTVALUEFLAG = $(CUTVALUEFLAG_FLAGS)
explicit_cache_cutvalue: MEMFLAG += $(MEMFLAG_FLAGS)
explicit_cache_cutvalue: CACHEFLAG += $(CACHEFLAG_FLAGS)
explicit_cache_cutvalue: $(LIB)

clean:
	rm -f *.opp \
	rm -f *.o \
	rm -f *.d \
	rm -f *.so \
