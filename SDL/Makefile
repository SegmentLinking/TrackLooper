
#
# stuff to make
#

CCSOURCES=$(wildcard *.cc)
CCOBJECTS=$(CCSOURCES:.cc=_cpu.o)
CCHEADERS=$(CCSOURCES:.cc=.h)

CUSOURCES=$(wildcard *.cu)
CUOBJECTS=$(CUSOURCES:.cu=_cuda.o)
CUHEADERS=$(CUSOURCES:.cu=.cuh)
LIB=libsdl.so

#
# flags to keep track of
#

# AMD Opteron and Intel EM64T (64 bit mode) Linux with gcc 3.x
CXX                  = nvcc
CXXFLAGS             =  -g --compiler-options -Wall --compiler-options -Wshadow --compiler-options -Woverloaded-virtual --compiler-options -fPIC --compiler-options -fopenmp -dc -lineinfo --ptxas-options=-v --cudart shared -arch=compute_70 -I/mnt/data1/dsr/cub --use_fast_math --default-stream per-thread -I..
ROOTCFLAGS           = --compiler-options -pthread --compiler-options -std=c++17 -m64 -I/cvmfs/cms.cern.ch/slc7_amd64_gcc900/cms/cmssw/CMSSW_11_2_0_pre5/external/slc7_amd64_gcc900/bin/../../../../../../../slc7_amd64_gcc900/lcg/root/6.20.06-ghbfee3/include
LD                   = nvcc 
SOFLAGS              = -g -shared --compiler-options -fPIC --cudart shared -arch=compute_70 -code=sm_72
PRINTFLAG            = -DT4FromT3 #-DWarnings
DUPLICATES           = -DDUP_pLS -DDUP_T5 -DDUP_pT5 -DDUP_pT3 -DCrossclean_T5 -DCrossclean_pT3 -DFP16_Base -DFP16_dPhi
MEMFLAG              =
CACHEFLAG            =
CUDALAUNCHFLAG       =  
MEMFLAG_FLAGS        =
CACHEFLAG_FLAGS      = -DCACHE_ALLOC
CUDALAUNCHFLAG_FLAGS = 
PT0P8                =
PRELOAD              =
T3T3EXTENSION        =
T5CUTFLAGS           = $(T5DNNFLAG) $(T5RZCHI2FLAG) $(T5RPHICHI2FLAG)
#
# how to make it 
#
CUTVALUEFLAG =
CUTVALUEFLAG_FLAGS = -DCUT_VALUE_DEBUG
%_cuda.o : %.cu %.cuh
	$(LD) -x cu $(PT0P8) $(PRELOAD) $(T3T3EXTENSION) $(CXXFLAGS) $(LDFLAGS) $(ROOTLIBS) $(MEMFLAG) $(PRINTFLAG) $(CACHEFLAG) $(CUDALAUNCHFLAG) $(T5CUTFLAGS) $(CUTVALUEFLAG) $(DUPLICATES) $< -o $@

%_cpu.o : %.cc %.h
	$(LD) -O2   $(PT0P8) $(PRELOAD) $(T3T3EXTENSION) $(CXXFLAGS) $(LDFLAGS) $(ROOTLIBS) $(MEMFLAG) $(PRINTFLAG) $(CACHEFLAG) $(CUDALAUNCHFLAG) $(DUPLICATES) $(ROOTCFLAGS) $< -o $@

$(LIB):$(CCOBJECTS) $(CUOBJECTS)
#$(LIB):$(CUOBJECTS)
	$(LD)  $(SOFLAGS) $^ -o $@

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
