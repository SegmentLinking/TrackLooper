
#
# stuff to make
#

CCSOURCES=$(wildcard *.cc)
CCOBJECTS=$(CCSOURCES:.cc=_cpu.o)
CCHEADERS=$(CCSOURCES:.cc=.h)

CUSOURCES=$(wildcard *.cu)
CUOBJECTS=$(CUSOURCES:.cu=_cuda.o)
CUHEADERS=$(CUSOURCES:.cu=.cuh)
LIB=sdl.so

#
# flags to keep track of
#

# AMD Opteron and Intel EM64T (64 bit mode) Linux with gcc 3.x
CXX                  = nvcc
CXXFLAGS             =  -g --compiler-options -Wall --compiler-options -Wshadow --compiler-options -Woverloaded-virtual --compiler-options -fPIC --compiler-options -fopenmp -dc -lineinfo --ptxas-options=-v --cudart shared -arch=compute_70 -I/mnt/data1/dsr/cub --use_fast_math --default-stream per-thread
LD                   = nvcc 
SOFLAGS              = -g -shared --compiler-options -fPIC --cudart shared -arch=compute_70 -code=sm_72
PRINTFLAG            = -DAddObjects -DT4FromT3 #-DWarnings
FINALSTATE           = -DFINAL_pT5 -DFINAL_pT3 -DFINAL_T5 -DFINAL_pLS
DUPLICATES           = -DDUP_pLS -DDUP_T5 -DDUP_pT5 -DDUP_pT3 -DCrossclean_T5 -DCrossclean_pT3 -DFP16_Base -DFP16_dPhi #-DFP16_circle -DFP16_seg -DFP16_T5
MEMFLAG              =
CACHEFLAG            =
CUDALAUNCHFLAG       = #-DNESTED_PARA 
MEMFLAG_FLAGS        =
CACHEFLAG_FLAGS      = -DCACHE_ALLOC
CUDALAUNCHFLAG_FLAGS = #-DNEWGRID_MD -DNEWGRID_Seg -DNEWGRID_Trips -DNEWGRID_Tracklet -DNEWGRID_Pixel -DNEWGRID_Track -DNEWGRID_T5 -DNEWGRID_pT3 -DNEWGRID_pT5
PT0P8       =
PRELOAD       =
CMSSW12GEOM =
T3T3EXTENSION=
#
# how to make it 
#
CUTVALUEFLAG = 
CUTVALUEFLAG_FLAGS = -DCUT_VALUE_DEBUG
%_cuda.o : %.cu %.cuh
	$(LD) -x cu $(PT0P8) $(PRELOAD) $(CMSSW12GEOM) $(T3T3EXTENSION) $(CXXFLAGS) $(LDFLAGS) $(ROOTLIBS) $(MEMFLAG) $(PRINTFLAG) $(CACHEFLAG) $(CUDALAUNCHFLAG) $(CUTVALUEFLAG) $(FINALSTATE) $(DUPLICATES) $< -o $@

%_cpu.o : %.cc %.h
	$(LD) -O2   $(PT0P8) $(PRELOAD) $(CMSSW12GEOM) $(T3T3EXTENSION) $(CXXFLAGS) $(LDFLAGS) $(ROOTLIBS) $(MEMFLAG) $(PRINTFLAG) $(CACHEFLAG) $(CUDALAUNCHFLAG) $(FINALSTATE) $(DUPLICATES) $< -o $@

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
