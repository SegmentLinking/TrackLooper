
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
CXXFLAGS             =  -g --compiler-options -Wall --compiler-options -Wshadow --compiler-options -Woverloaded-virtual --compiler-options -fPIC --compiler-options -fopenmp -dc -lineinfo --ptxas-options=-v --cudart shared -arch=compute_70 -I/mnt/data1/dsr/cub --maxrregcount 128 
LD                   = nvcc 
SOFLAGS              = -g -shared --compiler-options -fPIC --cudart shared -arch=compute_70
PRINTFLAG            = -DAddObjects -DT4FromT3 #-DWarnings
FINALSTATE           = -DFINAL_pT5 -DFINAL_pT3 -DFINAL_T5 # -DFINAL_pLS
DUPLICATES           = -DDUP_pLS -DDUP_T5 -DDUP_pT5 -DDUP_pT3 -DCrossclean_T5 -DCrossclean_pT3
MEMFLAG              =
CACHEFLAG            =
CUDALAUNCHFLAG       = -DNESTED_PARA 
MEMFLAG_FLAGS        = -DExplicit_MD -DExplicit_Seg -DExplicit_Tracklet -DExplicit_Trips -DExplicit_Hit -DExplicit_Track -DExplicit_Module -DExplicit_T5 -DExplicit_PT3 -DExplicit_PT5
CACHEFLAG_FLAGS      = -DCACHE_ALLOC
CUDALAUNCHFLAG_FLAGS = -DNEWGRID_MD -DNEWGRID_Seg -DNEWGRID_Trips -DNEWGRID_Tracklet -DNEWGRID_Pixel -DNEWGRID_Track -DNEWGRID_T5 -DNEWGRID_pT3 -DNEWGRID_pT5

#
# how to make it 
#
CUTVALUEFLAG = 
CUTVALUEFLAG_FLAGS = -DCUT_VALUE_DEBUG
%_cuda.o : %.cu %.cuh
	$(LD) -x cu $(CXXFLAGS) $(LDFLAGS) $(ROOTLIBS) $(MEMFLAG) $(PRINTFLAG) $(CACHEFLAG) $(CUDALAUNCHFLAG) $(CUTVALUEFLAG) $(FINALSTATE) $(DUPLICATES) $< -o $@

%_cpu.o : %.cc %.h
	$(LD) -O2 $(CXXFLAGS) $(LDFLAGS) $(ROOTLIBS) $(MEMFLAG) $(PRINTFLAG) $(CACHEFLAG) $(CUDALAUNCHFLAG) $(FINALSTATE) $(DUPLICATES) $< -o $@

$(LIB):$(CCOBJECTS) $(CUOBJECTS)
	$(LD)  $(SOFLAGS) $^ -o $@

unified_cutvalue: CUTVALUEFLAG = $(CUTVALUEFLAG_FLAGS)
unified_cutvalue: $(LIB) 
unified: $(LIB)

unified_newgrid: CUDALAUNCHFLAG = $(CUDALAUNCHFLAG_FLAGS)
unified_newgrid: $(LIB)


unified_cache: CACHEFLAG += $(CACHEFLAG_FLAGS)
unified_cache: $(LIB)

unified_cache_newgrid: CACHEFLAG += $(CACHEFLAG_FLAGS)
unified_cache_newgrid: CUDALAUNCHFLAG = $(CUDALAUNCHFLAG_FLAGS)
unified_cache_newgrid: $(LIB)

explicit: MEMFLAG += $(MEMFLAG_FLAGS)
explicit: $(LIB)

explicit_newgrid: MEMFLAG += $(MEMFLAG_FLAGS)
explicit_newgrid: CUDALAUNCHFLAG = $(CUDALAUNCHFLAG_FLAGS)
explicit_newgrid: $(LIB)

explicit_cache: MEMFLAG += $(MEMFLAG_FLAGS)
explicit_cache: CACHEFLAG += $(CACHEFLAG_FLAGS)
explicit_cache: $(LIB)

explicit_cache_newgrid: MEMFLAG += $(MEMFLAG_FLAGS)
explicit_cache_newgrid: CACHEFLAG += $(CACHEFLAG_FLAGS)
explicit_cache_newgrid: CUDALAUNCHFLAG = $(CUDALAUNCHFLAG_FLAGS)
explicit_cache_newgrid: $(LIB)

all: unified

clean:
	rm -f *.opp \
	rm -f *.o \
	rm -f *.d \
	rm -f *.so \
