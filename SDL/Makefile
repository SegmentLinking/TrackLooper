
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
CXXFLAGS             =  -g --compiler-options -Wall --compiler-options -Wshadow --compiler-options -Woverloaded-virtual --compiler-options -fPIC --compiler-options -fopenmp -dc -lineinfo --ptxas-options=-v --cudart shared -arch=compute_52 -I/mnt/data1/dsr/cub 
LD                   = nvcc 
SOFLAGS              = -g -shared --compiler-options -fPIC --cudart shared -arch=compute_52
PRINTFLAG            = -DAddObjects #-DT4FromT3 -DWarnings
MEMFLAG              =
CACHEFLAG            =
CUDALAUNCHFLAG       = -DNESTED_PARA 
MEMFLAG_FLAGS        = -DExplicit_MD -DExplicit_Seg -DExplicit_Tracklet -DExplicit_Trips -DExplicit_Hit -DExplicit_Track -DExplicit_Module 
CACHEFLAG_FLAGS      = -DCACHE_ALLOC
CUDALAUNCHFLAG_FLAGS = -DNEWGRID_MD -DNEWGRID_Seg -DNEWGRID_Trips -DNEWGRID_Tracklet -DNEWGRID_Pixel -DNEWGRID_Track

#
# how to make it 
#
CUTVALUEFLAG = 
CUTVALUEFLAG_FLAGS = -DCUT_VALUE_DEBUG
%_cuda.o : %.cu %.cuh
	$(LD) -x cu $(CXXFLAGS) $(LDFLAGS) $(ROOTLIBS) $(MEMFLAG) $(PRINTFLAG) $(CACHEFLAG) $(CUDALAUNCHFLAG) $(CUTVALUEFLAG) $< -o $@

%_cpu.o : %.cc %.h
	$(LD) -O2 $(CXXFLAGS) $(LDFLAGS) $(ROOTLIBS) $(MEMFLAG) $(PRINTFLAG) $(CACHEFLAG) $(CUDALAUNCHFLAG) $< -o $@

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
