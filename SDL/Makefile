
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
CXX           = nvcc
CXXFLAGS      =  -g --compiler-options -Wall --compiler-options -Wshadow --compiler-options -Woverloaded-virtual --compiler-options -fPIC --compiler-options -fopenmp -dc -lineinfo --ptxas-options=-v --cudart shared -arch=compute_52 -I/mnt/data1/dsr/cub 
LD            = nvcc 
SOFLAGS       = -g -shared --compiler-options -fPIC --cudart shared -arch=compute_52
PRINTFLAG     = -DAddObjects
MEMFLAG       = 
CACHEFLAG     = 

#
# how to make it 
#

%_cuda.o : %.cu %.cuh
	$(LD) -x cu $(CXXFLAGS) $(LDFLAGS) $(ROOTLIBS) $(MEMFLAG) $(PRINTFLAG) $(CACHEFLAG) $< -o $@

%_cpu.o : %.cc %.h
	$(LD) $(CXXFLAGS) $(LDFLAGS) $(ROOTLIBS) $(MEMFLAG) $(PRINTFLAG) $(CACHEFLAG) $< -o $@

$(LIB):$(CCOBJECTS) $(CUOBJECTS)
	$(LD)  $(SOFLAGS) $^ -o $@

explicit: MEMFLAG += -DExplicit_MD -DExplicit_Seg -DExplicit_Tracklet -DExplicit_Trips -DExplicit_Track -DExplicit_Hit
explicit: $(LIB)

explicit_cache: MEMFLAG += -DExplicit_MD -DExplicit_Seg -DExplicit_Tracklet -DExplicit_Trips -DExplicit_Track -DExplicit_Hit
explicit_cache: CACHEFLAG += -DCACHE_ALLOC
explicit_cache: $(LIB)

unified: $(LIB)

unified_cache: CACHEFLAG += -DCACHE_ALLOC
unified_cache: $(LIB)

all: unified

clean:
	rm -f *.opp \
	rm -f *.o \
	rm -f *.d \
	rm -f *.so \
