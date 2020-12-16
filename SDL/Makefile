
#
# stuff to make
#
CCSOURCES=$(wildcard *.cc)
CCOBJECTS=$(CCSOURCES:.cc=.o)
CCHEADERS=$(CCSOURCES:.cc=.h)

CUSOURCES=$(wildcard *.cu)
CUOBJECTS=$(CUSOURCES:.cu=.o)
CUHEADERS=$(CUSOURCES:.cu=.cuh)
LIB=sdl.so
# AMD Opteron and Intel EM64T (64 bit mode) Linux with gcc 3.x
#CXX           = g++4 
CXX           = nvcc
CXXFLAGS      =  -g --compiler-options -Wall --compiler-options -Wshadow --compiler-options -Woverloaded-virtual --compiler-options -fPIC --compiler-options -fopenmp -dc -lineinfo --ptxas-options=-v --cudart shared -arch=compute_52 -I/mnt/data1/dsr/cub 
#LD            = g++4 
LD            = nvcc 
#LDFLAGS       = -g -O2
SOFLAGS       = -g -shared --compiler-options -fPIC --cudart shared -arch=compute_52
MEMFLAG       = -DExplicit_MD -DExplicit_Seg -DExplicit_Tracklet -DExplicit_Trips 
#MEMFLAG       += -DFull_Explicit
PRINTFLAG     = -DAddObjects
CACHEFLAG     = -DCACHE_ALLOC
# how to make it 
#

%.o : %.cu %.cuh
	$(LD) -x cu $(CXXFLAGS) $(LDFLAGS) $(ROOTLIBS) $(MEMFLAG) $(PRINTFLAG) $(CACHEFLAG) $< -o $@

%.o : %.cc %.h
	$(LD) $(CXXFLAGS) $(LDFLAGS) $(ROOTLIBS) $(MEMFLAG) $(PRINTFLAG) $(CACHEFLAG) $< -o $@



$(LIB):$(CCOBJECTS) $(CUOBJECTS)
	$(LD)  $(SOFLAGS) $^ -o $@

all: $(LIB)
clean:
	rm -f *.o \
	rm -f *.d \
	rm -f *.so \
