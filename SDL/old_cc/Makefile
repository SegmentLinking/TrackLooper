
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
CXXFLAGS      =  -g -G -O2 --compiler-options -Wall --compiler-options -Wshadow --compiler-options -Woverloaded-virtual --compiler-options -fPIC --compiler-options -fopenmp -dc 
#LD            = g++4 
LD            = nvcc 
#LDFLAGS       = -g -O2
SOFLAGS       = -g -G -shared --compiler-options -fPIC
# how to make it 
#

%.o : %.cu %.cuh
	$(LD) -x cu $(CXXFLAGS) $(LDFLAGS) $(ROOTLIBS) $< -o $@

%.o : %.cc %.h
	$(LD) $(CXXFLAGS) $(LDFLAGS) $(ROOTLIBS) $< -o $@



$(LIB):$(CCOBJECTS) $(CUOBJECTS)
	$(LD)  $(SOFLAGS) $^ -o $@

all: $(LIB)
clean:
	rm -f *.o \
	rm -f *.d \
	rm -f *.so \
