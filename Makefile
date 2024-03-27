
# Simple makefile

EXES := bin/sdl_cpu bin/sdl_cuda

SOURCES=$(wildcard code/core/*.cc)
OBJECTS=$(SOURCES:.cc=.o)
OBJECTS_CPU=$(SOURCES:.cc=_cpu.o)
OBJECTS_CUDA=$(SOURCES:.cc=_cuda.o)
HEADERS=$(SOURCES:.cc=.h)

CXX         = g++
CXXFLAGS    = -g -O2 -Wall -fPIC -Wshadow -Woverloaded-virtual -lineinfo  -fopenmp -lgomp --default-stream per-thread
LDFLAGS     = -g -O2 -Wall -fPIC -Wshadow -Woverloaded-virtual
SOFLAGS     = -g -shared
CXXFLAGS    = -g -O2 -Wall -fPIC -Wshadow -Woverloaded-virtual
LDFLAGS     = -g -O2 $(SDLLIB) -L${TRACKLOOPERDIR}/SDL
ROOTLIBS    = $(shell root-config --libs)
ROOTCFLAGS  = $(foreach option, $(shell root-config --cflags), $(option))
ALPAKAINCLUDE = -I${ALPAKA_ROOT}/include -I/${BOOST_ROOT}/include -std=c++17 -DALPAKA_DEBUG=0
ALPAKA_CPU = -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
ALPAKA_CUDA = -DALPAKA_ACC_GPU_CUDA_ENABLED -DALPAKA_HOST_ONLY
CFLAGS      = $(ROOTCFLAGS)  -Wall  -Wno-unused-function  -g  -O2  -fPIC  -fno-var-tracking -ISDL -I$(shell pwd) -Icode  -Icode/core -I${CUDA_HOME}/include  -fopenmp
EXTRACFLAGS = $(shell rooutil-config) -g
EXTRAFLAGS  = -fPIC -ITMultiDrawTreePlayer -Wunused-variable -lTMVA -lEG -lGenVector -lXMLIO -lMLP -lTreePlayer -L${CUDA_HOME}/lib64 -lcudart -fopenmp
DOQUINTUPLET = #-DFP16_Base
PTCUTFLAG    =
CUTVALUEFLAG = 
CUTVALUEFLAG_FLAGS = -DCUT_VALUE_DEBUG

PRIMITIVEFLAG = 
PRIMITIVEFLAG_FLAGS = -DPRIMITIVE_STUDY

all: rooutil efficiency $(EXES)


cutvalue: CUTVALUEFLAG = ${CUTVALUEFLAG_FLAGS}
cutvalue: rooutil efficiency $(EXES)

primitive: PRIMITIVEFLAG = ${PRIMITIVEFLAG_FLAGS}
primitive: rooutil efficiency $(EXES)

cutvalue_primitive: CUTVALUEFLAG = ${CUTVALUEFLAG_FLAGS}
cutvalue_primitive: PRIMITIVEFLAG = ${PRIMITIVEFLAG_FLAGS}
cutvalue_primitive: rooutil efficiency $(EXES)

bin/doAnalysis: bin/doAnalysis.o $(OBJECTS)
	$(CXX) $(PTCUTFLAG) $(LDFLAGS) $^ $(ROOTLIBS) $(EXTRACFLAGS) $(CUTVALUEFLAG) $(PRIMITIVEFLAG) $(EXTRAFLAGS) $(DOQUINTUPLET) $(ALPAKAINCLUDE) $(ALPAKA_CPU) -o $@

bin/sdl_cpu: SDLLIB=-lsdl_cpu
bin/sdl_cpu: bin/sdl_cpu.o $(OBJECTS_CPU)
	$(CXX) $(PTCUTFLAG) $(LDFLAGS) $^ $(ROOTLIBS) $(EXTRACFLAGS) $(CUTVALUEFLAG) $(PRIMITIVEFLAG) $(EXTRAFLAGS) $(DOQUINTUPLET) $(ALPAKAINCLUDE) $(ALPAKA_CPU) -o $@
bin/sdl_cuda: SDLLIB=-lsdl_cuda
bin/sdl_cuda: bin/sdl_cuda.o $(OBJECTS_CUDA)
	$(CXX) $(PTCUTFLAG) $(LDFLAGS) $^ $(ROOTLIBS) $(EXTRACFLAGS) $(CUTVALUEFLAG) $(PRIMITIVEFLAG) $(EXTRAFLAGS) $(DOQUINTUPLET) $(ALPAKAINCLUDE) $(ALPAKA_CUDA) -o $@

%_cpu.o: %.cc rooutil
	$(CXX) $(PTCUTFLAG) $(CFLAGS) $(EXTRACFLAGS) $(CUTVALUEFLAG) $(PRIMITIVEFLAG) $(DOQUINTUPLET) $(ALPAKAINCLUDE) $(ALPAKA_CPU) $< -c -o $@
%_cuda.o: %.cc rooutil
	$(CXX) $(PTCUTFLAG) $(CFLAGS) $(EXTRACFLAGS) $(CUTVALUEFLAG) $(PRIMITIVEFLAG) $(DOQUINTUPLET) $(ALPAKAINCLUDE) $(ALPAKA_CUDA) $< -c -o $@

rooutil:
	$(MAKE) -C code/rooutil/

efficiency: rooutil
	$(MAKE) -C efficiency/

clean:
	rm -f $(OBJECTS) bin/*.o $(EXES) bin/sdl
	rm -f code/rooutil/*.so code/rooutil/*.o
	rm -f bin/sdl.o
	rm -f SDL/*.o
	cd efficiency/ && make clean

.PHONY: rooutil efficiency
