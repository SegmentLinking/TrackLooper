
# Simple makefile

EXES=bin/sdl

ROOUTIL=code/rooutil/

SOURCES=$(wildcard code/core/*.cc)  #$(wildcard SDL/*.cc)
OBJECTS=$(SOURCES:.cc=.o) $(wildcard ${TRACKLOOPERDIR}/SDL/libsdl.so)
HEADERS=$(SOURCES:.cc=.h)

CC          = g++
CXX         = g++
CXXFLAGS    = -g -O2 -Wall -fPIC -std=c++17 -Wshadow -Woverloaded-virtual -lineinfo  -fopenmp -lgomp --default-stream per-thread
LD          = g++
LDFLAGS     = -g -O2 -Wall -fPIC -Wshadow -Woverloaded-virtual -I/mnt/data1/dsr/cub
SOFLAGS     = -g -shared
CXXFLAGS    = -g -O2 -Wall -fPIC -Wshadow -Woverloaded-virtual
LDFLAGS     = -g -O2
ROOTLIBS    = $(shell root-config --libs)
ROOTCFLAGS  = $(foreach option, $(shell root-config --cflags), $(option))
ALPAKAINCLUDE = -I${ALPAKA_ROOT}/include -I/${BOOST_ROOT}/include -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED -DALPAKA_DEBUG=0
CFLAGS      = $(ROOTCFLAGS)  -Wall  -Wno-unused-function  -g  -O2  -fPIC  -fno-var-tracking -ISDL -I$(shell pwd) -Icode  -Icode/core -I/mnt/data1/dsr/cub -I${CUDA_HOME}/include  -fopenmp
EXTRACFLAGS = $(shell rooutil-config) -g
EXTRAFLAGS  = -fPIC -ITMultiDrawTreePlayer -Wunused-variable -lTMVA -lEG -lGenVector -lXMLIO -lMLP -lTreePlayer -L${CUDA_HOME}/lib64 -lcudart -fopenmp
DOQUINTUPLET = -DFP16_Base -DFP16_dPhi #-DFP16_circle -DFP16_seg -DFP16_T5 #-DDO_QUINTUPLET #-DDO_QUADRUPLET
PT0P8       =
T3T3EXTENSION=
CUTVALUEFLAG = 
CUTVALUEFLAG_FLAGS = -DCUT_VALUE_DEBUG

PRIMITIVEFLAG = 
PRIMITIVEFLAG_FLAGS = -DPRIMITIVE_STUDY

all: $(ROOUTIL) efficiency $(EXES)


cutvalue: CUTVALUEFLAG = ${CUTVALUEFLAG_FLAGS}
cutvalue: $(ROOUTIL) efficiency $(EXES)

primitive: PRIMITIVEFLAG = ${PRIMITIVEFLAG_FLAGS}
primitive: $(ROOUTIL) efficiency $(EXES)

cutvalue_primitive: CUTVALUEFLAG = ${CUTVALUEFLAG_FLAGS}
cutvalue_primitive: PRIMITIVEFLAG = ${PRIMITIVEFLAG_FLAGS}
cutvalue_primitive: $(ROOUTIL) efficiency $(EXES)


bin/doAnalysis: bin/doAnalysis.o $(OBJECTS)
	$(CC) $(PT0P8) $(T3T3EXTENSION) $(LDFLAGS) $^ $(ROOTLIBS) $(EXTRACFLAGS) $(CUTVALUEFLAG) $(PRIMITIVEFLAG) $(EXTRAFLAGS) $(DOQUINTUPLET) $(ALPAKAINCLUDE) -o $@

bin/sdl: bin/sdl.o $(OBJECTS)
	$(LD) $(PT0P8) $(T3T3EXTENSION) $(LDFLAGS) $^ $(ROOTLIBS) $(EXTRACFLAGS) $(CUTVALUEFLAG) $(PRIMITIVEFLAG) $(EXTRAFLAGS) $(DOQUINTUPLET) $(ALPAKAINCLUDE) -o $@

%.o: %.cc
	$(CC) $(PT0P8) $(T3T3EXTENSION) $(CFLAGS) $(EXTRACFLAGS) $(CUTVALUEFLAG) $(PRIMITIVEFLAG) $(DOQUINTUPLET) $(ALPAKAINCLUDE) $< -c -o $@

$(ROOUTIL):
	$(MAKE) -C code/rooutil/

efficiency:
	$(MAKE) -C efficiency/

clean:
	rm -f $(OBJECTS) bin/*.o $(EXES)
	rm -f code/rooutil/*.so code/rooutil/*.o
	rm -f bin/sdl.o
	rm -f SDL/*.o
	cd efficiency/ && make clean

.PHONY: $(ROOUTIL) efficiency
