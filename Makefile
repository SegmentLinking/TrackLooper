
# Simple makefile

EXES=bin/sdl

ROOUTIL=code/rooutil/

SOURCES=$(wildcard code/core/*.cc) $(wildcard code/AnalysisInterface/*.cc) #$(wildcard SDL/*.cc)
OBJECTS=$(SOURCES:.cc=.o) $(wildcard ${TRACKLOOPERDIR}/SDL/libsdl.so)
HEADERS=$(SOURCES:.cc=.h)

CC          = nvcc
CXX         = nvcc
CXXFLAGS    = -g -O2 --compiler-options -Wall --compiler-options -fPIC --compiler-options -Wshadow --compiler-options -Woverloaded-virtual -G -lineinfo  -fopenmp -lgomp --default-stream per-thread
LD          = g++
LDFLAGS     = -g -O2 -Wall -fPIC -Wshadow -Woverloaded-virtual -I/mnt/data1/dsr/cub
SOFLAGS     = -g -shared
CXXFLAGS    = -g -O2 -Wall -fPIC -Wshadow -Woverloaded-virtual
LDFLAGS     = -g -O2
ROOTLIBS    = $(shell root-config --libs)
ROOTCFLAGS = --compiler-options -pthread --compiler-options -std=c++17 -m64 -I/cvmfs/cms.cern.ch/el8_amd64_gcc10/cms/cmssw/CMSSW_12_5_0_pre5/external/el8_amd64_gcc10/bin/../../../../../../../el8_amd64_gcc10/lcg/root/6.24.07-a31cbfc28a0c92b3c007615905b5b9b2/include
CXXFLAGS    = $(ROOTCFLAGS) -ISDL -I$(shell pwd) -Icode -Icode/AnalysisInterface -Icode/core
CFLAGS      = $(ROOTCFLAGS) --compiler-options -Wall --compiler-options -Wno-unused-function --compiler-options -g --compiler-options -O2 --compiler-options -fPIC --compiler-options -fno-var-tracking -ISDL -I$(shell pwd) -Icode -Icode/AnalysisInterface -Icode/core -I/mnt/data1/dsr/cub -I/cvmfs/cms.cern.ch/el8_amd64_gcc10/external/cuda/11.5.2-c927b7e765e06433950d8a7eab9eddb4/include --compiler-options -fopenmp
EXTRACFLAGS = $(shell rooutil-config)
EXTRAFLAGS  = -fPIC -ITMultiDrawTreePlayer -Wunused-variable -lTMVA -lEG -lGenVector -lXMLIO -lMLP -lTreePlayer -L/cvmfs/cms.cern.ch/el8_amd64_gcc10/external/cuda/11.5.2-c927b7e765e06433950d8a7eab9eddb4/lib64 -lcudart -fopenmp
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
	$(LD) $(PT0P8) $(T3T3EXTENSION) $(LDFLAGS) $^ $(ROOTLIBS) $(EXTRACFLAGS) $(CUTVALUEFLAG) $(PRIMITIVEFLAG) $(EXTRAFLAGS) $(DOQUINTUPLET) -o $@

bin/sdl: bin/sdl.o $(OBJECTS)
	$(LD) $(PT0P8) $(T3T3EXTENSION) $(LDFLAGS) $^ $(ROOTLIBS) $(EXTRACFLAGS) $(CUTVALUEFLAG) $(PRIMITIVEFLAG) $(EXTRAFLAGS) $(DOQUINTUPLET) -o $@

%.o: %.cc
	$(CC) $(PT0P8) $(T3T3EXTENSION) $(CFLAGS) $(EXTRACFLAGS) $(CUTVALUEFLAG) $(PRIMITIVEFLAG) $(DOQUINTUPLET) $< -dc -o $@

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
