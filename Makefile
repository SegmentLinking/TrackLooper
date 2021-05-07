
# Simple makefile

EXES=bin/sdl

ROOUTIL=code/rooutil/

SOURCES=$(wildcard code/core/*.cc) $(wildcard code/AnalysisInterface/*.cc) #$(wildcard SDL/*.cc)
OBJECTS=$(SOURCES:.cc=.o) $(wildcard ${TRACKLOOPERDIR}/SDL/sdl.so)
HEADERS=$(SOURCES:.cc=.h)

CC          = nvcc
CXX         = nvcc
CXXFLAGS    = -g -O2 --compiler-options -Wall --compiler-options -fPIC --compiler-options -Wshadow --compiler-options -Woverloaded-virtual
LD          = g++
LDFLAGS     = -g -O2 -Wall -fPIC -Wshadow -Woverloaded-virtual -I/mnt/data1/dsr/cub
SOFLAGS     = -g -shared
CXXFLAGS    = -g -O2 -Wall -fPIC -Wshadow -Woverloaded-virtual
LDFLAGS     = -g -O2
ROOTLIBS    = $(shell root-config --libs)
#ROOTCFLAGS  = $(shell `root-config --cflags)
#ROOTCFLAGS   = --compiler-options -pthread --compiler-options -std=c++17 -m64 -I/cvmfs/cms.cern.ch/slc7_amd64_gcc700/cms/cmssw/CMSSW_11_0_0_pre6/external/slc7_amd64_gcc700/bin/../../../../../../../slc7_amd64_gcc700/lcg/root/6.14.09-nmpfii5/include
ROOTCFLAGS = --compiler-options -pthread --compiler-options -std=c++17 -m64 -I/cvmfs/cms.cern.ch/slc7_amd64_gcc900/cms/cmssw/CMSSW_11_2_0_pre5/external/slc7_amd64_gcc900/bin/../../../../../../../slc7_amd64_gcc900/lcg/root/6.20.06-ghbfee3/include
CXXFLAGS    = $(ROOTCFLAGS) -ISDL -I$(shell pwd) -Icode -Icode/AnalysisInterface -Icode/core
CFLAGS      = $(ROOTCFLAGS) --compiler-options -Wall --compiler-options -Wno-unused-function --compiler-options -g --compiler-options -O2 --compiler-options -fPIC --compiler-options -fno-var-tracking -ISDL -I$(shell pwd) -Icode -Icode/AnalysisInterface -Icode/core -I/mnt/data1/dsr/cub -I/cvmfs/cms.cern.ch/slc7_amd64_gcc900/external/cuda/11.0.3/include --compiler-options -fopenmp
EXTRACFLAGS = $(shell rooutil-config)
EXTRAFLAGS  = -fPIC -ITMultiDrawTreePlayer -Wunused-variable -lTMVA -lEG -lGenVector -lXMLIO -lMLP -lTreePlayer -L/cvmfs/cms.cern.ch/slc7_amd64_gcc900/external/cuda/11.0.3/lib64 -lcudart -fopenmp
DOQUINTUPLET = -DDO_QUINTUPLET

CUTVALUEFLAG = 
CUTVALUEFLAG_FLAGS = -DCUT_VALUE_DEBUG

PRIMITIVEFLAG = 
PRIMITIVEFLAG_FLAGS = -DPRIMITIVE_STUDY

all: $(ROOUTIL) efficiency $(EXES)


cutvalue: CUTVALUEFLAG = ${CUTVALUEFLAG_FLAGS}
cutvalue: $(ROOUTIL) efficiency $(EXES)

primitive: PRIMITIVEFLAG = ${PRIMITIVEFLAG_FLAGS}
primitive: $(ROOUTIL) efficiency $(EXES)


bin/doAnalysis: bin/doAnalysis.o $(OBJECTS)
	$(LD) $(LDFLAGS) $^ $(ROOTLIBS) $(EXTRACFLAGS) $(CUTVALUEFLAG) $(PRIMITIVEFLAG) $(EXTRAFLAGS) $(DOQUINTUPLET) -o $@

bin/sdl: bin/sdl.o $(OBJECTS)
	$(LD) $(LDFLAGS) $^ $(ROOTLIBS) $(EXTRACFLAGS) $(CUTVALUEFLAG) $(PRIMITIVEFLAG) $(EXTRAFLAGS) $(DOQUINTUPLET) -o $@

%.o: %.cc
	$(CC) $(CFLAGS) $(EXTRACFLAGS) $(CUTVALUEFLAG) $(PRIMITIVEFLAG) $(DOQUINTUPLET) $< -dc -o $@

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
