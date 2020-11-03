# Simple makefile

EXES=bin/doAnalysis bin/sdl #bin/mtv bin/tracklet bin/sdl

SOURCES=$(wildcard src/*.cc) $(wildcard src/AnalysisInterface/*.cc) #$(wildcard SDL/*.cc)
OBJECTS=$(SOURCES:.cc=.o) $(wildcard SDL/sdl.so)
HEADERS=$(SOURCES:.cc=.h)

CC          = nvcc
CXX         = nvcc
CXXFLAGS    = -g -O2 --compiler-options -Wall --compiler-options -fPIC --compiler-options -Wshadow --compiler-options -Woverloaded-virtual
LD          = g++
LDFLAGS     = -g -O2 -Wall -fPIC -Wshadow -Woverloaded-virtual
SOFLAGS     = -g -shared
CXXFLAGS    = -g -O2 -Wall -fPIC -Wshadow -Woverloaded-virtual
LDFLAGS     = -g -O2 
ROOTLIBS    = $(shell root-config --libs)
#ROOTCFLAGS  = $(shell `root-config --cflags)
ROOTCFLAGS   = --compiler-options -pthread --compiler-options -std=c++17 -m64 -I/cvmfs/cms.cern.ch/slc7_amd64_gcc700/cms/cmssw/CMSSW_11_0_0_pre6/external/slc7_amd64_gcc700/bin/../../../../../../../slc7_amd64_gcc700/lcg/root/6.14.09-nmpfii5/include
CXXFLAGS    = $(ROOTCFLAGS) -ISDL -I$(shell pwd) -Isrc -Isrc/AnalysisInterface
CFLAGS      = $(ROOTCFLAGS) --compiler-options -Wall --compiler-options -Wno-unused-function --compiler-options -g --compiler-options -O2 --compiler-options -fPIC --compiler-options -fno-var-tracking -ISDL -I$(shell pwd) -Isrc -Isrc/AnalysisInterface -I/cvmfs/cms.cern.ch/slc7_amd64_gcc700/external/cuda/10.1.105-pafccj2/include --compiler-options -fopenmp
EXTRACFLAGS = $(shell rooutil-config)
EXTRAFLAGS  = -fPIC -ITMultiDrawTreePlayer -Wunused-variable -lTMVA -lEG -lGenVector -lXMLIO -lMLP -lTreePlayer -L/cvmfs/cms.cern.ch/slc7_amd64_gcc700/external/cuda/10.1.105-pafccj2/lib64 -lcudart -fopenmp

all: $(EXES)

bin/doAnalysis: bin/doAnalysis.o $(OBJECTS)
	$(LD) $(LDFLAGS) $^ $(ROOTLIBS) $(EXTRAFLAGS) -o $@

bin/mtv: bin/mtv.o $(OBJECTS)
	$(LD) $(LDFLAGS) $^ $(ROOTLIBS) $(EXTRAFLAGS) -o $@

bin/tracklet: bin/tracklet.o $(OBJECTS)
	$(LD)  $(LDFLAGS) $^ $(ROOTLIBS) $(EXTRAFLAGS) -o $@

bin/sdl: bin/sdl.o $(OBJECTS)
	$(LD) $(LDFLAGS) $^ $(ROOTLIBS) $(EXTRAFLAGS) -o $@

%.o: %.cc
	$(CC) $(CFLAGS) $(EXTRACFLAGS) $< -dc -o $@

clean:
	rm -f $(OBJECTS) bin/*.o $(EXES)
