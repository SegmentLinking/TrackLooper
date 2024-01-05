
# Simple makefile

EXES=bin/sdl

ROOUTIL=code/rooutil/

SOURCES=$(wildcard code/core/*.cc)
OBJECTS=$(SOURCES:.cc=.o)
HEADERS=$(SOURCES:.cc=.h)

CXX         = g++
CXXFLAGS    = -g -O2 -Wall -fPIC -Wshadow -Woverloaded-virtual -lineinfo  -fopenmp -lgomp --default-stream per-thread
LDFLAGS     = -g -O2 -Wall -fPIC -Wshadow -Woverloaded-virtual
SOFLAGS     = -g -shared
CXXFLAGS    = -g -O2 -Wall -fPIC -Wshadow -Woverloaded-virtual
LDFLAGS     = -g -O2 -lsdl -L${TRACKLOOPERDIR}/SDL/cuda -L${TRACKLOOPERDIR}/SDL/cpu
ROOTLIBS    = $(shell root-config --libs)
ROOTCFLAGS  = $(foreach option, $(shell root-config --cflags), $(option))
ALPAKAINCLUDE = -I${ALPAKA_ROOT}/include -I/${BOOST_ROOT}/include -std=c++17 -DALPAKA_DEBUG=0
ALPAKASERIAL = -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED
CFLAGS      = $(ROOTCFLAGS)  -Wall  -Wno-unused-function  -g  -O2  -fPIC  -fno-var-tracking -ISDL -I$(shell pwd) -Icode  -Icode/core -I${CUDA_HOME}/include  -fopenmp
EXTRACFLAGS = $(shell rooutil-config) -g
EXTRAFLAGS  = -fPIC -ITMultiDrawTreePlayer -Wunused-variable -lTMVA -lEG -lGenVector -lXMLIO -lMLP -lTreePlayer -L${CUDA_HOME}/lib64 -lcudart -fopenmp
DOQUINTUPLET = #-DFP16_Base
PTCUTFLAG    =
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
	$(CXX) $(PTCUTFLAG) $(T3T3EXTENSION) $(LDFLAGS) $^ $(ROOTLIBS) $(EXTRACFLAGS) $(CUTVALUEFLAG) $(PRIMITIVEFLAG) $(EXTRAFLAGS) $(DOQUINTUPLET) $(ALPAKAINCLUDE) $(ALPAKASERIAL) -o $@

bin/sdl: bin/sdl.o $(OBJECTS)
	$(CXX) $(PTCUTFLAG) $(T3T3EXTENSION) $(LDFLAGS) $^ $(ROOTLIBS) $(EXTRACFLAGS) $(CUTVALUEFLAG) $(PRIMITIVEFLAG) $(EXTRAFLAGS) $(DOQUINTUPLET) $(ALPAKAINCLUDE) $(ALPAKASERIAL) -o $@

%.o: %.cc
	$(CXX) $(PTCUTFLAG) $(T3T3EXTENSION) $(CFLAGS) $(EXTRACFLAGS) $(CUTVALUEFLAG) $(PRIMITIVEFLAG) $(DOQUINTUPLET) $(ALPAKAINCLUDE) $(ALPAKASERIAL) $< -c -o $@


.PHONY: $(ROOUTIL) efficiency clean format check check-fix

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

format:
	clang-format -i SDL/*.cc SDL/*.h

check:
	clang-tidy SDL/*.cc SDL/*.h -- --language=c++ -std=c++17 -I. -I$(ROOT_ROOT)/include -I$(CMSSW_BASE)/src -I${ALPAKA_ROOT}/include -I/${BOOST_ROOT}/include -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED -I/cvmfs/cms.cern.ch/el8_amd64_gcc11/external/gcc/11.4.1-30ebdc301ebd200f2ae0e3d880258e65/include/c++/11.4.1 -I/cvmfs/cms.cern.ch/el8_amd64_gcc11/external/gcc/11.4.1-30ebdc301ebd200f2ae0e3d880258e65/include/c++/11.4.1/x86_64-redhat-linux-gnu/

check-fix:
	clang-tidy --fix --fix-errors --fix-notes SDL/*.cc SDL/*.h -- --language=c++ -std=c++17 -I. -I$(ROOT_ROOT)/include -I$(CMSSW_BASE)/src -I${ALPAKA_ROOT}/include -I/${BOOST_ROOT}/include -DALPAKA_ACC_CPU_B_SEQ_T_SEQ_ENABLED -I/cvmfs/cms.cern.ch/el8_amd64_gcc11/external/gcc/11.4.1-30ebdc301ebd200f2ae0e3d880258e65/include/c++/11.4.1 -I/cvmfs/cms.cern.ch/el8_amd64_gcc11/external/gcc/11.4.1-30ebdc301ebd200f2ae0e3d880258e65/include/c++/11.4.1/x86_64-redhat-linux-gnu/
