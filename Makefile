
# Simple makefile

EXES=bin/sdl

ROOUTIL=code/rooutil/

SOURCES=$(wildcard code/core/*.cc)  #$(wildcard SDL/*.cc)
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
ROOTCFLAGS  = $(foreach option, $(shell root-config --cflags), --compiler-options $(option))
CFLAGS      = $(ROOTCFLAGS) --compiler-options -Wall --compiler-options -Wno-unused-function --compiler-options -g --compiler-options -O2 --compiler-options -fPIC --compiler-options -fno-var-tracking -ISDL -I$(shell pwd) -Icode  -Icode/core -I/mnt/data1/dsr/cub -I${CUDA_HOME}/include --compiler-options -fopenmp
EXTRACFLAGS = $(shell rooutil-config)
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
