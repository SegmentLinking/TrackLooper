# Simple makefile

EXE=doAnalysis

SOURCES=$(wildcard *.cc)
OBJECTS=$(SOURCES:.cc=.o)
HEADERS=$(SOURCES:.cc=.h)

CC          = g++
CXX         = g++
CXXFLAGS    = -g -O2 -Wall -fPIC -Wshadow -Woverloaded-virtual
LD          = g++
LDFLAGS     = -g -O2
SOFLAGS     = -g -shared
CXXFLAGS    = -g -O2 -Wall -fPIC -Wshadow -Woverloaded-virtual
LDFLAGS     = -g -O2
ROOTLIBS    = $(shell root-config --libs)
ROOTCFLAGS  = $(shell root-config --cflags)
CXXFLAGS   += $(ROOTCFLAGS)
CFLAGS      = $(ROOTCFLAGS) -Wall -Wno-unused-function -g -O2 -fPIC -fno-var-tracking
EXTRACFLAGS = $(shell rooutil-config)
EXTRAFLAGS  = -fPIC -ITMultiDrawTreePlayer -Wunused-variable -lTMVA -lEG -lGenVector -lXMLIO -lMLP -lTreePlayer $(shell rooutil-config)

.PHONY: rooutil

all: doAnalysis

$(EXE): rooutil efficiency.o helper.o SDL.o
	$(LD) $(CXXFLAGS) $(LDFLAGS) efficiency.o helper.o SDL.o $(ROOTLIBS) $(EXTRAFLAGS) -o $@

SDL.o: SDL.cc SDL.h
	$(CC) $(CFLAGS) $(EXTRACFLAGS) $< -c -o $@

efficiency.o: efficiency.cc efficiency.h sdl_types.h
	$(CC) $(CFLAGS) $(EXTRACFLAGS) $< -c -o $@

helper.o: helper.cc helper.h sdl_types.h
	$(CC) $(CFLAGS) $(EXTRACFLAGS) $< -c -o $@

rooutil:
	$(MAKE) -C ../src/rooutil

clean:
	rm -f $(OBJECTS) $(EXE)
