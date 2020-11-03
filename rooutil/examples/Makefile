#include Makefile.arch

# Extra Packages
SRCS = $(wildcard *.cc)
OBJS = $(SRCS:.cc=.o)
TARGETS = $(SRCS:.cc=.out)
ROOTLIBS:= $(shell root-config --libs) -lTMVA -lEG -lGenVector -lXMLIO -lMLP -lTreePlayer -lRooFit -lRooFitCore

all: $(TARGETS) ../rooutil.so

%.out : %.o ../rooutil.so
	g++ -o $@ $^ $(ROOTLIBS) -L../ -lrooutil -I../

%.o : %.cc
	g++ -Wunused-variable -g -O2 -Wall -fPIC -Wshadow -Woverloaded-virtual $(shell root-config --cflags) -I../ -c $< -o $@

clean:
	rm *.o
	rm *.out
