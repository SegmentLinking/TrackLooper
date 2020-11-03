include Makefile.arch

#
# stuff to make
#
SOURCES=$(wildcard *.cc)
OBJECTS=$(SOURCES:.cc=.o)
HEADERS=$(SOURCES:.cc=.h)
SOURCES=rooutil.cc
HEADERS=rooutil.h
OBJECTS=rooutil.o
LIB=rooutil.so

#
# how to make it 
#

$(LIB): $(SOURCES) $(HEADERS)
	$(LD) $(CXXFLAGS) $(LDFLAGS) -fPIC -ITMultiDrawTreePlayer -Wunused-variable $(SOFLAGS) $(SOURCES) $(ROOTLIBS) -lTMVA -lEG -lGenVector -lXMLIO -lMLP -lTreePlayer -o $@
	ln -sf rooutil.so librooutil.so

all: $(LIB) 
clean:
	rm -f *.o \
	rm -f *.d \
	rm -f *.so \
