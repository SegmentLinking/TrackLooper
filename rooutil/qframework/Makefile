# this is a standalone makefile for building this package independent
# of any external framework 
#
# if you are aiming at using this framework in connection with other
# packages, please use RootCore for build
#
# for questions or comments, please contact cburgard@cern.ch

# change names (and possibly paths) of binaries
CXX=g++
ROOTCONFIG=root-config
ADDCFLAGS=
ADDCXXFLAGS=-Wno-misleading-indentation
ADDROOTCXXFLAGS=
ADDLDFLAGS=
DICTFLAGS=

-include MyConfig.mk

ifndef ROOTCINT
ifeq ($(shell root-config --version | cut -d. -f1 ),6)
ROOTCINT=rootcling -v0
else
ROOTCINT=rootcint
CINTFLAGS=-c
endif
endif

# you might want to set the TQPATH variable to point towards your
# (primary) checkout directory of the QFramework
ifndef TQPATH
	TQPATH      = $(PWD)
endif

# verbosity control
ifeq ($(MAKE_VERBOSE),)
	MAKE_SILENT=@	
else
	MAKE_SILENT=
endif	

# directory structure
SRC_DIR=Root
INC_DIR=QFramework
LIB_DIR=lib
OBJ_DIR=obj

# setup compiler flags
CFLAGS     = -I/usr/include/libxml2 -I. `xml2-config --cflags` 
ifeq ($(shell ${CXX} --version | grep -q clang),1)
CXXFLAGS   = -Wall -pedantic -std=c++14 $(ADDCXXFLAGS) $(VCXXFLAGS) -Wno-deprecated-declarations -Wno-tautological-compare $(ADDCFLAGS) -fPIC
else
CXXFLAGS   = -Wall -pedantic -std=c++14 $(ADDCXXFLAGS) $(VCXXFLAGS) -Wno-deprecated-declarations $(ADDCFLAGS) -fPIC
endif
ROOTCXXFLAGS = -I$(shell $(ROOTCONFIG) --incdir) $(ADDROOTCXXFLAGS)


# setup linker flags
LIBS       = -pthread  -m64 -lxml2 `xml2-config --libs`
SOFLAGS    = -shared 
ROOTLIBS     = $(shell $(ROOTCONFIG) --libs) -lTreePlayer -lHistPainter
ROOTLDFLAGS  = $(shell $(ROOTCONFIG) --ldflags) 

# automatically search for all classes and headers
SOURCES 	 = $(shell ls $(SRC_DIR)/TQ*.cxx)
TEMPLATES        = $(shell ls $(INC_DIR)/TQ*.tpp)

HEADERS_ 	 = $(shell ls $(INC_DIR)/TQ*.h)
HEADERS 	 = $(HEADERS_:$(SRC_DIR)/%=$(INC_DIR)/%)

OBJECTS_ 	 = $(SOURCES:.cxx=.o)
OBJECTS 	 = $(OBJECTS_:$(SRC_DIR)/%=$(OBJ_DIR)/%)

DICTHDR 	 = $(OBJ_DIR)/QFrameworkCINT.h
DICTSRC 	 = $(OBJ_DIR)/QFrameworkCINT.cxx
DICTOBJ 	 = $(OBJ_DIR)/QFrameworkCINT.dict

#ifeq ($(shell echo | g++ -E --std=c++11 - &> /dev/null && echo $$?),0)
# obligatory 'all' target 
all: $(LIB_DIR)/libQFramework.so $(LIB_DIR)/QFrameworkCINT_rdict.pcm pythonbindings | $(DICTOBJ)
#else
# print warning if compiler is not c++11-compliant
#cpp11warn:
#	@echo "please use a c++11-compliant compiler!"
#endif

# create directory required for build
$(LIB_DIR):
	@echo "creating lib directory"
	$(MAKE_SILENT)mkdir $(LIB_DIR)

$(OBJ_DIR):
	@echo "creating obj directory"
	$(MAKE_SILENT)mkdir $(OBJ_DIR)

# generate LinkDef.h (and other input files)
ADDTARGETS=$(SRC_DIR)/definitions.h $(SRC_DIR)/locals.h $(SRC_DIR)/LinkDef.h
$(ADDTARGETS): addtargets

addtargets: $(HEADERS)
	@echo "creating LinkDef.h"
	$(MAKE_SILENT)cd $(TQPATH)/cmt && bash precompile.RootCore

$(TQPATH)/lib/QFramework.py: $(LIB_DIR)
	$(MAKE_SILENT)ln -sf $(TQPATH)/python $(TQPATH)/lib/QFramework

pythonbindings: $(TQPATH)/lib/QFramework.py
	$(MAKE_SILENT)echo "please ensure that the following is part of your \$$PYTHONPATH:"
	$(MAKE_SILENT)echo "    "$(TQPATH)/lib
	$(MAKE_SILENT)echo "your current \$$PYTHONPATH is:"
	$(MAKE_SILENT)echo "    "$(PYTHONPATH)

# create the dictionaries 
# currently, this still uses CINT
# at some point, we will switch to ROOT6
$(DICTSRC) $(DICTHDR): $(HEADERS) $(ADDTARGETS) | $(OBJ_DIR)
	@echo "creating dictionary"
	$(MAKE_SILENT)$(ROOTCINT) -f $(DICTSRC) $(CINTFLAGS) $(CFLAGS) -p $(HEADERS) $(SRC_DIR)/LinkDef.h

# actually compile the code
$(DICTOBJ): $(DICTSRC)
	@echo "compiling dictionary"
	$(MAKE_SILENT)$(CXX) -c $(CFLAGS) $(CXXFLAGS) $(DICTFLAGS) $(ROOTCXXFLAGS) $^ -o $@ 

$(OBJ_DIR)/%.o: $(SRC_DIR)/%.cxx $(SRC_DIR)/locals.h $(SRC_DIR)/definitions.h | $(OBJ_DIR) 
	@echo "compiling $@"
	$(MAKE_SILENT)$(CXX) -c $(CFLAGS) $(CXXFLAGS) $(ROOTCXXFLAGS) $(SRC_DIR)/$*.cxx -o $@ 

# do the linking
$(LIB_DIR)/libQFramework.so:  $(DICTOBJ) $(OBJECTS) | $(LIB_DIR)
	@echo "linking shared object"
	$(MAKE_SILENT)$(CXX) $(SOFLAGS) $^ -o $@ $(ROOTLDFLAGS) $(LIBS) $(ROOTLIBS) $(ADDLDFLAGS)

# copy the pcm file
$(LIB_DIR)/QFrameworkCINT_rdict.pcm: $(OBJ_DIR)/QFrameworkCINT.cxx | $(LIB_DIR)
	@cp $(OBJ_DIR)/QFrameworkCINT_rdict.pcm $@

cleanconfig:
	rm $(ADDTARGETS)	

clean: cleanconfig
	rm -rf obj lib 

