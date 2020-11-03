#!/bin/env python

def fhash(fname):
    # get a hash of a file (using stupid filesize replacement for now)
    import os
    statinfo = os.stat(fname)
    return statinfo.st_size

def gettimehash(path):
    # get a modification date and hash of a file 
    from os.path import getmtime,isfile
    if not isfile(path): return (None,None)
    return(getmtime(path),fhash(path))

def uptimehash(path,oldtime,oldhash):
    # reset timestamp of file if hash did not change
    if oldhash==None or oldtime==None: return
    newtime,newhash=gettimehash(path)
    from os import utime
    if oldhash==newhash:
        utime(path,(oldtime,oldtime))

# helper class to allow conditional "with" wrapping
class Dummysink(object):
    def write(self, data):
        pass # ignore the data
    def __enter__(self): return self
    def __exit__(*x): pass

def datasink(filename):
    if filename:
        return open(filename, "wt")
    else:
        return Dummysink()

def isdummy(sink):
    return isinstance(sink,Dummysink)


# main function doing all the work
def main(args):
    # some text parsing
    import re
    from os.path import basename,isfile,splitext
    from os.path import join as pjoin
    classre = re.compile("^[ ]*class[ ]*([A-Za-z0-9]+)[ ]*[:{]")
    typedefre = re.compile("^[ ]*typedef[ ]*[A-Za-z0-9<>]*[ ]*([A-Za-z0-9]+)[ ]*.*")
    namespacere = re.compile("^[ ]*namespace[ ]*([A-Za-z0-9]+)[ ]*[:{]")
    nestedre = re.compile(".*//[ ]*nested")
    excludere = re.compile(".*//[ ]*EXCLUDE")

    pytime,pyhash,ldtime,ldhash=None,None,None,None
    if args.python and isfile(args.python):
        pytime,pyhash=gettimehash(args.python)
    if args.linkdef and isfile(args.linkdef):
        ldtime,ldhash=gettimehash(args.linkdef)

    with datasink(args.linkdef) as linkdef:
        with datasink(args.python) as python:
            # generate the linkdef header
            if args.verbose and not isdummy(linkdef):
                print("Generating LinkDef.h")
            linkdef.write("//this is an automatically generated -*- c++ -*- file - EDITS WILL BE LOST!\n")
            linkdef.write("\n")
            linkdef.write("#ifndef __"+args.pkgname+"DICT__\n")
            linkdef.write("#define __"+args.pkgname+"DICT__\n")
            linkdef.write("\n")
            linkdef.write("#pragma GCC diagnostic push\n")
            linkdef.write('#pragma GCC diagnostic ignored "-Winconsistent-missing-override"\n')
            linkdef.write("\n")
            for hfile in args.headers:
                linkdef.write('#include "'+pjoin(args.pkgname,basename(hfile))+'"\n')
            linkdef.write("\n")
            linkdef.write("#ifdef __CINT__\n")
            linkdef.write("\n")
            linkdef.write("#pragma link off all globals;\n")
            linkdef.write("#pragma link off all classes;\n")
            linkdef.write("#pragma link off all functions;\n")
            linkdef.write("#pragma link C++ nestedclass;\n")
            linkdef.write("#pragma link C++ nestedtypedef;\n")
        
            # generate the python header
            if args.verbose and not isdummy(python):
                print("Generating __init__.py")
            python.write("# this is an automatically generated -*- python -*- file - EDITS WILL BE LOST!\n")
            if args.pythonhead:
                python.write("\n\n")
                python.write("################################################\n")
                python.write("###         begin of imported section        ###\n")
                python.write("################################################\n")
                python.write("\n\n")
                with open(args.pythonhead,"rt") as inpython:
                    for line in inpython:
                        python.write(line)

            python.write("\n\n")
            python.write("################################################\n")
            python.write("### begin of automatically generated section ###\n")
            python.write("################################################\n")
            python.write("\n\n")

            # loop over all  header files, collecting class definition, namespace definition and typedef lines
            for hfile in args.headers:
                linkdef.write("// >>> begin "+basename(hfile)+"\n")
                with open(hfile,"rt") as header:
                    for line in header:
                        classmatch = classre.match(line)
                        namespacematch = namespacere.match(line)
                        typedefmatch = typedefre.match(line)
                        # write the #pragma lines
                        if classmatch and not nestedre.match(line):
                            classname = classmatch.group(1)
                            linkdef.write("#pragma link C++ class "+classname+"+;\n")
                            python.write(classname+"=ROOT."+classname+"\n")
                        if typedefmatch:
                            linkdef.write("#pragma link C++ typedef "+typedefmatch.group(1)+";\n")
                        if namespacematch and not excludere.match(line):
                            namespacename = namespacematch.group(1)
                            linkdef.write("#pragma link C++ namespace "+namespacename+";\n")
                            python.write(namespacename+"=ROOT."+namespacename+"\n")
                # import additional linkdef code
                addfiles = [ splitext(hfile)[0]+"LinkDef.h.add" ]
                for addfile in addfiles:
                    if isfile(addfile):
                        with open(addfile,"rt") as afile:
                            for line in afile:
                                linkdef.write(line)
                linkdef.write("// <<< end "+basename(hfile)+"\n")
                linkdef.write("\n")
        
            # write the linkdef footer
            linkdef.write("\n")
            linkdef.write("#endif //__CINT__\n")
            linkdef.write("#endif //__"+args.pkgname+"DICT__\n")

    if args.python:
        uptimehash(args.python,pytime,pyhash)
    if args.linkdef:
        uptimehash(args.linkdef,ldtime,ldhash)


if __name__ == "__main__":
    # parse the CLI arguments
    from argparse import ArgumentParser
    parser = ArgumentParser(description='auto-generate a LinkDef.h file from a set of header files')
    parser.add_argument('--linkdef', metavar='LinkDef.h', type=str, help='path of output LinkDef file')
    parser.add_argument('--python-head', dest="pythonhead", metavar='head.py', type=str, help='header of python bindings')
    parser.add_argument('--python', metavar='__init__.py', type=str, help='path of output python file')
    parser.add_argument('--pkgname', metavar='PKGNAME', type=str, help='name of the package', required=True)
    parser.add_argument('--headers', nargs='+', metavar='HEADER', type=str, help='input files', required=True)
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    main(args)
