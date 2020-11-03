#!/bin/env python2

import os
import fileinput

path = os.path.dirname(os.path.realpath(__file__))

print("Welcome to the TQObservable wizard!")

# check if we're inside a RootCore release and ask for package name
try:
    rcdir=os.path.dirname(os.environ['ROOTCOREBIN'])
    print("Found a RootCore setup in "+rcdir)
    print("Should the observable wizard put the files into your current working directory (leave empty) or into some package (type package name)?")
    print("These are your current local packages: ")
    dirs = os.listdir(rcdir)
    pkgs = []
    for pkg in dirs:
        if os.path.exists(os.path.join(rcdir,pkg,"cmt","Makefile.RootCore")):
            pkgs.append(pkg)
            print(pkg)
    pkgname = raw_input("type your choice: ")
    pkgdir = os.path.join(rcdir,pkgname)
    if pkgname and not pkgname in pkgs:
        print("no such package found, aborting")
        quit()
except KeyError:
    print("You do not currently have RootCore setup. If you want to add your new observable to a package, you should first setup RootCore. Please choose:")
    if raw_input("Continue and write files to current working directory (c) or abort (anything else)? ") == "c":
        pkgname = ""
    else:
        print("aborted")
        quit()
    pkgdir = os.getcwd()
    
# ask for classname (for QFramework classes, start with TQ)
classname = ""
if pkgname == "QFramework":
    while not "TQ" in classname:
        classname = raw_input("What is the name of the observable you would like to create? Since you selected QFramework as a target package, the name is required to start with 'TQ'. ")
else:
    classname = raw_input('What is the name of the observable you would like to create? ')

# ask for observable type (select template file)
print("What type of observable would you like to create?")
print("If you want to read xAODs using the xAOD::TEvent mechanism, please type 'Event'")
print("If you want to access the TTree pointer and use TTreeFormula or (not recommended) TTree::SetBranchAddress, please type 'Tree'")
print("If your new observable does not need direct data access but uses some custom mechanism to work, please leave empty.")
type = raw_input("Please specify the type of observable, choose  from {Event,Tree,<empty>}: ")
if type == "Event":
    typename = "TQEventObservable"
elif type == "Tree":
    typename = "TQTreeObservable"
else:
    typename = "TQObservable"

# ask if expression handling is requested
print("Some observable classes have an 'expression' member variable that allows to alter the configuration based on sample tags, but complicates identifying the right observable.")
include_expression = ( raw_input("Should your class have an 'expression' member variable? (y/N) ").lower() == "y")
advanced_expression = False
if include_expression:
    advanced_expression = ( raw_input("Does your class rely on advanced string parsing capabilities for the 'expression' member variable? (y/N) ").lower() == "y")
    
# check for factory
has_factory = ( raw_input("Are you planning to provide a factory for your observable class? (y/N)").lower() == "y")

# print summary and ask for confirmation
print("Your choices:")
print("class name: " + classname)
print("inherits from: " + typename)
print("in package: " + pkgname)
print("written to directory: " + pkgdir)
if include_expression:
    if advanced_expression:
        print("including advanced expression member")
    else:
        print("including expression member")
else:
    print("not including expression member")
if has_factory:
    print("configured for factory use")
build = raw_input("build this observable now? (y/N) ")
if not build.lower() == "y":
    print("aborted")
    quit()

# search and replace the classname in the template files
def searchandreplace(code):
    code = code.replace("ObsName",classname).replace("OBSNAME",classname.upper()).replace("ParentClass",typename)
    if pkgname:
        code = code.replace("PkgName",pkgname)
    else:
        code = code.replace("PkgName/","")
    return code

# write a modified template file
def writefile(fname,code):
    if os.path.exists(fname) and not raw_input("file '"+fname+"' exists - overwrite? (y/N) ").lower() == "y":
        print("aborted")
        quit()
    else:
        f = open(fname, 'w')
        f.write(code)
        f.close()
        print("wrote '"+fname+"'!")

# read the contents of a file to a string
def readfile(fname):
    with open(fname,"r") as f:
        return f.read()
    
# open the header files and patch them together
hcode =readfile(os.path.join(path,"header","head.h"))
hcode = hcode + readfile(os.path.join(path,"header",typename+".h"))
if type != "Event" or advanced_expression:
    hcode = hcode + readfile(os.path.join(path,"header","initializefinalize.h"))
if include_expression:
    hcode = hcode + readfile(os.path.join(path,"header","expression.h"))
    if advanced_expression:
        hcode = hcode + readfile(os.path.join(path,"header","advancedexpression.h"))
else:
    hcode = hcode + readfile(os.path.join(path,"header","noexpression.h"))
if has_factory:
    hcode = hcode + readfile(os.path.join(path,"header","factory.h"))
else:
    hcode = hcode + readfile(os.path.join(path,"header","nofactory.h"))
hcode = hcode + readfile(os.path.join(path,"header","foot.h"))
hcode = searchandreplace(hcode)

cxxcode =readfile(os.path.join(path,"implementation","head.cxx"))
cxxcode = cxxcode + readfile(os.path.join(path,"implementation",typename+".cxx"))
if not advanced_expression:
    if type == "Tree":
        cxxcode = cxxcode + readfile(os.path.join(path,"implementation","initializefinalize.tree.cxx"))
    elif type != "Event":
        cxxcode = cxxcode + readfile(os.path.join(path,"implementation","initializefinalize.cxx"))
if include_expression:
    cxxcode = cxxcode + readfile(os.path.join(path,"implementation","expression.cxx"))
    if advanced_expression:
        cxxcode = cxxcode + readfile(os.path.join(path,"implementation","advancedexpression.cxx"))
else:
    cxxcode = cxxcode + readfile(os.path.join(path,"implementation","noexpression.cxx"))
if has_factory:
    cxxcode = cxxcode + readfile(os.path.join(path,"implementation","factory.cxx"))
cxxcode = searchandreplace(cxxcode)
    
# write the new source files to the target directory
if pkgname:
    cxxfile = os.path.join(pkgdir,"Root",classname+".cxx")
    hfile   = os.path.join(pkgdir,pkgname,classname+".h")
else:
    cxxfile = os.path.join(pkgdir,classname+".cxx")
    hfile   = os.path.join(pkgdir,classname+".h")
writefile(hfile,hcode)
writefile(cxxfile,cxxcode)

# look for a linkdeffile, ask the user if he/she wants us to patch it
linkdeffile = os.path.join(pkgdir,"Root","LinkDef.h")
if os.path.exists(linkdeffile):
    if raw_input("the wizard has detected a LinkDef.h file in the target package - do you want to add an entry for your observable class? (Y/n)").lower() != "n":
        watch_include = False
        watch_pragma = False
        inclstr = '#include "'+os.path.join(pkgname,classname)+'.h"'
        pragmastr = '#pragma link C++ class '+classname+"+;"
        # check if it's already patched
        with open(linkdeffile, 'r') as f:
            if any(pragmastr == x.rstrip('\r\n') for x in f):
                done_pragma = True
            else:
                done_pragma = False
            if any(inclstr == x.rstrip('\r\n') for x in f):
                done_include = True
            else:
                done_include = False
        # search for the 'right' place to insert it
        for line in fileinput.input(linkdeffile, inplace=1):
            if not done_include and line.lower().startswith('#include'):
                watch_include = True
            if watch_include and not line.lower().startswith('#include'):
                print(inclstr)
                watch_include = False
                done_include = True
            if not done_pragma and line.lower().startswith('#pragma link c++ class'):
                watch_pragma = True
            if watch_pragma and not line.lower().startswith('#pragma link c++ class'):
                print(pragmastr)
                watch_pragma = False
                done_pragma = True
            print(line.strip())




