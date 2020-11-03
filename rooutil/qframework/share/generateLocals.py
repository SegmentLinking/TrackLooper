#/bin/env python

import subprocess
if "check_output" not in dir( subprocess ): # duck punch it in!
    def f(*popenargs, **kwargs):
        if 'stdout' in kwargs:
            raise ValueError('stdout argument not allowed, it will be overridden.')
        process = subprocess.Popen(stdout=subprocess.PIPE, *popenargs, **kwargs)
        output, unused_err = process.communicate()
        retcode = process.poll()
        if retcode:
            cmd = kwargs.get("args")
            if cmd is None:
                cmd = popenargs[0]
            raise subprocess.CalledProcessError(retcode, cmd)
        return output
    subprocess.check_output = f

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

def writestrdef(output,key,value):
    # write a string preprocessor define
    if value == None: return
    output.write("#define "+key+" \""+str(value)+"\"\n")

def writelistdef(output,key,value):
    # write a list preprocessor define
    if value == None or len(value)==0: return
    output.write("#define "+key+" {"+",".join([ '"'+str(v)+'"' for v in value])+"}\n")
    
def writebooldef(output,key,value):
    # write a boolean preprocessor define
    output.write("#define "+key+" "+str(int(value))+"\n")

def getstring(cmd,cwd):
    # obtain output from a command
    from subprocess import check_output as call
    from subprocess import CalledProcessError
    try:
        s = call(cmd,cwd=cwd,shell=True).strip()
        return s.decode()
    except CalledProcessError:
        return "";
    
def testheader(hfile):
    # test if a header file is present
    import subprocess
    compiler = subprocess.Popen("g++ -xc++ -E $(root-config --cflags) -",shell=True,stdin=subprocess.PIPE,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    compiler.communicate(input=str('#include <'+hfile+'>').encode("ascii"))
    if compiler.returncode == 0:
        return True
    return False

def findbinary(binname):
    # find a binary
    from subprocess import check_output as call
    from subprocess import CalledProcessError,PIPE
    try: 
        return str(call(("which",binname),stderr=PIPE)).strip()
    except CalledProcessError:
        return None

def main(args):
    # main function putting together all the pieces
    if args.verbose:
        print("Generating "+args.output)
    ftime,fhash = gettimehash(args.output)
    with open(args.output,"wt") as output:
        if args.root:
            writestrdef(output,"ROOTVERSION",getstring("root-config --version",args.wd))
        if args.gcc:
            writestrdef(output,"GCCVERSION",getstring("gcc -dumpversion",args.wd))
        if args.svn:
            writestrdef(output,"SVNVERSION",getstring("svnversion 2>/dev/null",args.wd))
        if args.git:
            writestrdef(output,"GITVERSION",getstring("git show -s --format=%H",args.wd))
        for name,path in args.namedpath:
            writestrdef(output,name,path)
        for var,bin in args.findbin:
            writestrdef(output,var,findbinary(bin))
        for flag,header in args.flagheader:
            if testheader(header):
                writebooldef(output,flag,1)
        if len(args.packages) > 0:
            writelistdef(output,"PACKAGES",[item for element in args.packages for item in element.split()])
    uptimehash(args.output,ftime,fhash)

if __name__ == "__main__":
    # parse the CLI arguments
    from argparse import ArgumentParser
    parser = ArgumentParser(description='auto-generate a header file with some local definitions')
    parser.add_argument('--output', metavar='definitions.h', type=str, help='path of output file',required=True)
    parser.add_argument('--set-working-directory', metavar='definitions.h', dest="wd", type=str, help='working directory',required=True)
    parser.add_argument('--root', action='store_true')
    parser.add_argument('--git', action='store_true')
    parser.add_argument('--svn', action='store_true')
    parser.add_argument('--gcc', action='store_true')
    parser.add_argument('--packages', nargs="*", type=str, default=[])
    parser.add_argument('--set-named-path', action='append',nargs=2, dest="namedpath",  metavar=('name','path'),      default=[])
    parser.add_argument('--flag-header',    action='append',nargs=2, dest="flagheader", metavar=('has_flag','header'),default=[])
    parser.add_argument('--find-binary',    action='append',nargs=2, dest="findbin",    metavar=('variable','binary'),default=[])
    parser.add_argument('--verbose', action='store_true')
    args = parser.parse_args()
    main(args)
