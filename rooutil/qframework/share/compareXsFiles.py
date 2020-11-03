import sys
import argparse

def main():
	# parse the CLI arguments
    parser = argparse.ArgumentParser(description='Create plots and cutflows for the HSG3 Run2 Analysis.')
    parser.add_argument('--file1', metavar='FILE1', type=str, help='First cross-section file to use')
    parser.add_argument('--file2', metavar='FILE2', type=str, help='Second cross-section file to use')
    parser.add_argument('--cutoff', metavar='cutoff', type=float, default=0.1,help='cutoff value for warning')
    parser.add_argument('--dsColumnName1', metavar='dsColumnName1', type=str, default='DatasetID',help='name for dataset id column in file 1')
    parser.add_argument('--dsColumnName2', metavar='dsColumnName2', type=str, default='',help='name for dataset id column in file 2')
    parser.add_argument('--compareColumns1', metavar='compareColumns1', type=str, default='xsection',help='')
    parser.add_argument('--compareColumns2', metavar='compareColumns2', type=str, default='xsection',help='')
    parser.add_argument('--compareOnly', metavar='compareOnly', type=str, default='',help='filter which datasets to compare')

    args = parser.parse_args()
    parser1 = TQXSecParser(args.file1)
    parser2 = TQXSecParser(args.file2)
    if args.dsColumnName2 == '':
        args.dsColumnName2 = args.dsColumnName1
    dscol1 = parser1.findColumn(args.dsColumnName1,False)
    dscol2 = parser2.findColumn(args.dsColumnName2,False)
    print((TQStringUtils.makeBoldWhite("Comparing cross-section files {0} and {1}")).Data().format(args.file1,args.file2))
    print((TQStringUtils.makeBoldWhite("Samples only in {0} will not be compared")).Data().format(args.file2))
    
    cols1 =  []
    cols2 = []
    for colname in args.compareColumns1.split(','):
        cols1.append(parser1.findColumn(colname,False))
    for colname in args.compareColumns2.split(','):
        cols2.append(parser2.findColumn(colname,False))
    for row1 in range(1,parser1.getNrows()):
        ds = parser1.getEntryPlain(row1,dscol1)
        if not args.compareOnly == '' and not TQStringUtils.matches(ds,TString(args.compareOnly)):
            continue
        found = False
        for row2 in range(1,parser2.getNrows()):
            if  parser2.getEntryPlain(row2,dscol2) == ds:
                found = True
                xs1 = 1
                xs2 = 1
                for col in cols1:
                    try:
                        xs1 *= float(parser1.getEntryPlain(row1,col).Data())
                    except TypeError:
                        xs1 = -999
                for col in cols2:
                    try:
                        xs2 *= float(parser2.getEntryPlain(row2,col).Data())
                    except TypeError:
                        xs2 = -999
        if not found:
            ERROR(ds.Data()+' not found in second xs-file '+args.file2)
        else:
            relDiff = abs((xs1/xs2)-1)*100
            if relDiff > args.cutoff:
                WARN('Cross-section for dataset {:s} not identical between files. Difference is {:f}%%'.format(ds.Data(), relDiff))

if __name__ == "__main__":
	# this 'black magic' is required to stop ROOT from interfering with the argument parsing	
	if len(sys.argv) < 2 or (sys.argv[1] != "-h" and sys.argv[1] != "--help"):
		from ROOT import *
		from libQFramework import *
		TQLibrary.gQFramework.setApplicationName("compareXsFiles");
		gROOT.SetBatch(True)
	main();

