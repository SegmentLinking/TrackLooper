#include "commandutil.h"

std::tuple<TString, TString, int> RooUtil::CommandUtil::parseArgs(int argc, char** argv)
{
    // Argument checking
    if (argc < 3)
    {
        std::cout << "Usage:" << std::endl;
        std::cout << "  $ ./<executable name> INPUTFILES OUTPUTFILE [NEVENTS]" << std::endl;
        std::cout << std::endl;
        std::cout << "  INPUTFILES      comma separated file list" << std::endl;
        std::cout << "  OUTPUTFILE      output file name" << std::endl;
        std::cout << "  [NEVENTS=-1]    # of events to run over" << std::endl;
        std::cout << std::endl;
        exit(255);
    }

    TString inputFileList = argv[1];
    TString outputFileName = argv[2];
    int nEvents = argc > 3 ? atoi(argv[3]) : -1;

    return std::make_tuple(inputFileList, outputFileName, nEvents);

}
