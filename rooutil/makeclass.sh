#!/bin/bash

# Neat bash trick to get the path where this file sits
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

# Loading pretty print utilities (colorized with some icons)
source $DIR/utils.sh
ERROR=e_error
UL=e_underline
SUCCESS=e_success


echo ""
echo "========================================================================================================="
echo ""
echo ""
echo "                                       MakeClass script"
echo ""
echo ""
echo "========================================================================================================="

usage(){
    echo ""
    echo "This script creates CLASSNAME.cc and CLASSNAME.h file."
    echo "The generated source C++ codes can be used to read a ROOT's TTree with rest of the RooUtil tool at ease."
    echo "The generation of the code relies on:"
    echo "  > https://github.com/cmstas/Software/blob/master/makeCMS3ClassFiles/makeCMS3ClassFiles.C"
    echo ""
    echo "Usage:"
    echo ""
    echo ${green}"  > sh $(basename $0) [-f] [-h] [-x] ROOTFILE TTREENAME CLASSNAME [NAMESPACENAME=tas] [CLASSINSTANCENAME=cms3] "${reset}
    echo ""
    echo ""
    echo ${green}" -h ${reset}: print this message"
    echo ${green}" -f ${reset}: force run this script"
    echo ${green}" -x ${reset}: create additional looper template files (i.e. process.cc, Makefile)"
    echo ""
    echo ${green}" ROOTFILE          ${reset}= Path to the root file that holds an example TTree that you wish to study."
    echo ${green}" TREENAME          ${reset}= The TTree object TKey name in the ROOTFILE"
    echo ${green}" CLASSNAME         ${reset}= The name you want to give to the class you are creating"
    echo ${green}" NAMESPACENAME     ${reset}= The name you want to give to the namespace for accessing the ttree"
    echo ${green}" CLASSINSTANCENAME ${reset}= The name of the global instance of the class that you are trying to create"
    echo "                     (defaults to 'cms3')"
    echo ""
    e_underline "${red}NOTE: If no argument is given, it will assume to create a CMS3 looper"${reset}
    echo ""
    echo ""
    echo ""
    exit
}

# Command-line opts
while getopts ":fxh" OPTION; do
  case $OPTION in
    f) FORCE=true;;
    x) GENERATEEXTRACODE=true;;
    h) usage;;
    :) usage;;
  esac
done

# To shift away the parsed options
shift $(($OPTIND - 1))

if [ -z $1 ]; then
    if [ -z $2 ]; then
        if [ -z $3 ]; then
            echo "RooUtil:: No argument is given"
            echo "RooUtil:: The script will assume to create a CMS3 looper"
        fi
    fi
fi

if [ -z $1 ]; then usage; fi
if [ -z $2 ]; then usage; fi
if [ -z $3 ]; then usage; fi
ROOTFILE=$1
TTREENAME=$2
MAKECLASSNAME=$3
NAMESPACENAME=$4
TREEINSTANCENAME=$5
if [ -z ${NAMESPACENAME} ]; then NAMESPACENAME=tas; fi
if [ -z ${TREEINSTANCENAME} ]; then TREEINSTANCENAME=cms3; fi

echo ""
e_arrow "RooUtil:: The user has provided following options"
e_arrow "RooUtil:: $(date)"
e_arrow "RooUtil:: =========================================="
e_arrow "RooUtil::  ROOTFILE=$ROOTFILE"
e_arrow "RooUtil::  TREENAME=$TTREENAME"
e_arrow "RooUtil::  MAKECLASSNAME=$MAKECLASSNAME"
e_arrow "RooUtil::  TREEINSTANCENAME=$TREEINSTANCENAME"
e_arrow "RooUtil:: =========================================="
e_arrow "RooUtil:: "

# Print
e_arrow "RooUtil:: Generating ${MAKECLASSNAME}.cc/h file which can load the TTree content from ${ROOTFILE}:${TREENAME} ..."

ROOTFILE=$('cd' $(dirname ${ROOTFILE}); pwd)/$(basename $1)

# Check whether the file already exists
if [ -e ${MAKECLASSNAME}.cc ]; then
    if [ "${FORCE}" == true ]; then
        :
    else
        e_error "RooUtil:: ${MAKECLASSNAME}.cc already exists!"
        e_error "RooUtil:: Do you want to override? If so, provide option -f. For more info use option -h"
        exit
    fi
fi

# Check whether the file already exists
if [ -e ${MAKECLASSNAME}.h ]; then
    if [ "${FORCE}" == true ]; then
        :
    else
        e_error "RooUtil:: ${MAKECLASSNAME}.h already exists!"
        e_error "RooUtil:: Do you want to override? If so, provide option -f. For more info use option -h"
        exit
    fi
fi

source $DIR/root.sh ""

if [ -e $DIR/makeCMS3ClassFiles.C ]; then
  echo "running makeCMS3ClassFiles.C"
  root -l -b -q $DIR/makeCMS3ClassFiles.C\(\"${ROOTFILE}\",\"${TTREENAME}\",\"${MAKECLASSNAME}\",\"${NAMESPACENAME}\",\"${TREEINSTANCENAME}\"\)  &> /dev/null
fi

if [ $? -eq 0 ]; then
    e_arrow "RooUtil:: Generated ${MAKECLASSNAME}.cc/h successfully!"
else
    e_error "RooUtil:: Failed to generate ${MAKECLASSNAME}.cc/h!"
    exit
fi

if [ "$GENERATEEXTRACODE" == true ]; then

#    #
#    # Add "rooutil" to the class
#    #
#    echo "#include \"rooutil/rooutil.cc\"" >> ${MAKECLASSNAME}.cc
#    echo "#include \"rooutil/rooutil.h\"" >> ${MAKECLASSNAME}.h


    if [ -e process.cc ]; then
        e_error "RooUtil:: process.cc already exists. We will leave it alone. Erase it if you want to override"
    else

        #
        # Create process.cc
        #
        echo "#include \"${MAKECLASSNAME}.h\"                                                                                                                                                                                                   ">>process.cc
        echo "#include \"rooutil.h\"                                                                                                                                                                                                            ">>process.cc
        echo "#include \"cxxopts.h\"                                                                                                                                                                                                            ">>process.cc
        echo "                                                                                                                                                                                                                                  ">>process.cc
        echo "class AnalysisConfig {                                                                                                                                                                                                            ">>process.cc
        echo "                                                                                                                                                                                                                                  ">>process.cc
        echo "public:                                                                                                                                                                                                                           ">>process.cc
        echo "                                                                                                                                                                                                                                  ">>process.cc
        echo "    // TString that holds the input file list (comma separated)                                                                                                                                                                   ">>process.cc
        echo "    TString input_file_list_tstring;                                                                                                                                                                                              ">>process.cc
        echo "                                                                                                                                                                                                                                  ">>process.cc
        echo "    // TString that holds the name of the TTree to open for each input files                                                                                                                                                      ">>process.cc
        echo "    TString input_tree_name;                                                                                                                                                                                                      ">>process.cc
        echo "                                                                                                                                                                                                                                  ">>process.cc
        echo "    // Output TFile                                                                                                                                                                                                               ">>process.cc
        echo "    TFile* output_tfile;                                                                                                                                                                                                          ">>process.cc
        echo "                                                                                                                                                                                                                                  ">>process.cc
        echo "    // Number of events to loop over                                                                                                                                                                                              ">>process.cc
        echo "    int n_events;                                                                                                                                                                                                                 ">>process.cc
        echo "                                                                                                                                                                                                                                  ">>process.cc
        echo "    // Jobs to split (if this number is positive, then we will skip certain number of events)                                                                                                                                     ">>process.cc 
        echo "    // If there are N events, and was asked to split 2 ways, then depending on job_index, it will run over first half or latter half                                                                                              ">>process.cc 
        echo "    int nsplit_jobs;                                                                                                                                                                                                              ">>process.cc 
        echo "                                                                                                                                                                                                                                  ">>process.cc 
        echo "    // Job index (assuming nsplit_jobs is set, the job_index determine where to loop over)                                                                                                                                        ">>process.cc 
        echo "    int job_index;                                                                                                                                                                                                                ">>process.cc 
        echo "                                                                                                                                                                                                                                  ">>process.cc 
        echo "    // Debug boolean                                                                                                                                                                                                              ">>process.cc
        echo "    bool debug;                                                                                                                                                                                                                   ">>process.cc
        echo "                                                                                                                                                                                                                                  ">>process.cc
        echo "    // TChain that holds the input TTree's                                                                                                                                                                                        ">>process.cc
        echo "    TChain* events_tchain;                                                                                                                                                                                                        ">>process.cc
        echo "                                                                                                                                                                                                                                  ">>process.cc
        echo "    // Custom Looper object to facilitate looping over many files                                                                                                                                                                 ">>process.cc
        echo "    RooUtil::Looper<${MAKECLASSNAME}> looper;                                                                                                                                                                                     ">>process.cc
        echo "                                                                                                                                                                                                                                  ">>process.cc
        echo "    // Custom Cutflow framework                                                                                                                                                                                                   ">>process.cc
        echo "    RooUtil::Cutflow cutflow;                                                                                                                                                                                                     ">>process.cc
        echo "                                                                                                                                                                                                                                  ">>process.cc
        echo "    // Custom Histograms object compatible with RooUtil::Cutflow framework                                                                                                                                                        ">>process.cc
        echo "    RooUtil::Histograms histograms;                                                                                                                                                                                               ">>process.cc
        echo "                                                                                                                                                                                                                                  ">>process.cc
        echo "};                                                                                                                                                                                                                                ">>process.cc
        echo "                                                                                                                                                                                                                                  ">>process.cc
        echo "AnalysisConfig ana;                                                                                                                                                                                                               ">>process.cc
        echo "                                                                                                                                                                                                                                  ">>process.cc
        echo "// ./process INPUTFILEPATH OUTPUTFILE [NEVENTS]                                                                                                                                                                                   ">>process.cc
        echo "int main(int argc, char** argv)                                                                                                                                                                                                   ">>process.cc
        echo "{                                                                                                                                                                                                                                 ">>process.cc
        echo "                                                                                                                                                                                                                                  ">>process.cc
        echo "//********************************************************************************                                                                                                                                                ">>process.cc
        echo "//                                                                                                                                                                                                                                ">>process.cc
        echo "// 1. Parsing options                                                                                                                                                                                                             ">>process.cc
        echo "//                                                                                                                                                                                                                                ">>process.cc
        echo "//********************************************************************************                                                                                                                                                ">>process.cc
        echo "                                                                                                                                                                                                                                  ">>process.cc
        echo "    // cxxopts is just a tool to parse argc, and argv easily                                                                                                                                                                      ">>process.cc
        echo "                                                                                                                                                                                                                                  ">>process.cc
        echo "    // Grand option setting                                                                                                                                                                                                       ">>process.cc
        echo "    cxxopts::Options options(\"\n  $ doAnalysis\",  \"\n         **********************\n         *                    *\n         *       Looper       *\n         *                    *\n         **********************\n\"); ">>process.cc
        echo "                                                                                                                                                                                                                                  ">>process.cc
        echo "    // Read the options                                                                                                                                                                                                           ">>process.cc
        echo "    options.add_options()                                                                                                                                                                                                         ">>process.cc
        echo "        (\"i,input\"       , \"Comma separated input file list OR if just a directory is provided it will glob all in the directory BUT must end with '/' for the path\", cxxopts::value<std::string>())                          ">>process.cc
        echo "        (\"t,tree\"        , \"Name of the tree in the root file to open and loop over\"                                             , cxxopts::value<std::string>())                                                             ">>process.cc
        echo "        (\"o,output\"      , \"Output file name\"                                                                                    , cxxopts::value<std::string>())                                                             ">>process.cc
        echo "        (\"n,nevents\"     , \"N events to loop over\"                                                                               , cxxopts::value<int>()->default_value(\"-1\"))                                              ">>process.cc
        echo "        (\"j,nsplit_jobs\" , \"Enable splitting jobs by N blocks (--job_index must be set)\"                                         , cxxopts::value<int>())                                                                     ">>process.cc
        echo "        (\"I,job_index\"   , \"job_index of split jobs (--nsplit_jobs must be set. index starts from 0. i.e. 0, 1, 2, 3, etc...)\"   , cxxopts::value<int>())                                                                     ">>process.cc
        echo "        (\"d,debug\"       , \"Run debug job. i.e. overrides output option to 'debug.root' and 'recreate's the file.\")                                                                                                           ">>process.cc
        echo "        (\"h,help\"        , \"Print help\")                                                                                                                                                                                      ">>process.cc
        echo "        ;                                                                                                                                                                                                                         ">>process.cc
        echo "                                                                                                                                                                                                                                  ">>process.cc
        echo "    auto result = options.parse(argc, argv);                                                                                                                                                                                      ">>process.cc
        echo "                                                                                                                                                                                                                                  ">>process.cc
        echo "    // NOTE: When an option was provided (e.g. -i or --input), then the result.count(\"<option name>\") is more than 0                                                                                                            ">>process.cc
        echo "    // Therefore, the option can be parsed easily by asking the condition if (result.count(\"<option name>\");                                                                                                                    ">>process.cc
        echo "    // That's how the several options are parsed below                                                                                                                                                                            ">>process.cc
        echo "                                                                                                                                                                                                                                  ">>process.cc
        echo "    //_______________________________________________________________________________                                                                                                                                             ">>process.cc
        echo "    // --help                                                                                                                                                                                                                     ">>process.cc
        echo "    if (result.count(\"help\"))                                                                                                                                                                                                   ">>process.cc
        echo "    {                                                                                                                                                                                                                             ">>process.cc
        echo "        std::cout << options.help() << std::endl;                                                                                                                                                                                 ">>process.cc
        echo "        exit(1);                                                                                                                                                                                                                  ">>process.cc
        echo "    }                                                                                                                                                                                                                             ">>process.cc
        echo "                                                                                                                                                                                                                                  ">>process.cc
        echo "    //_______________________________________________________________________________                                                                                                                                             ">>process.cc
        echo "    // --input                                                                                                                                                                                                                    ">>process.cc
        echo "    if (result.count(\"input\"))                                                                                                                                                                                                  ">>process.cc
        echo "    {                                                                                                                                                                                                                             ">>process.cc
        echo "        ana.input_file_list_tstring = result[\"input\"].as<std::string>();                                                                                                                                                        ">>process.cc
        echo "    }                                                                                                                                                                                                                             ">>process.cc
        echo "    else                                                                                                                                                                                                                          ">>process.cc
        echo "    {                                                                                                                                                                                                                             ">>process.cc
        echo "        std::cout << options.help() << std::endl;                                                                                                                                                                                 ">>process.cc
        echo "        std::cout << \"ERROR: Input list is not provided! Check your arguments\" << std::endl;                                                                                                                                    ">>process.cc
        echo "        exit(1);                                                                                                                                                                                                                  ">>process.cc
        echo "    }                                                                                                                                                                                                                             ">>process.cc
        echo "                                                                                                                                                                                                                                  ">>process.cc
        echo "    //_______________________________________________________________________________                                                                                                                                             ">>process.cc
        echo "    // --tree                                                                                                                                                                                                                     ">>process.cc
        echo "    if (result.count(\"tree\"))                                                                                                                                                                                                   ">>process.cc
        echo "    {                                                                                                                                                                                                                             ">>process.cc
        echo "        ana.input_tree_name = result[\"tree\"].as<std::string>();                                                                                                                                                                 ">>process.cc
        echo "    }                                                                                                                                                                                                                             ">>process.cc
        echo "    else                                                                                                                                                                                                                          ">>process.cc
        echo "    {                                                                                                                                                                                                                             ">>process.cc
        echo "        std::cout << options.help() << std::endl;                                                                                                                                                                                 ">>process.cc
        echo "        std::cout << \"ERROR: Input tree name is not provided! Check your arguments\" << std::endl;                                                                                                                               ">>process.cc
        echo "        exit(1);                                                                                                                                                                                                                  ">>process.cc
        echo "    }                                                                                                                                                                                                                             ">>process.cc
        echo "                                                                                                                                                                                                                                  ">>process.cc
        echo "    //_______________________________________________________________________________                                                                                                                                             ">>process.cc
        echo "    // --debug                                                                                                                                                                                                                    ">>process.cc
        echo "    if (result.count(\"debug\"))                                                                                                                                                                                                  ">>process.cc
        echo "    {                                                                                                                                                                                                                             ">>process.cc
        echo "        ana.output_tfile = new TFile(\"debug.root\", \"recreate\");                                                                                                                                                               ">>process.cc
        echo "    }                                                                                                                                                                                                                             ">>process.cc
        echo "    else                                                                                                                                                                                                                          ">>process.cc
        echo "    {                                                                                                                                                                                                                             ">>process.cc
        echo "        //_______________________________________________________________________________                                                                                                                                         ">>process.cc
        echo "        // --output                                                                                                                                                                                                               ">>process.cc
        echo "        if (result.count(\"output\"))                                                                                                                                                                                             ">>process.cc
        echo "        {                                                                                                                                                                                                                         ">>process.cc
        echo "            ana.output_tfile = new TFile(result[\"output\"].as<std::string>().c_str(), \"create\");                                                                                                                               ">>process.cc
        echo "            if (not ana.output_tfile->IsOpen())                                                                                                                                                                                   ">>process.cc
        echo "            {                                                                                                                                                                                                                     ">>process.cc
        echo "                std::cout << options.help() << std::endl;                                                                                                                                                                         ">>process.cc
        echo "                std::cout << \"ERROR: output already exists! provide new output name or delete old file. OUTPUTFILE=\" << result[\"output\"].as<std::string>() << std::endl;                                                      ">>process.cc
        echo "                exit(1);                                                                                                                                                                                                          ">>process.cc
        echo "            }                                                                                                                                                                                                                     ">>process.cc
        echo "        }                                                                                                                                                                                                                         ">>process.cc
        echo "        else                                                                                                                                                                                                                      ">>process.cc
        echo "        {                                                                                                                                                                                                                         ">>process.cc
        echo "            std::cout << options.help() << std::endl;                                                                                                                                                                             ">>process.cc
        echo "            std::cout << \"ERROR: Output file name is not provided! Check your arguments\" << std::endl;                                                                                                                          ">>process.cc
        echo "            exit(1);                                                                                                                                                                                                              ">>process.cc
        echo "        }                                                                                                                                                                                                                         ">>process.cc
        echo "    }                                                                                                                                                                                                                             ">>process.cc
        echo "                                                                                                                                                                                                                                  ">>process.cc
        echo "    //_______________________________________________________________________________                                                                                                                                             ">>process.cc
        echo "    // --nevents                                                                                                                                                                                                                  ">>process.cc
        echo "    ana.n_events = result[\"nevents\"].as<int>();                                                                                                                                                                                 ">>process.cc
        echo "                                                                                                                                                                                                                                  ">>process.cc
        echo "    //_______________________________________________________________________________                                                                                                                                             ">>process.cc 
        echo "    // --nsplit_jobs                                                                                                                                                                                                              ">>process.cc 
        echo "    if (result.count(\"nsplit_jobs\"))                                                                                                                                                                                            ">>process.cc 
        echo "    {                                                                                                                                                                                                                             ">>process.cc 
        echo "        ana.nsplit_jobs = result[\"nsplit_jobs\"].as<int>();                                                                                                                                                                      ">>process.cc 
        echo "        if (ana.nsplit_jobs <= 0)                                                                                                                                                                                                 ">>process.cc 
        echo "        {                                                                                                                                                                                                                         ">>process.cc 
        echo "            std::cout << options.help() << std::endl;                                                                                                                                                                             ">>process.cc 
        echo "            std::cout << \"ERROR: option string --nsplit_jobs\" << ana.nsplit_jobs << \" has zero or negative value!\" << std::endl;                                                                                              ">>process.cc 
        echo "            std::cout << \"I am not sure what this means...\" << std::endl;                                                                                                                                                       ">>process.cc 
        echo "            exit(1);                                                                                                                                                                                                              ">>process.cc 
        echo "        }                                                                                                                                                                                                                         ">>process.cc 
        echo "    }                                                                                                                                                                                                                             ">>process.cc 
        echo "    else                                                                                                                                                                                                                          ">>process.cc 
        echo "    {                                                                                                                                                                                                                             ">>process.cc 
        echo "        ana.nsplit_jobs = -1;                                                                                                                                                                                                     ">>process.cc 
        echo "    }                                                                                                                                                                                                                             ">>process.cc 
        echo "                                                                                                                                                                                                                                  ">>process.cc 
        echo "    //_______________________________________________________________________________                                                                                                                                             ">>process.cc 
        echo "    // --nsplit_jobs                                                                                                                                                                                                              ">>process.cc 
        echo "    if (result.count(\"job_index\"))                                                                                                                                                                                              ">>process.cc 
        echo "    {                                                                                                                                                                                                                             ">>process.cc 
        echo "        ana.job_index = result[\"job_index\"].as<int>();                                                                                                                                                                          ">>process.cc 
        echo "        if (ana.job_index < 0)                                                                                                                                                                                                    ">>process.cc 
        echo "        {                                                                                                                                                                                                                         ">>process.cc 
        echo "            std::cout << options.help() << std::endl;                                                                                                                                                                             ">>process.cc 
        echo "            std::cout << \"ERROR: option string --job_index\" << ana.job_index << \" has negative value!\" << std::endl;                                                                                                          ">>process.cc 
        echo "            std::cout << \"I am not sure what this means...\" << std::endl;                                                                                                                                                       ">>process.cc 
        echo "            exit(1);                                                                                                                                                                                                              ">>process.cc 
        echo "        }                                                                                                                                                                                                                         ">>process.cc 
        echo "    }                                                                                                                                                                                                                             ">>process.cc 
        echo "    else                                                                                                                                                                                                                          ">>process.cc 
        echo "    {                                                                                                                                                                                                                             ">>process.cc 
        echo "        ana.job_index = -1;                                                                                                                                                                                                       ">>process.cc 
        echo "    }                                                                                                                                                                                                                             ">>process.cc 
        echo "                                                                                                                                                                                                                                  ">>process.cc 
        echo "                                                                                                                                                                                                                                  ">>process.cc 
        echo "    // Sanity check for split jobs (if one is set the other must be set too)                                                                                                                                                      ">>process.cc 
        echo "    if (result.count(\"job_index\") or result.count(\"nsplit_jobs\"))                                                                                                                                                             ">>process.cc 
        echo "    {                                                                                                                                                                                                                             ">>process.cc 
        echo "        // If one is not provided then throw error                                                                                                                                                                                ">>process.cc 
        echo "        if ( not (result.count(\"job_index\") and result.count(\"nsplit_jobs\")))                                                                                                                                                 ">>process.cc 
        echo "        {                                                                                                                                                                                                                         ">>process.cc 
        echo "            std::cout << options.help() << std::endl;                                                                                                                                                                             ">>process.cc 
        echo "            std::cout << \"ERROR: option string --job_index and --nsplit_jobs must be set at the same time!\" << std::endl;                                                                                                       ">>process.cc 
        echo "            exit(1);                                                                                                                                                                                                              ">>process.cc 
        echo "        }                                                                                                                                                                                                                         ">>process.cc 
        echo "        // If it is set then check for sanity                                                                                                                                                                                     ">>process.cc 
        echo "        else                                                                                                                                                                                                                      ">>process.cc 
        echo "        {                                                                                                                                                                                                                         ">>process.cc 
        echo "            if (ana.job_index >= ana.nsplit_jobs)                                                                                                                                                                                 ">>process.cc 
        echo "            {                                                                                                                                                                                                                     ">>process.cc 
        echo "                std::cout << options.help() << std::endl;                                                                                                                                                                         ">>process.cc 
        echo "                std::cout << \"ERROR: --job_index >= --nsplit_jobs ! This does not make sense...\" << std::endl;                                                                                                                  ">>process.cc 
        echo "                exit(1);                                                                                                                                                                                                          ">>process.cc 
        echo "            }                                                                                                                                                                                                                     ">>process.cc 
        echo "        }                                                                                                                                                                                                                         ">>process.cc 
        echo "    }                                                                                                                                                                                                                             ">>process.cc 
        echo "                                                                                                                                                                                                                                  ">>process.cc
        echo "    //                                                                                                                                                                                                                            ">>process.cc
        echo "    // Printing out the option settings overview                                                                                                                                                                                  ">>process.cc
        echo "    //                                                                                                                                                                                                                            ">>process.cc
        echo "    std::cout <<  \"=========================================================\" << std::endl;                                                                                                                                     ">>process.cc
        echo "    std::cout <<  \" Setting of the analysis job based on provided arguments \" << std::endl;                                                                                                                                     ">>process.cc
        echo "    std::cout <<  \"---------------------------------------------------------\" << std::endl;                                                                                                                                     ">>process.cc
        echo "    std::cout <<  \" ana.input_file_list_tstring: \" << ana.input_file_list_tstring <<  std::endl;                                                                                                                                ">>process.cc
        echo "    std::cout <<  \" ana.output_tfile: \" << ana.output_tfile->GetName() <<  std::endl;                                                                                                                                           ">>process.cc
        echo "    std::cout <<  \" ana.n_events: \" << ana.n_events <<  std::endl;                                                                                                                                                              ">>process.cc
        echo "    std::cout <<  \" ana.nsplit_jobs: \" << ana.nsplit_jobs <<  std::endl;                                                                                                                                                        ">>process.cc
        echo "    std::cout <<  \" ana.job_index: \" << ana.job_index <<  std::endl;                                                                                                                                                            ">>process.cc
        echo "    std::cout <<  \"=========================================================\" << std::endl;                                                                                                                                     ">>process.cc
        echo "                                                                                                                                                                                                                                  ">>process.cc
        echo "//********************************************************************************                                                                                                                                                ">>process.cc
        echo "//                                                                                                                                                                                                                                ">>process.cc
        echo "// 2. Opening input baby files                                                                                                                                                                                                    ">>process.cc
        echo "//                                                                                                                                                                                                                                ">>process.cc
        echo "//********************************************************************************                                                                                                                                                ">>process.cc
        echo "                                                                                                                                                                                                                                  ">>process.cc
        echo "    // Create the TChain that holds the TTree's of the baby ntuples                                                                                                                                                               ">>process.cc
        echo "    ana.events_tchain = RooUtil::FileUtil::createTChain(ana.input_tree_name, ana.input_file_list_tstring);                                                                                                                        ">>process.cc
        echo "                                                                                                                                                                                                                                  ">>process.cc
        echo "    // Create a Looper object to loop over input files                                                                                                                                                                            ">>process.cc
        echo "    // the \"www\" object is defined in the wwwtree.h/cc                                                                                                                                                                          ">>process.cc
        echo "    // This is an instance which helps read variables in the WWW baby TTree                                                                                                                                                       ">>process.cc
        echo "    // It is a giant wrapper that facilitates reading TBranch values.                                                                                                                                                             ">>process.cc
        echo "    // e.g. if there is a TBranch named \"lep_pt\" which is a std::vector<float> then, one can access the branch via                                                                                                              ">>process.cc
        echo "    //                                                                                                                                                                                                                            ">>process.cc
        echo "    //    std::vector<float> lep_pt = www.lep_pt();                                                                                                                                                                               ">>process.cc
        echo "    //                                                                                                                                                                                                                            ">>process.cc
        echo "    // and no need for \"SetBranchAddress\" and declaring variable shenanigans necessary                                                                                                                                          ">>process.cc
        echo "    // This is a standard thing SNT does pretty much every looper we use                                                                                                                                                          ">>process.cc
        echo "    ana.looper.init(ana.events_tchain, &${TREEINSTANCENAME}, ana.n_events);                                                                                                                                                       ">>process.cc
        echo "                                                                                                                                                                                                                                  ">>process.cc
        echo "//********************************************************************************                                                                                                                                                ">>process.cc
        echo "//                                                                                                                                                                                                                                ">>process.cc
        echo "// Interlude... notes on RooUtil framework                                                                                                                                                                                        ">>process.cc
        echo "//                                                                                                                                                                                                                                ">>process.cc
        echo "//********************************************************************************                                                                                                                                                ">>process.cc
        echo "                                                                                                                                                                                                                                  ">>process.cc
        echo "    // ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=                                                                                                                                                               ">>process.cc
        echo "    // Quick tutorial on RooUtil::Cutflow object cut tree formation                                                                                                                                                               ">>process.cc
        echo "    // ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=                                                                                                                                                               ">>process.cc
        echo "    //                                                                                                                                                                                                                            ">>process.cc
        echo "    // The RooUtil::Cutflow object facilitates creating a tree structure of cuts                                                                                                                                                  ">>process.cc
        echo "    //                                                                                                                                                                                                                            ">>process.cc
        echo "    // To add cuts to each node of the tree with cuts defined, use \"addCut\" or \"addCutToLastActiveCut\"                                                                                                                        ">>process.cc
        echo "    // The \"addCut\" or \"addCutToLastActiveCut\" accepts three argument, <name>, and two lambda's that define the cut selection, and the weight to apply to that cut stage                                                      ">>process.cc
        echo "    //                                                                                                                                                                                                                            ">>process.cc
        echo "    // e.g. To create following cut-tree structure one does                                                                                                                                                                       ">>process.cc
        echo "    //                                                                                                                                                                                                                            ">>process.cc
        echo "    //             (Root) <--- Always exists as soon as RooUtil::Cutflow object is created. But this is basically hidden underneath and users do not have to care                                                                 ">>process.cc
        echo "    //                |                                                                                                                                                                                                           ">>process.cc
        echo "    //            CutWeight                                                                                                                                                                                                       ">>process.cc
        echo "    //            |       |                                                                                                                                                                                                       ">>process.cc
        echo "    //     CutPreSel1    CutPreSel2                                                                                                                                                                                               ">>process.cc
        echo "    //       |                  |                                                                                                                                                                                                 ">>process.cc
        echo "    //     CutSel1           CutSel2                                                                                                                                                                                              ">>process.cc
        echo "    //                                                                                                                                                                                                                            ">>process.cc
        echo "    //                                                                                                                                                                                                                            ">>process.cc
        echo "    //   code:                                                                                                                                                                                                                    ">>process.cc
        echo "    //                                                                                                                                                                                                                            ">>process.cc
        echo "    //      // Create the object (Root node is created as soon as the instance is created)                                                                                                                                        ">>process.cc
        echo "    //      RooUtil::Cutflow cutflow;                                                                                                                                                                                             ">>process.cc
        echo "    //                                                                                                                                                                                                                            ">>process.cc
        echo "    //      cutflow.addCut(\"CutWeight\"                 , <lambda> , <lambda>); // CutWeight is added below \"Root\"-node                                                                                                        ">>process.cc
        echo "    //      cutflow.addCutToLastActiveCut(\"CutPresel1\" , <lambda> , <lambda>); // The last \"active\" cut is \"CutWeight\" since I just added that. So \"CutPresel1\" is added below \"CutWeight\"                              ">>process.cc
        echo "    //      cutflow.addCutToLastActiveCut(\"CutSel1\"    , <lambda> , <lambda>); // The last \"active\" cut is \"CutPresel1\" since I just added that. So \"CutSel1\" is added below \"CutPresel1\"                               ">>process.cc
        echo "    //                                                                                                                                                                                                                            ">>process.cc
        echo "    //      cutflow.getCut(\"CutWeight\"); // By \"getting\" the cut object, this makes the \"CutWeight\" the last \"active\" cut.                                                                                                ">>process.cc
        echo "    //      cutflow.addCutToLastActiveCut(\"CutPresel2\" , <lambda> , <lambda>); // The last \"active\" cut is \"CutWeight\" since I \"getCut\" on it. So \"CutPresel2\" is added below \"CutWeight\"                             ">>process.cc
        echo "    //      cutflow.addCutToLastActiveCut(\"CutSel2\"    , <lambda> , <lambda>); // The last \"active\" cut is \"CutPresel2\" since I just added that. So \"CutSel2\" is added below \"CutPresel1\"                               ">>process.cc
        echo "    //                                                                                                                                                                                                                            ">>process.cc
        echo "    // (Side note: \"UNITY\" lambda is defined in the framework to just return 1. This so that use don't have to type [&]() {return 1;} so many times.)                                                                           ">>process.cc
        echo "    //                                                                                                                                                                                                                            ">>process.cc
        echo "    // Once the cutflow is formed, create cutflow histograms can be created by calling RooUtil::Cutflow::bookCutflows())                                                                                                          ">>process.cc
        echo "    // This function looks through the terminating nodes of the tree structured cut tree. and creates a histogram that will fill the yields                                                                                       ">>process.cc
        echo "    // For the example above, there are two terminationg nodes, \"CutSel1\", and \"CutSel2\"                                                                                                                                      ">>process.cc
        echo "    // So in this case Root::Cutflow::bookCutflows() will create two histograms. (Actually four histograms.)                                                                                                                      ">>process.cc
        echo "    //                                                                                                                                                                                                                            ">>process.cc
        echo "    //  - TH1F* type object :  CutSel1_cutflow (4 bins, with first bin labeled \"Root\", second bin labeled \"CutWeight\", third bin labeled \"CutPreSel1\", fourth bin labeled \"CutSel1\")                                      ">>process.cc
        echo "    //  - TH1F* type object :  CutSel2_cutflow (...)                                                                                                                                                                              ">>process.cc
        echo "    //  - TH1F* type object :  CutSel1_rawcutflow (...)                                                                                                                                                                           ">>process.cc
        echo "    //  - TH1F* type object :  CutSel2_rawcutflow (...)                                                                                                                                                                           ">>process.cc
        echo "    //                                ^                                                                                                                                                                                           ">>process.cc
        echo "    //                                |                                                                                                                                                                                           ">>process.cc
        echo "    // NOTE: There is only one underscore \"_\" between <CutName>_cutflow or <CutName>_rawcutflow                                                                                                                                 ">>process.cc
        echo "    //                                                                                                                                                                                                                            ">>process.cc
        echo "    // And later in the loop when RooUtil::Cutflow::fill() function is called, the tree structure will be traversed through and the appropriate yields will be filled into the histograms                                         ">>process.cc
        echo "    //                                                                                                                                                                                                                            ">>process.cc
        echo "    // After running the loop check for the histograms in the output root file                                                                                                                                                    ">>process.cc
        echo "                                                                                                                                                                                                                                  ">>process.cc
        echo "    // ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=                                                                                                                                                                               ">>process.cc
        echo "    // Quick tutorial on RooUtil::Histograms object                                                                                                                                                                               ">>process.cc
        echo "    // ~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=~=                                                                                                                                                                               ">>process.cc
        echo "    //                                                                                                                                                                                                                            ">>process.cc
        echo "    // The RooUtil::Histograms object facilitates book keeping histogram definitions                                                                                                                                              ">>process.cc
        echo "    // And in conjunction with RooUtil::Cutflow object, one can book same histograms across different cut stages easily without copy pasting codes many times by hand.                                                            ">>process.cc
        echo "    //                                                                                                                                                                                                                            ">>process.cc
        echo "    // The histogram addition happens in two steps.                                                                                                                                                                               ">>process.cc
        echo "    // 1. Defining histograms                                                                                                                                                                                                     ">>process.cc
        echo "    // 2. Booking histograms to cuts                                                                                                                                                                                              ">>process.cc
        echo "    //                                                                                                                                                                                                                            ">>process.cc
        echo "    // Histograms are defined via following functions                                                                                                                                                                             ">>process.cc
        echo "    //                                                                                                                                                                                                                            ">>process.cc
        echo "    //      RooUtil::Histograms::addHistogram       : Typical 1D histogram (TH1F*) \"Fill()\" called once per event                                                                                                               ">>process.cc
        echo "    //      RooUtil::Histograms::addVecHistogram    : Typical 1D histogram (TH1F*) \"Fill()\" called multiple times per event                                                                                                     ">>process.cc
        echo "    //      RooUtil::Histograms::add2DHistogram     : Typical 2D histogram (TH2F*) \"Fill()\" called once per event                                                                                                               ">>process.cc
        echo "    //      RooUtil::Histograms::add2DVecHistogram  : Typical 2D histogram (TH2F*) \"Fill()\" called multiple times per event                                                                                                     ">>process.cc
        echo "    // e.g.                                                                                                                                                                                                                       ">>process.cc
        echo "    //                                                                                                                                                                                                                            ">>process.cc
        echo "    //    RooUtil::Histograms histograms;                                                                                                                                                                                         ">>process.cc
        echo "    //    histograms.addHistogram   (\"MllSS\"    , 180 , 0. , 300. , [&]() { return www.MllSS()  ; }); // The lambda returns float                                                                                               ">>process.cc
        echo "    //    histograms.addVecHistogram(\"AllLepPt\" , 180 , 0. , 300. , [&]() { return www.lep_pt() ; }); // The lambda returns vector<float>                                                                                       ">>process.cc
        echo "    //                                                                                                                                                                                                                            ">>process.cc
        echo "    // The addVecHistogram will have lambda to return vector<float> and it will loop over the values and call TH1F::Fill() for each item                                                                                          ">>process.cc
        echo "    //                                                                                                                                                                                                                            ">>process.cc
        echo "    // To book histograms to cuts one uses                                                                                                                                                                                        ">>process.cc
        echo "    //                                                                                                                                                                                                                            ">>process.cc
        echo "    //      RooUtil::Cutflow::bookHistogramsForCut()                                                                                                                                                                              ">>process.cc
        echo "    //      RooUtil::Cutflow::bookHistogramsForCutAndBelow()                                                                                                                                                                      ">>process.cc
        echo "    //      RooUtil::Cutflow::bookHistogramsForCutAndAbove()                                                                                                                                                                      ">>process.cc
        echo "    //      RooUtil::Cutflow::bookHistogramsForEndCuts()                                                                                                                                                                          ">>process.cc
        echo "    //                                                                                                                                                                                                                            ">>process.cc
        echo "    // e.g. Given a tree like the following, we can book histograms to various cuts as we want                                                                                                                                    ">>process.cc
        echo "    //                                                                                                                                                                                                                            ">>process.cc
        echo "    //              Root                                                                                                                                                                                                          ">>process.cc
        echo "    //                |                                                                                                                                                                                                           ">>process.cc
        echo "    //            CutWeight                                                                                                                                                                                                       ">>process.cc
        echo "    //            |       |                                                                                                                                                                                                       ">>process.cc
        echo "    //     CutPreSel1    CutPreSel2                                                                                                                                                                                               ">>process.cc
        echo "    //       |                  |                                                                                                                                                                                                 ">>process.cc
        echo "    //     CutSel1           CutSel2                                                                                                                                                                                              ">>process.cc
        echo "    //                                                                                                                                                                                                                            ">>process.cc
        echo "    // For example,                                                                                                                                                                                                               ">>process.cc
        echo "    //                                                                                                                                                                                                                            ">>process.cc
        echo "    //    1. book a set of histograms to one cut:                                                                                                                                                                                 ">>process.cc
        echo "    //                                                                                                                                                                                                                            ">>process.cc
        echo "    //       cutflow.bookHistogramsForCut(histograms, \"CutPreSel2\")                                                                                                                                                             ">>process.cc
        echo "    //                                                                                                                                                                                                                            ">>process.cc
        echo "    //    2. book a set of histograms to a cut and below                                                                                                                                                                          ">>process.cc
        echo "    //                                                                                                                                                                                                                            ">>process.cc
        echo "    //       cutflow.bookHistogramsForCutAndBelow(histograms, \"CutWeight\") // will book a set of histograms to CutWeight, CutPreSel1, CutPreSel2, CutSel1, and CutSel2                                                          ">>process.cc
        echo "    //                                                                                                                                                                                                                            ">>process.cc
        echo "    //    3. book a set of histograms to a cut and above (... useless...?)                                                                                                                                                        ">>process.cc
        echo "    //                                                                                                                                                                                                                            ">>process.cc
        echo "    //       cutflow.bookHistogramsForCutAndAbove(histograms, \"CutPreSel2\") // will book a set of histograms to CutPreSel2, CutWeight (nothing happens to Root node)                                                            ">>process.cc
        echo "    //                                                                                                                                                                                                                            ">>process.cc
        echo "    //    4. book a set of histograms to a terminating nodes                                                                                                                                                                      ">>process.cc
        echo "    //                                                                                                                                                                                                                            ">>process.cc
        echo "    //       cutflow.bookHistogramsForEndCuts(histograms) // will book a set of histograms to CutSel1 and CutSel2                                                                                                                 ">>process.cc
        echo "    //                                                                                                                                                                                                                            ">>process.cc
        echo "    // The naming convention of the booked histograms are as follows                                                                                                                                                              ">>process.cc
        echo "    //                                                                                                                                                                                                                            ">>process.cc
        echo "    //   cutflow.bookHistogramsForCut(histograms, \"CutSel1\");                                                                                                                                                                   ">>process.cc
        echo "    //                                                                                                                                                                                                                            ">>process.cc
        echo "    //  - TH1F* type object : CutSel1__MllSS;                                                                                                                                                                                     ">>process.cc
        echo "    //  - TH1F* type object : CutSel1__AllLepPt;                                                                                                                                                                                  ">>process.cc
        echo "    //                               ^^                                                                                                                                                                                           ">>process.cc
        echo "    //                               ||                                                                                                                                                                                           ">>process.cc
        echo "    // NOTE: There are two underscores \"__\" between <CutName>__<HistogramName>                                                                                                                                                  ">>process.cc
        echo "    //                                                                                                                                                                                                                            ">>process.cc
        echo "    // And later in the loop when RooUtil::CutName::fill() function is called, the tree structure will be traversed through and the appropriate histograms will be filled with appropriate variables                              ">>process.cc
        echo "    // After running the loop check for the histograms in the output root file                                                                                                                                                    ">>process.cc
        echo "                                                                                                                                                                                                                                  ">>process.cc
        echo "    // Set the cutflow object output file                                                                                                                                                                                         ">>process.cc
        echo "    ana.cutflow.setTFile(ana.output_tfile);                                                                                                                                                                                       ">>process.cc
        echo "                                                                                                                                                                                                                                  ">>process.cc
        echo "    ana.cutflow.addCut(\"DiElChannel\", [&]() { return 1/*set your cut here*/; }, [&]() { return 1/*set your weight here*/; } );                                                                                                  ">>process.cc
        echo "    ana.cutflow.addCut(\"DiMuChannel\", [&]() { return 1/*set your cut here*/; }, [&]() { return 1/*set your weight here*/; } );                                                                                                  ">>process.cc
        echo "                                                                                                                                                                                                                                  ">>process.cc
        echo "    ana.cutflow.getCut(\"DiElChannel\");                                                                                                                                                                                          ">>process.cc
        echo "    ana.cutflow.addCutToLastActiveCut(\"DiElChannelCutA\", [&]() { return 1/*set your cut here*/; }, [&]() { return 1/*set your weight here*/; } );                                                                               ">>process.cc
        echo "    ana.cutflow.addCutToLastActiveCut(\"DiElChannelCutB\", [&]() { return 1/*set your cut here*/; }, [&]() { return 1/*set your weight here*/; } );                                                                               ">>process.cc
        echo "    ana.cutflow.addCutToLastActiveCut(\"DiElChannelCutC\", [&]() { return 1/*set your cut here*/; }, [&]() { return 1/*set your weight here*/; } );                                                                               ">>process.cc
        echo "                                                                                                                                                                                                                                  ">>process.cc
        echo "    ana.cutflow.getCut(\"DiMuChannel\");                                                                                                                                                                                          ">>process.cc
        echo "    ana.cutflow.addCutToLastActiveCut(\"DiMuChannelCutA\", [&]() { return 1/*set your cut here*/; }, [&]() { return 1/*set your weight here*/; } );                                                                               ">>process.cc
        echo "    ana.cutflow.addCutToLastActiveCut(\"DiMuChannelCutB\", [&]() { return 1/*set your cut here*/; }, [&]() { return 1/*set your weight here*/; } );                                                                               ">>process.cc
        echo "    ana.cutflow.addCutToLastActiveCut(\"DiMuChannelCutC\", [&]() { return 1/*set your cut here*/; }, [&]() { return 1/*set your weight here*/; } );                                                                               ">>process.cc
        echo "                                                                                                                                                                                                                                  ">>process.cc
        echo "    // Print cut structure                                                                                                                                                                                                        ">>process.cc
        echo "    ana.cutflow.printCuts();                                                                                                                                                                                                      ">>process.cc
        echo "                                                                                                                                                                                                                                  ">>process.cc
        echo "    // Histogram utility object that is used to define the histograms                                                                                                                                                             ">>process.cc
        echo "    ana.histograms.addHistogram(\"Mll\", 180, 0, 250, [&]() { return 1/* set your variable here*/; } );                                                                                                                           ">>process.cc
        echo "                                                                                                                                                                                                                                  ">>process.cc
        echo "    // Book cutflows                                                                                                                                                                                                              ">>process.cc
        echo "    ana.cutflow.bookCutflows();                                                                                                                                                                                                   ">>process.cc
        echo "                                                                                                                                                                                                                                  ">>process.cc
        echo "    // Book Histograms                                                                                                                                                                                                            ">>process.cc
        echo "    ana.cutflow.bookHistogramsForCutAndBelow(ana.histograms, \"DiElChannel\");                                                                                                                                                    ">>process.cc
        echo "    ana.cutflow.bookHistogramsForCutAndBelow(ana.histograms, \"DiMuChannel\");                                                                                                                                                    ">>process.cc
        echo "    // ana.cutflow.bookHistograms(ana.histograms); // if just want to book everywhere                                                                                                                                             ">>process.cc
        echo "                                                                                                                                                                                                                                  ">>process.cc
        echo "    // Looping input file                                                                                                                                                                                                         ">>process.cc
        echo "    while (ana.looper.nextEvent())                                                                                                                                                                                                ">>process.cc
        echo "    {                                                                                                                                                                                                                             ">>process.cc
        echo "                                                                                                                                                                                                                                  ">>process.cc
        echo "        // If splitting jobs are requested then determine whether to process the event or not based on remainder                                                                                                                  ">>process.cc 
        echo "        if (result.count(\"job_index\") and result.count(\"nsplit_jobs\"))                                                                                                                                                        ">>process.cc 
        echo "        {                                                                                                                                                                                                                         ">>process.cc 
        echo "            if (ana.looper.getNEventsProcessed() % ana.nsplit_jobs != (unsigned int) ana.job_index)                                                                                                                               ">>process.cc 
        echo "                continue;                                                                                                                                                                                                         ">>process.cc 
        echo "        }                                                                                                                                                                                                                         ">>process.cc 
        echo "                                                                                                                                                                                                                                  ">>process.cc
        echo "        //Do what you need to do in for each event here                                                                                                                                                                           ">>process.cc
        echo "        //To save use the following function                                                                                                                                                                                      ">>process.cc
        echo "        ana.cutflow.fill();                                                                                                                                                                                                       ">>process.cc
        echo "    }                                                                                                                                                                                                                             ">>process.cc
        echo "                                                                                                                                                                                                                                  ">>process.cc
        echo "    // Writing output file                                                                                                                                                                                                        ">>process.cc
        echo "    ana.cutflow.saveOutput();                                                                                                                                                                                                     ">>process.cc
        echo "                                                                                                                                                                                                                                  ">>process.cc
        echo "    // The below can be sometimes crucial                                                                                                                                                                                         ">>process.cc
        echo "    delete ana.output_tfile;                                                                                                                                                                                                      ">>process.cc
        echo "}                                                                                                                                                                                                                                 ">>process.cc

    fi

    #
    # Create Makefile
    #
    if [ -e Makefile ]; then
        e_error "RooUtil:: Makefile already exists. We will leave it alone. Erase it if you want to override"
    else
        echo '# Simple makefile'                                                                                                                            >  Makefile
        echo ''                                                                                                                                             >> Makefile
        echo 'EXE=doAnalysis'                                                                                                                               >> Makefile
        echo ''                                                                                                                                             >> Makefile
        echo 'SOURCES=$(wildcard *.cc)'                                                                                                                     >> Makefile
        echo 'OBJECTS=$(SOURCES:.cc=.o)'                                                                                                                    >> Makefile
        echo 'HEADERS=$(SOURCES:.cc=.h)'                                                                                                                    >> Makefile
        echo ''                                                                                                                                             >> Makefile
        echo 'CC          = g++'                                                                                                                            >> Makefile
        echo 'CXX         = g++'                                                                                                                            >> Makefile
        echo 'CXXFLAGS    = -g -O2 -Wall -fPIC -Wshadow -Woverloaded-virtual'                                                                               >> Makefile
        echo 'LD          = g++'                                                                                                                            >> Makefile
        echo 'LDFLAGS     = -g -O2'                                                                                                                         >> Makefile
        echo 'SOFLAGS     = -g -shared'                                                                                                                     >> Makefile
        echo 'CXXFLAGS    = -g -O2 -Wall -fPIC -Wshadow -Woverloaded-virtual'                                                                               >> Makefile
        echo 'LDFLAGS     = -g -O2'                                                                                                                         >> Makefile
        echo 'ROOTLIBS    = $(shell root-config --libs)'                                                                                                    >> Makefile
        echo 'ROOTCFLAGS  = $(shell root-config --cflags)'                                                                                                  >> Makefile
        echo 'CXXFLAGS   += $(ROOTCFLAGS)'                                                                                                                  >> Makefile
        echo 'CFLAGS      = $(ROOTCFLAGS) -Wall -Wno-unused-function -g -O2 -fPIC -fno-var-tracking'                                                        >> Makefile
        echo 'EXTRACFLAGS = $(shell rooutil-config)'                                                                                                        >> Makefile
        echo 'EXTRAFLAGS  = -fPIC -ITMultiDrawTreePlayer -Wunused-variable -lTMVA -lEG -lGenVector -lXMLIO -lMLP -lTreePlayer'                              >> Makefile
        echo ''                                                                                                                                             >> Makefile
        echo '$(EXE): $(OBJECTS) '${MAKECLASSNAME}'.o'                                                                                                      >> Makefile
        echo '	$(LD) $(CXXFLAGS) $(LDFLAGS) $(OBJECTS) $(ROOTLIBS) $(EXTRAFLAGS) -o $@'                                                                    >> Makefile
        echo ''                                                                                                                                             >> Makefile
        echo '%.o: %.cc'                                                                                                                                    >> Makefile
        echo '	$(CC) $(CFLAGS) $(EXTRACFLAGS) $< -c -o $@'                                                                                                 >> Makefile
        echo ''                                                                                                                                             >> Makefile
        echo 'clean:'                                                                                                                                       >> Makefile
        echo '	rm -f $(OBJECTS) $(EXE)'                                                                                                                    >> Makefile
    fi

    #echo "	sh rooutil/makeclass.sh -f -x TEMPLATE_TREE_PATH ${TTREENAME} ${MAKECLASSNAME} ${NAMESPACENAME} ${TREEINSTANCENAME}  > /dev/null 2>&1"  >> Makefile

    e_arrow "RooUtil:: "
    e_arrow "RooUtil:: Contact Philip Chang <philip@ucsd.edu> for any questions."
    e_arrow "RooUtil:: "
    e_arrow "RooUtil:: Happy Coding!"
    e_arrow "RooUtil:: "
    e_arrow "RooUtil:: Compile via 'make'"

else

    e_arrow "RooUtil:: "
    e_arrow "RooUtil:: "
    e_arrow "RooUtil:: "
    e_arrow "RooUtil:: "
    e_arrow "RooUtil:: To use these classes, one includes the source files to your code and get a TTree object from one of the root file and do the following:"
    e_arrow "RooUtil:: "
    e_arrow "RooUtil::  ... "
    e_arrow "RooUtil::  ... "
    e_arrow "RooUtil::  TFile* file = new TFile(\"/path/to/my/rootfile.root\");"
    e_arrow "RooUtil::  TTree* tree = (TTree*) file->Get(\"${TREENAME}\");"
    e_arrow "RooUtil::  ${TREEINSTANCENAME}.Init(tree);"
    e_arrow "RooUtil::  ... "
    e_arrow "RooUtil::  ... "
    e_arrow "RooUtil:: "
    e_arrow "RooUtil:: Contact Philip Chang <philip@ucsd.edu> for any questions."
    e_arrow "RooUtil:: "
    e_arrow "RooUtil:: Happy Coding!"

fi
#eof

