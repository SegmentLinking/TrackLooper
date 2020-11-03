DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
export PYTHONPATH=$DIR/lib:$PYTHONPATH
export LD_LIBRARY_PATH=$DIR/lib:$LD_LIBRARY_PATH
export LIBXMLPATH=/cvmfs/cms.cern.ch/slc6_amd64_gcc530/external/libxml2/2.9.1-oenich/lib/libxml2.so
export TQPATH=$PWD
