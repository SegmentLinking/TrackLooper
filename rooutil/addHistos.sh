# Originally written by Samuel May <sjmay@ucsd.edu>
# Modified (slightly) by Philip Chang <philip@ucsd.edu>

### Function to add large numbers (>100) of histograms using hadd 		###
### hadd seems to have a limit on the number of histograms you can give it, 	###
### this adds histos in groups of 100 so you can sleep easy knowing that hadd 	###
### hasn't skipped over any of your well-deserved histograms 			###
### Arg1: name of output histogram (e.g. "histosMaster") 			###
### Arg2: name of input histogram (e.g. "histosToAdd_4.root" would be input 	###
### 	  as "histosToAdd"							###
### Arg3: (OPTIONAL) number of cores to run on (default 1) 			###
function addHistos 
{
  if (( $# < 3 ))
  then
    nPar=1
  else
    nPar=$3
  fi

  NFILES=$(ls ${2}*.root | wc -l)
  echo $NFILES

  histosToAdd=""
  bigHistos=""
  idx1=1
  idx2=1
  while (($idx1 <= ${NFILES}))
  do
    for ((i=1; i<=100; i++))
    do
      if (($idx1 <= ${NFILES}))
      then
        if [ -e $2"_"$idx1".root" ]; then 
          histosToAdd=$histosToAdd" "$2"_"$idx1".root"
        fi
        let "idx1++"
      fi
    done
    hadd -f -k $1"_"$idx2".root" $histosToAdd &
    if (($idx2 % $nPar == 0))
    then
      wait
    fi
    bigHistos=$bigHistos" "$1"_"$idx2".root"
    histosToAdd=""
    let "idx2++"
  done
  wait
  hadd -f -k $1".root" $bigHistos
}

addHistos $*
