#include "PkgName/ObsName.h"
#include <limits>

// uncomment the following line to enable debug printouts
// #define _DEBUG_
// you can perform debug printouts with statements like this
// DEBUG("error number %d occurred",someInteger);

// be careful to not move the _DEBUG_ flag behind the following line
// otherwise, it will show no effect
#include "QFramework/TQLibrary.h"

ClassImp(ObsName)

//______________________________________________________________________________________________

ObsName::ObsName(){
  // default constructor
  DEBUGclass("default constructor called");
}

//______________________________________________________________________________________________

ObsName::~ObsName(){
  // default destructor
  DEBUGclass("destructor called");
} 

