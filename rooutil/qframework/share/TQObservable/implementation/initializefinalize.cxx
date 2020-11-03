//______________________________________________________________________________________________

bool ObsName::initializeSelf(){
  DEBUGclass("initializing");
  // initialize this observable
  return true;
}

//______________________________________________________________________________________________

bool ObsName::finalizeSelf(){
  // initialize this observable
  DEBUGclass("finalizing");
  // remember to undo anything you did in initializeSelf() !
  return true;
}
