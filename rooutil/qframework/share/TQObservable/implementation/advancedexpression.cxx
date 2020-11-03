//______________________________________________________________________________________________

bool ObsName::parseExpression(const TString& expr){
  // parse the expression
  return true;
}

//______________________________________________________________________________________________

void ObsName::clearParsedExpression(){
  // clear the current expression
}

//______________________________________________________________________________________________

TString ObsName::getActiveExpression() const {
  // retrieve the expression associated with this incarnation
  return /* you have to build the expression here */;
}

//______________________________________________________________________________________________

bool ObsName::initializeSelf(){
  // initialize self - compile container name, construct accessor
  if(!this->parseExpression(TQObservable::compileExpression(this->fExpression,this->fSample))){
    return false;
  }
  return true;
}
 
//______________________________________________________________________________________________

bool ObsName::finalizeSelf(){
  // finalize self - delete accessor
  this->clearParsedExpression();
  return true;
}
