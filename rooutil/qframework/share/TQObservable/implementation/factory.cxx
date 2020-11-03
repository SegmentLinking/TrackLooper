//______________________________________________________________________________________________

// the following preprocessor macro defines an "observable factory"
// that is supposed to create instances of your class based on input
// expressions.

// it should receive an expression as an input and decide whether this
// expression should be used to construct an instance of your class,
// in which case it should take care of this, or return NULL.

// in addition to defining your observable factory here, you need to
// register it with the TQObservable manager. This can either be done
// from C++ code, using the line
//   TQObservable::manager.registerFactory(ObsName::getFactory(),true);
// or from python code, using the line
//   TQObservable.manager.registerFactory(ObsName.getFactory(),True)
// Either of these lines need to be put in a location where they are
// executed before observables are retrieved. You might want to 'grep'
// for 'registerFactory' in the package you are adding your observable
// to in order to get some ideas on where to put these!

DEFINE_OBSERVABLE_FACTORY(ObsName,TString expr){
  // try to create an instance of this observable from the given expression
  // return the newly created observable upon success
  // or NULL upon failure

  // first, check if the expression fits your observable type
  // for example, you can grab all expressions that begin wth "ObsName:"
  if(TQStringUtils::removeLeadingText(expr,"ObsName:")){
    // if this is the case, then we call the expression-constructor
    return new ObsName(expr);
  }
  // else, that is, if the expression doesn't match the pattern we
  // expect, we return this is important, because only in doing so we
  // give other observable types the chance to check for themselves
  return NULL;
  // if you do not return NULL here, your observable will become the
  // default observable type for all expressions that don't match
  // anything else, which is probably not what you want...
}

