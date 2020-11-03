//______________________________________________________________________________________________

double ObsName::getValue() const {
  // value retrieval function, called on every event for every cut and histogram
  DEBUGclass("entering function");

  // the contents of this function highly depend on the way your
  // observable is supposed to retrieve (or create?) data
  // good luck, you're on your own here!
  
  return 0;
}

//______________________________________________________________________________________________

Long64_t ObsName::getCurrentEntry() const {
  // retrieve the current entry from the tree

  // since we don't have any tree or event pointer, there is usually
  // no way for us to know what entry we are currently looking
  // at. hence, we return -1 by default
  
  return -1;
}

//______________________________________________________________________________________________

TObjArray* ObsName::getBranchNames() const {
  // retrieve the list of branch names for this observable
  // ownership of the list belongs to the caller of the function
  DEBUGclass("retrieving branch names");

  // since we don't have a tree pointer, we probably also don't need any branches
  return NULL;
}
