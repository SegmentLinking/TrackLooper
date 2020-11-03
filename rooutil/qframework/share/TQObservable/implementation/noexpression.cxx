//______________________________________________________________________________________________

ObsName::ObsName(const TString& name):
ParentClass(name)
{
  // constructor with name argument
  DEBUGclass("constructor called with '%s'",name.Data());
}
