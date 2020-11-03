#include "QFramework/TQStringUtils.h"
#include "QFramework/TQValue.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>

////////////////////////////////////////////////////////////////////////////////////////////////
//
// TQValue:
//
// The TQValue class is the base class for several classes wrapping primitive data types. It
// allows to specify a name associated with the value, which is used to return a hash value
// when calling TQValue::Hash().
//
////////////////////////////////////////////////////////////////////////////////////////////////

ClassImp(TQValue)


//__________________________________________________________________________________|___________

TQValue::TQValue() : TObject() {
  // Default constructor of TQValue class:

  fName = "undefined_name";
}


//__________________________________________________________________________________|___________

TQValue::TQValue(const TString& name) : TObject() {
  // Constructor of TQValue class:

  fName = name;
}


//__________________________________________________________________________________|___________

TQValue::TQValue(TQValue * value, const TString& newName) : TObject() {
  // Copy a TQValue object

  if (value) {
    if (newName.IsNull())
      fName = value->GetName();
    else if (isValidName(newName))
      fName = newName;
    else
      fName = "invalid_name";
  } else {
    fName = "invalid_value";
  }
}


//__________________________________________________________________________________|___________

bool TQValue::setName(const TString &name) {
  // set the name of this object

  if (isValidName(name)) {
    fName = name;
    return true;
  } else {
    return false;
  }
}


//__________________________________________________________________________________|___________

TString TQValue::getName() {
  // retrieve the name of this object
  return this->fName;
}

//__________________________________________________________________________________|___________

const TString& TQValue::getNameConst() const {
  // retrieve a const reference to the name of this object
  return this->fName;
}



//__________________________________________________________________________________|___________

const TString& TQValue::getValidNameCharacters() {
  // Return a string containing all valid characters that can be used as a
  // TQValue's name

  return TQStringUtils::getDefaultIDCharacters();
}


//__________________________________________________________________________________|___________

bool TQValue::isValidName(const TString& name) {
  // Return true if <name> is a valid name for a TQValue object

  return TQStringUtils::isValidIdentifier(name, getValidNameCharacters(), 1, -1);
}


//__________________________________________________________________________________|___________

TQValue * TQValue::newInstance(const TString& name, double value) {
  // Create a new instance of TQValueDouble

  return TQValueDouble::newInstance(name, value);
}


//__________________________________________________________________________________|___________

TQValue * TQValue::newInstance(const TString& name, int value) {
  // Create a new instance of TQValueInteger

  return TQValueInteger::newInstance(name, value);
}


//__________________________________________________________________________________|___________

TQValue * TQValue::newInstance(const TString& name, bool value) {
  // Create a new instance of TQValueBool

  return TQValueBool::newInstance(name, value);
}


//__________________________________________________________________________________|___________

TQValue * TQValue::newInstance(const TString& name, const TString& value) {
  // Create a new instance of TQValueString

  return TQValueString::newInstance(name, value);
}


//__________________________________________________________________________________|___________

TQValue * TQValue::newDouble(const TString& name, double value) {
  // Create a new instance of TQValueDouble

  return TQValueDouble::newInstance(name, value);
}


//__________________________________________________________________________________|___________

TQValue * TQValue::newInteger(const TString& name, int value) {
  // Create a new instance of TQValueInteger

  return TQValueInteger::newInstance(name, value);
}


//__________________________________________________________________________________|___________

TQValue * TQValue::newBool(const TString& name, bool value) {
  // Create a new instance of TQValueBool

  return TQValueBool::newInstance(name, value);
}


//__________________________________________________________________________________|___________

TQValue * TQValue::newString(const TString& name, const TString& value) {
  // Create a new instance of TQValueString

  return TQValueString::newInstance(name, value);
}


//__________________________________________________________________________________|___________

ULong_t TQValue::Hash() const {
  // Return a hash value computed from the name of this object. The name used is
  // the same as the one returned by TQValue::GetName()

  return fName.Hash();
}


//__________________________________________________________________________________|___________

const char * TQValue::GetName() const {
  // Return the name of this object. The name returned is the same as the one used
  // to computed the hash value returned by TQValue::Hash()

  return fName.Data();
}


//__________________________________________________________________________________|___________

int TQValue::Compare(const TObject * obj) const {

  if (obj && obj->InheritsFrom(TQValue::Class())) {

    /* cast the object to compare to */
    TQValue * value = (TQValue*)obj;

    /* compare doubles */
    if (isValidDouble() && value->isValidDouble()) {
      if (getDouble() < value->getDouble())
        return -1;
      else if (getDouble() > value->getDouble())
        return 1;
      else
        return 0;
      /* compare strings */
    } else {
      int compare = getString().CompareTo(value->getString());
      if (compare < 0)
        return -1;
      else if (compare > 0)
        return 1;
      else
        return 0;
    }
  } else {
    /* object are not comparable */
    return -1;
  }
}


//__________________________________________________________________________________|___________

bool TQValue::IsSortable() const {

  return true;
}


//__________________________________________________________________________________|___________

bool TQValue::isEquivalentTo(TQValue * val) const {
 
  if (!val) {
    return false;
  }
 
  // check double values
  if (this->isValidDouble() != val->isValidDouble()) {
    return false;
  }
  if (this->isValidDouble() && val->isValidDouble() && this->getDouble() != val->getDouble()) {
    return false;
  }
 
  // check bool values
  if (this->isValidBool() != val->isValidBool()) {
    return false;
  }
  if (this->isValidBool() && val->isValidBool() && this->getBool() != val->getBool()) {
    return false;
  }
 
  // check integer values
  if (this->isValidInteger() != val->isValidInteger()) {
    return false;
  }
  if (this->isValidInteger() && val->isValidInteger() && this->getInteger() != val->getInteger()) {
    return false;
  }
 
  // check string values
  if (this->getString().CompareTo(val->getString()) != 0) {
    return false;
  }
 
  return true;
}


//__________________________________________________________________________________|___________

bool TQValue::isDouble() const {
  // Return true if the value wrapped by this object is a double

  return false;
}


//__________________________________________________________________________________|___________

bool TQValue::isInteger() const {
  // Return true if the value wrapped by this object is an integer

  return false;
}


//__________________________________________________________________________________|___________

bool TQValue::isBool() const {
  // Return true if the value wrapped by this object is a bool

  return false;
}


//__________________________________________________________________________________|___________

bool TQValue::isString() const {
  // Return true if the value wrapped by this object is a string

  return false;
}


//__________________________________________________________________________________|___________

TString TQValue::getValueAsString() {

  return getString();
}


//__________________________________________________________________________________|___________

TQValue::~TQValue() {
  // Destructor of TQValue class:

}




////////////////////////////////////////////////////////////////////////////////////////////////
//
// TQValueDouble:
//
//
////////////////////////////////////////////////////////////////////////////////////////////////

ClassImp(TQValueDouble)


//__________________________________________________________________________________|___________

TQValueDouble::TQValueDouble() : TQValue() {
  // Default constructor of TQValueDouble class:

  fValue = 0.;
}


//__________________________________________________________________________________|___________

TQValueDouble::TQValueDouble(const TString& name, double value) : TQValue(name) {
  // Constructor of TQValueDouble class:

  fValue = value;
}


//__________________________________________________________________________________|___________

TQValueDouble::TQValueDouble(TQValueDouble * value, const TString& newName) : TQValue(value, newName) {
  // Copy a TQValueDouble object
 
  if (value)
    fValue = value->getDouble();
  else
    fValue = 0.;
}


//__________________________________________________________________________________|___________

TQValueDouble * TQValueDouble::newInstance(const TString& name, double value) {
  // Create a new instance of TQValueDouble

  if (isValidName(name))
    return new TQValueDouble(name, value);
  else
    return 0;
}


//__________________________________________________________________________________|___________

TQValue * TQValueDouble::copy(const TString& newName) {
  // Return a copy of this TQValueDouble object

  if (newName.IsNull() || isValidName(newName))
    return new TQValueDouble(this, newName);
  else
    return 0;
}


//__________________________________________________________________________________|___________

double TQValueDouble::getDouble() const {
  // Return the value wrapped by this object

  return fValue;
}


//__________________________________________________________________________________|___________

int TQValueDouble::getInteger() const {
  // Return the value wrapped by this object as int

  return (int)fValue;
}


//__________________________________________________________________________________|___________

bool TQValueDouble::getBool() const {
  // Return true if the value wrapped by this object is not equal to zero

  return (fValue != 0.);
}


//__________________________________________________________________________________|___________

TString TQValueDouble::getString() const {
  // Return a string describing the value wrapped by this object

  return TString::Format("%g", fValue);
}


//__________________________________________________________________________________|___________

bool TQValueDouble::isDouble() const {
  // Return true if the value wrapped by this object is a double

  return true;
}


//__________________________________________________________________________________|___________

bool TQValueDouble::isValidDouble() const {
  // Return true if the value wrapped by this object can be cast to a double (true
  // in this case)

  return true;
}


//__________________________________________________________________________________|___________

bool TQValueDouble::isValidInteger() const {
  // Return true if the value wrapped by this object can be cast to an integer (true
  // in this case)

  return true;
}


//__________________________________________________________________________________|___________

bool TQValueDouble::isValidBool() const {
  // Return true if the value wrapped by this object can be cast to a bool (true in
  // this case)

  return true;
}


//__________________________________________________________________________________|___________

TString TQValueDouble::getTypeAsString() {
  // Return a string describing the type of value wrapped by this object ("double"
  // in this case)

  return "double";
}


//__________________________________________________________________________________|___________

TString TQValueDouble::getAsString(bool forceQuotes) {
  // Return a string describing the value including its name

  if (forceQuotes)
    return TString::Format("%s = \"%g\"", GetName(), getDouble());
  else
    return TString::Format("%s = %g", GetName(), getDouble());
}


//__________________________________________________________________________________|___________

TQValueDouble::~TQValueDouble() {
  // Destructor of TQValueDouble class:

}




////////////////////////////////////////////////////////////////////////////////////////////////
//
// TQValueInteger:
//
//
////////////////////////////////////////////////////////////////////////////////////////////////

ClassImp(TQValueInteger)


//__________________________________________________________________________________|___________

TQValueInteger::TQValueInteger() : TQValue() {
  // Default constructor of TQValueInteger class:

  fValue = 0;
}


//__________________________________________________________________________________|___________

TQValueInteger::TQValueInteger(const TString& name, int value) : TQValue(name) {
  // Constructor of TQValueInteger class:

  fValue = value;
}


//__________________________________________________________________________________|___________

TQValueInteger::TQValueInteger(TQValueInteger * value, const TString& newName) : TQValue(value, newName) {
  // Copy a TQValueInteger object
 
  if (value)
    fValue = value->getInteger();
  else
    fValue = 0;
}


//__________________________________________________________________________________|___________

TQValueInteger * TQValueInteger::newInstance(const TString& name, int value) {
  // Create a new instance of TQValueDouble

  if (isValidName(name))
    return new TQValueInteger(name, value);
  else
    return 0;
}


//__________________________________________________________________________________|___________

TQValue * TQValueInteger::copy(const TString& newName) {
  // Return a copy of this TQValueInteger object

  if (newName.IsNull() || isValidName(newName))
    return new TQValueInteger(this, newName);
  else
    return 0;
}


//__________________________________________________________________________________|___________

double TQValueInteger::getDouble() const {
  // Return the value wrapped by this object as double

  return (double)fValue;
}


//__________________________________________________________________________________|___________

int TQValueInteger::getInteger() const {
  // Return the value wrapped by this object

  return fValue;
}


//__________________________________________________________________________________|___________

bool TQValueInteger::getBool() const {
  // Return true if the value wrapped by this object is not equal to zero

  return (fValue != 0);
}


//__________________________________________________________________________________|___________

TString TQValueInteger::getString() const {
  // Return a string describing the value wrapped by this object

  return TString::Format("%d", fValue);
}


//__________________________________________________________________________________|___________

bool TQValueInteger::isInteger() const {
  // Return true if the value wrapped by this object is an integer

  return true;
}


//__________________________________________________________________________________|___________

bool TQValueInteger::isValidDouble() const {
  // Return true if the value wrapped by this object can be cast to a double (true
  // in this case)

  return true;
}


//__________________________________________________________________________________|___________

bool TQValueInteger::isValidInteger() const {
  // Return true if the value wrapped by this object can be cast to an integer (true
  // in this case)

  return true;
}


//__________________________________________________________________________________|___________

bool TQValueInteger::isValidBool() const {
  // Return true if the value wrapped by this object can be cast to a bool (true
  // in this case)

  return true;
}


//__________________________________________________________________________________|___________

TString TQValueInteger::getTypeAsString() {
  // Return a string describing the type of value wrapped by this object ("integer"
  // in this case)

  return "integer";
}


//__________________________________________________________________________________|___________

TString TQValueInteger::getAsString(bool forceQuotes) {
  // Return a string describing the value including its name

  if (forceQuotes)
    return TString::Format("%s = \"%d\"", GetName(), getInteger());
  else
    return TString::Format("%s = %d", GetName(), getInteger());
}


//__________________________________________________________________________________|___________

TQValueInteger::~TQValueInteger() {
  // Destructor of TQValueInteger class:

}




////////////////////////////////////////////////////////////////////////////////////////////////
//
// TQValueBool:
//
//
////////////////////////////////////////////////////////////////////////////////////////////////

ClassImp(TQValueBool)


//__________________________________________________________________________________|___________

TQValueBool::TQValueBool() : TQValue() {
  // Default constructor of TQValueBool class:

  fValue = false;
}


//__________________________________________________________________________________|___________

TQValueBool::TQValueBool(const TString& name, bool value) : TQValue(name) {
  // Constructor of TQValueBool class:

  fValue = value;
}


//__________________________________________________________________________________|___________

TQValueBool::TQValueBool(TQValueBool * value, const TString& newName) : TQValue(value, newName) {
  // Copy a TQValueBool object
 
  if (value)
    fValue = value->getBool();
  else
    fValue = false;
}


//__________________________________________________________________________________|___________

TQValueBool * TQValueBool::newInstance(const TString& name, bool value) {
  // Create a new instance of TQValueDouble

  if (isValidName(name))
    return new TQValueBool(name, value);
  else
    return 0;
}


//__________________________________________________________________________________|___________

TQValue * TQValueBool::copy(const TString& newName) {
  // Return a copy of this TQValueBool object

  if (newName.IsNull() || isValidName(newName))
    return new TQValueBool(this, newName);
  else
    return 0;
}


//__________________________________________________________________________________|___________

double TQValueBool::getDouble() const {
  // Return the value wrapped by this object as double

  return (fValue ? 1. : 0.);
}


//__________________________________________________________________________________|___________

int TQValueBool::getInteger() const {
  // Return the value wrapped by this object as int

  return (fValue ? 1 : 0);
}


//__________________________________________________________________________________|___________

bool TQValueBool::getBool() const {
  // Return the value wrapped by this object

  return fValue;
}


//__________________________________________________________________________________|___________

TString TQValueBool::getString() const {
  // Return a string describing the value wrapped by this object

  return TQStringUtils::getStringFromBool(fValue);
}


//__________________________________________________________________________________|___________

bool TQValueBool::isBool() const {
  // Return true if the value wrapped by this object is a bool

  return true;
}


//__________________________________________________________________________________|___________

bool TQValueBool::isValidDouble() const {
  // Return true if the value wrapped by this object can be cast to a double (true
  // in this case)

  return true;
}


//__________________________________________________________________________________|___________

bool TQValueBool::isValidInteger() const {
  // Return true if the value wrapped by this object can be cast to an integer (true
  // in this case)

  return true;
}


//__________________________________________________________________________________|___________

bool TQValueBool::isValidBool() const {
  // Return true if the value wrapped by this object can be cast to a bool (true
  // in this case)

  return true;
}


//__________________________________________________________________________________|___________

TString TQValueBool::getTypeAsString() {
  // Return a string describing the type of value wrapped by this object ("bool"
  // in this case)

  return "bool";
}


//__________________________________________________________________________________|___________

TString TQValueBool::getAsString(bool forceQuotes) {
  // Return a string describing the value including its name

  if (forceQuotes)
    return TString::Format("%s = \"%s\"", GetName(), getString().Data());
  else
    return TString::Format("%s = %s", GetName(), getString().Data());
}


//__________________________________________________________________________________|___________

TQValueBool::~TQValueBool() {
  // Destructor of TQValueBool class:

}




////////////////////////////////////////////////////////////////////////////////////////////////
//
// TQValueString:
//
//
////////////////////////////////////////////////////////////////////////////////////////////////

ClassImp(TQValueString)


//__________________________________________________________________________________|___________

TQValueString::TQValueString() : TQValue() {
  // Default constructor of TQValueString class:

  fValue = "";
}


//__________________________________________________________________________________|___________

TQValueString::TQValueString(const TString& name, const TString& value) : TQValue(name) {
  // Constructor of TQValueString class:

  fValue = value;
}


//__________________________________________________________________________________|___________

TQValueString::TQValueString(TQValueString * value, const TString& newName) : TQValue(value, newName) {
  // Copy a TQValueString object
 
  if (value)
    fValue = value->getString();
  else
    fValue = "";
}


//__________________________________________________________________________________|___________

TQValueString * TQValueString::newInstance(const TString& name, const TString& value) {
  // Create a new instance of TQValueDouble

  if (isValidName(name))
    return new TQValueString(name, value);
  else
    return 0;
}


//__________________________________________________________________________________|___________

TQValue * TQValueString::copy(const TString& newName) {
  // Return a copy of this TQValueString object

  if (newName.IsNull() || isValidName(newName))
    return new TQValueString(this, newName);
  else
    return 0;
}


//__________________________________________________________________________________|___________

double TQValueString::getDouble() const {
  // Return the value wrapped by this object as double

  return fValue.Atof();
}


//__________________________________________________________________________________|___________

int TQValueString::getInteger() const {
  // Return the value wrapped by this object as int
  if(fValue.Index("k") < fValue.Length() || (fValue.Index("#") == 0)){
    int color = TQStringUtils::getColorFromString(fValue);
    if(color >= 0) return color;
    return fValue.Atoi();
  }
  return fValue.Atoi();
}

//__________________________________________________________________________________|___________

bool TQValueString::getBool() const {
  // Return true if the string wrapped by this object is either "yes", "true",
  // "ok" or a number not equal to zero
 
  return TQStringUtils::getBoolFromString(fValue);
}


//__________________________________________________________________________________|___________

TString TQValueString::getString() const {
  // Return the value wrapped by this object

  return fValue;
}


//__________________________________________________________________________________|___________

bool TQValueString::isString() const {
  // Return true if the value wrapped by this object is a string

  return true;
}


//__________________________________________________________________________________|___________

bool TQValueString::isValidDouble() const {
  // Return true if the value wrapped by this object can be cast to a double

  return fValue.IsFloat();
}


//__________________________________________________________________________________|___________

bool TQValueString::isValidInteger() const {
  // Return true if the value wrapped by this object can be cast to an integer

  return (fValue.IsFloat() || (TQStringUtils::getColorFromString(fValue) >= 0));
}


//__________________________________________________________________________________|___________

bool TQValueString::isValidBool() const {
  // Return true if the value wrapped by this object can be cast to a bool

  bool isBool = false;
  TQStringUtils::getBoolFromString(fValue, isBool);
  return isBool;
}


//__________________________________________________________________________________|___________

TString TQValueString::getAsString(bool forceQuotes) {
  // Return a string describing the value including its name
  return TString(GetName()) + " = " + (forceQuotes ? "\"" : "") + getValueAsString() + (forceQuotes ? "\"" : "");
}


//__________________________________________________________________________________|___________

TString TQValueString::getTypeAsString() {
  // Return a string describing the type of value wrapped by this object ("string"
  // in this case)

  return "string";
}


//__________________________________________________________________________________|___________

TString TQValueString::getValueAsString() {
 
  // TString value = TQStringUtils::insertEscapes(getString(), "\\\"");
  TString value = getString();
 
  /* put string into quotation marks */
  value.Prepend("\"");
  value.Append("\"");

  return value;
}


//__________________________________________________________________________________|___________

TQValueString::~TQValueString() {
  // Destructor of TQValueString class:

}


