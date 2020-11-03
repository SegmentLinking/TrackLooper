//this file looks like plain C, but it's actually -*- c++ -*-
#ifndef __TQValue__
#define __TQValue__

#include "TObject.h"
#include "TString.h"

class TQValue : public TObject {

protected:

  TString fName;
  //TQValue * nextElement; // TODO: include

  TQValue(const TString& name);
  TQValue(TQValue * value, const TString& newName = "");

public:

  virtual TString getName();
  virtual const TString& getNameConst() const;

  static const TString& getValidNameCharacters();
  static bool isValidName(const TString& name);

  static TQValue * newInstance(const TString& name, double value);
  static TQValue * newInstance(const TString& name, int value);
  static TQValue * newInstance(const TString& name, bool value);
  static TQValue * newInstance(const TString& name, const TString& value);

  static TQValue * newDouble (const TString& name, double value);
  static TQValue * newInteger (const TString& name, int value);
  static TQValue * newBool (const TString& name, bool value);
  static TQValue * newString (const TString& name, const TString& value);

  TQValue();

  virtual bool setName(const TString &name);

  virtual TQValue * copy(const TString& newName = "") = 0;
 
  virtual ULong_t Hash() const override;
  virtual const char * GetName() const override ;
  virtual int Compare(const TObject * obj) const override;
  virtual bool IsSortable() const override;
  virtual bool isEquivalentTo(TQValue * val) const;

  virtual double getDouble() const = 0;
  virtual int getInteger() const = 0;
  virtual bool getBool() const = 0;
  virtual TString getString() const = 0;

  virtual bool isDouble() const;
  virtual bool isInteger() const;
  virtual bool isBool() const;
  virtual bool isString() const;

  virtual bool isValidDouble() const = 0;
  virtual bool isValidInteger() const = 0;
  virtual bool isValidBool() const = 0;

  virtual TString getAsString(bool forceQuotes = false) = 0;
  virtual TString getTypeAsString() = 0;
  virtual TString getValueAsString();

  //virtual TQValue * addElement(TQValue * element) // TODO: include

  virtual ~TQValue();
 
  ClassDefOverride(TQValue, 1); // base class for information storage

};


class TQValueDouble : public TQValue {

protected:

  double fValue;

protected:

  TQValueDouble(const TString& name, double value);
  TQValueDouble(TQValueDouble * value, const TString& newName = "");

public:

  static TQValueDouble * newInstance(const TString& name, double value);

  TQValueDouble();

  virtual TQValue * copy(const TString& newName = "") override;

  virtual double getDouble() const override;
  virtual int getInteger() const override;
  virtual bool getBool() const override;
  virtual TString getString() const override;

  virtual bool isDouble() const override;

  virtual bool isValidDouble() const override;
  virtual bool isValidInteger() const override;
  virtual bool isValidBool() const override;

  virtual TString getAsString(bool forceQuotes = false) override;
  virtual TString getTypeAsString() override;

  virtual ~TQValueDouble();
 
  ClassDefOverride(TQValueDouble, 1); // class for storage of a double-precision floating point number

};


class TQValueInteger : public TQValue {

protected:

  int fValue;

protected:

  TQValueInteger(const TString& name, int value);
  TQValueInteger(TQValueInteger * value, const TString& newName = "");

public:

  static TQValueInteger * newInstance(const TString& name, int value);

  TQValueInteger();

  virtual TQValue * copy(const TString& newName = "") override;

  virtual double getDouble() const override;
  virtual int getInteger() const override;
  virtual bool getBool() const override;
  virtual TString getString() const override;

  virtual bool isInteger() const override;

  virtual bool isValidDouble() const override;
  virtual bool isValidInteger() const override;
  virtual bool isValidBool() const override;

  virtual TString getAsString(bool forceQuotes = false) override;
  virtual TString getTypeAsString() override;

  virtual ~TQValueInteger();
 
  ClassDefOverride(TQValueInteger, 1); // class for storage of an integer number

};


class TQValueBool : public TQValue {

protected:

  bool fValue;

protected:

  TQValueBool(const TString& name, bool value);
  TQValueBool(TQValueBool * value, const TString& newName = "");

public:

  static TQValueBool * newInstance(const TString& name, bool value);

  TQValueBool();

  virtual TQValue * copy(const TString& newName = "") override;

  virtual double getDouble() const override;
  virtual int getInteger() const override;
  virtual bool getBool() const override;
  virtual TString getString() const override;

  virtual bool isBool() const override;

  virtual bool isValidDouble() const override;
  virtual bool isValidInteger() const override;
  virtual bool isValidBool() const override;

  virtual TString getAsString(bool forceQuotes = false) override;
  virtual TString getTypeAsString() override;

  virtual ~TQValueBool();
 
  ClassDefOverride(TQValueBool, 1); // class for storage of a boolean value

};


class TQValueString : public TQValue {

protected:

  TString fValue;

protected:

  TQValueString(const TString& name, const TString& value);
  TQValueString(TQValueString * value, const TString& newName = "");

public:

  static TQValueString * newInstance(const TString& name, const TString& value);

  TQValueString();

  virtual TQValue * copy(const TString& newName = "") override;

  virtual double getDouble() const override;
  virtual int getInteger() const override;
  virtual bool getBool() const override;
  virtual TString getString() const override;

  virtual bool isString() const override;

  virtual bool isValidDouble() const override;
  virtual bool isValidInteger() const override;
  virtual bool isValidBool() const override;

  virtual TString getAsString(bool forceQuotes = false) override;
  virtual TString getTypeAsString() override;
  virtual TString getValueAsString() override;

  virtual ~TQValueString();
 
  ClassDefOverride(TQValueString, 1); // class for storage of a string value

};

#endif
