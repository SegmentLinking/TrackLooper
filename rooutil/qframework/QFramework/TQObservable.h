//this file looks like plain C, but it's actually -*- c++ -*-
#ifndef __TQObservable__
#define __TQObservable__

#include "TNamed.h"
#include "TTree.h"
#include "QFramework/TQSample.h"

#define DECLARE_OBSERVABLE_FACTORY(CLASSNAME,ARG) public:               \
  static const TQObservable::Factory<CLASSNAME> Factory;                \
  static const TQObservable::FactoryBase* getFactory()                  \
  { return &(CLASSNAME::Factory); }                                     \
  virtual bool hasFactory() const override { return true; }             \
  static TQObservable* tryCreateInstance(ARG);                          \
  virtual TQObservable* tryCreateInstanceVirtual(const TString& expr)  const final override \
  { return CLASSNAME::tryCreateInstance(expr); }

#define DEFINE_OBSERVABLE_FACTORY(CLASSNAME,ARG)                        \
  const TQObservable::Factory<CLASSNAME> CLASSNAME::Factory = TQObservable::Factory<CLASSNAME>(#CLASSNAME); \
  TQObservable* CLASSNAME::tryCreateInstance(ARG)

#define DEFINE_TEMPLATE_OBSERVABLE_FACTORY(CLASSNAME,TEMPLATE,ARG)      \
  template<class TEMPLATE> const TQObservable::Factory<CLASSNAME<TEMPLATE> > CLASSNAME<TEMPLATE>::Factory = TQObservable::Factory<CLASSNAME<TEMPLATE> >((std::string(#CLASSNAME) + "<" + TQObservableFactory::printType<TEMPLATE>() + ">").c_str()); \
  template<class TEMPLATE> TQObservable* CLASSNAME<TEMPLATE>::tryCreateInstance(ARG)

#define DEFINE_TEMPLATE_OBSERVABLE_FACTORY_SPECIALIZATION(CLASSNAME,TEMPLATE,ARG) \
  template<> TQObservable::Factory<CLASSNAME<TEMPLATE> > const CLASSNAME<TEMPLATE>::Factory = TQObservable::Factory<CLASSNAME<TEMPLATE> >(#CLASSNAME "<" #TEMPLATE ">"); \
  template<> TQObservable* CLASSNAME<TEMPLATE>::tryCreateInstance(ARG)    

#define DEFINE_TEMPLATE_ARGUMENT( type ) namespace TQObservableFactory { template<> constexpr const char* printType<type>() { return #type; } }

namespace TQObservableFactory {
  bool setupDefault();
  template <typename T> constexpr const char* printType() { return "undefined template argument, use DEFINE_TEMPLATE_ARGUMENT( ... ) to define!"; }
  // implement specializations for given types
}
  
class TQObservable : public TNamed {
  
private:
  static bool gIgnoreExpressionMismatch;
public:
  static TString makeObservableName(const TString& name);
  static bool matchExpressions(const TString& ex1, const TString& ex2, bool requirePrefix = false);
  static TString replaceBools(TString expression);
  static TString unreplaceBools(TString expression);
protected:
  TQSample * fSample = NULL; //!

  bool fIsManaged = false;
  bool fIsInitialized = false;

  class FactoryBase { // nested
  public:
    virtual const char* className() const = 0;
    virtual TQObservable* tryCreateInstance(const TString& expr) const = 0;
  };

  template<class ObsType> class Factory : public FactoryBase { // nested
  private:
    const std::string cName;
  public:
    Factory(const char* s) : cName(s) {};
    virtual const char* className() const final override { return this->cName.c_str(); }
    virtual TQObservable* tryCreateInstance(const TString& expr) const final override {
      return ObsType::tryCreateInstance(expr);
    }
  };

  static bool gAllowErrorMessages;

  virtual bool initializeSelf() = 0;
  virtual bool finalizeSelf() = 0;
  virtual bool hasFactory() const;
  
public:
  class Manager { // nested
    std::vector<const TQObservable::FactoryBase*> observableFactories;
    TFolder* activeSet = NULL;
    TFolder* sets = new TFolder("TQObservables","Global Observable List");
    Manager();
  public:
    TQObservable* createObservable(TString expression, TQTaggable* tags = NULL);
    void registerFactory(const TQObservable::FactoryBase* factory, bool putFirst);
    void printFactories();
    void clearFactories();
    bool setActiveSet(const TString& name);
    void cloneActiveSet(const TString& newName);
    void clearSet(const TString& name);
    void createEmptySet(const TString& name);
    TCollection* listOfSets();
		TFolder* getActiveSet();
    friend TQObservable;
  };
	static void clearAll();

	static Manager& getManager();

  bool isInitialized() const;
  virtual Long64_t getCurrentEntry() const = 0;

  static void allowErrorMessages(bool val);

  void setName(const TString& name);
  const TString& getNameConst() const;
  TString getName() const;

  static TString compileExpression(const TString& input, TQTaggable* tags, bool replaceBools=true);
  inline virtual bool isPrefixRequired() const {return false;}; //needs to be overridden if an observable (with a factory) should only be matched if it's prefix is present. In this case the observable must ensure to include its prefix in strings returned by getExpression, getActiveExpression,...
  
  TQObservable();
  TQObservable(const TString& expression);
  virtual ~TQObservable();

  static TQObservable* getObservable(const TString& exprname, TQTaggable* tags);
  static bool addObservable(TQObservable* obs);
  static bool addObservable(TQObservable* obs, const TString& name);

  static void printObservables(const TString& filter = "*");
  virtual void print() const;

  virtual bool initialize(TQSample * sample);
  virtual bool finalize();

  virtual bool isSetup() const;

  virtual double getValue() const = 0;
  inline virtual double getValueAt(int index) const {if (index!=0) {throw std::runtime_error("Attempt to retrieve value from illegal index in observable."); return -999.;} else return this->getValue();}
  
  inline virtual int getNevaluations() const {return 1;}
  
  enum ObservableType {scalar,vector,unknown};
  inline virtual TQObservable::ObservableType getObservableType() const {return TQObservable::ObservableType::scalar;}

  virtual TQObservable* tryCreateInstanceVirtual (const TString& expr) const;

  virtual bool hasExpression() const;
  virtual const TString& getExpression() const;
  virtual void setExpression(const TString& expr);
  virtual TString getActiveExpression() const;
  virtual TString getCompiledExpression(TQTaggable* tags) const;
  virtual TObjArray* getBranchNames() const = 0;
 
  virtual TQObservable* getClone() const;

  static TQObservable::Manager manager;


  ClassDefOverride(TQObservable, 0); // abstract base class for data retrieval from trees

};

#endif
