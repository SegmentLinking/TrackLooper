#include "QFramework/TQCompiledCut.h"
#include "QFramework/TQCutFactory.h"
#include "TObjArray.h"
#include "TObjString.h"
#include "TFormula.h"
#include "QFramework/TQUtils.h"
#include "QFramework/TQStringUtils.h"
#include "QFramework/TQIterator.h"
#include "QFramework/TQLibrary.h"

#include <iostream>
#include <fstream>
#include <sstream>
#include <cstdlib>

////////////////////////////////////////////////////////////////////////////////////////////////
//
// TQCutFactory
//
// This is a legacy class that should no longer be used. Its
// functionality has been added to the TQCut class itself.
//
////////////////////////////////////////////////////////////////////////////////////////////////

ClassImp(TQCutFactory)


//______________________________________________________________________________________________

TQCutFactory::TQCutFactory() : TObject() {
  fCuts = new TList();
  fTreeObservableTemplates = 0;
}


//______________________________________________________________________________________________

void TQCutFactory::addCut(TString definition) {
  fCuts->AddLast(new TObjString(definition));
}

//______________________________________________________________________________________________

TString TQCutFactory::findCut(TString name) {

  TString * cutName = new TString();
  TString * baseCutName = new TString();
  TString * cutExpr = new TString();
  TString * weightExpr = new TString();
  TString * cutName2 = new TString();
  TString * baseCutName2= new TString();
  TString * cutExpr2 = new TString();
  TString * weightExpr2 = new TString();

  TIterator * itr = fCuts->MakeIterator(false);
  TObject * obj;
  TString cutDef = "";
  while ((obj = itr->Next())) {
    TString tmpExp = ((TObjString*)obj)->String();

    //Parsing the cut expression
    (*cutName)=""; (*baseCutName)="";
    TQCompiledCut::parseCutDefinition(tmpExp, cutName, baseCutName, cutExpr, weightExpr);

    // cout << " considering " << (*cutName) << endl;
    if ((*cutName) == name) {
      cutDef = tmpExp;
      break;
    }
  }

  delete itr;

  delete cutName ;
  delete baseCutName ;
  delete cutExpr ;
  delete weightExpr ;
  delete cutName2 ;
  delete baseCutName2 ;
  delete cutExpr2 ;
  delete weightExpr2 ;

  return cutDef;
}

//______________________________________________________________________________________________

TString TQCutFactory::removeCut(TString name) {

  TString * cutName = new TString();
  TString * baseCutName = new TString();
  TString * cutExpr = new TString();
  TString * weightExpr = new TString();
  TString * cutName2 = new TString();
  TString * baseCutName2= new TString();
  TString * cutExpr2 = new TString();
  TString * weightExpr2 = new TString();

  TIterator * itr = fCuts->MakeIterator(false);
  TObject * obj;
  TString cutDef = "";
  bool found = false;
  while ((obj = itr->Next())) {
    cutDef = ((TObjString*)obj)->String();

    //Parsing the cut expression
    (*cutName)=""; (*baseCutName)="";
    TQCompiledCut::parseCutDefinition(cutDef, cutName, baseCutName, cutExpr, weightExpr);

    // cout << " considering " << (*cutName) << endl;
    if ((*cutName) == name) {

      found = true;
      // cout << "Removing " << (*cutName) << endl;
      fCuts->Remove(obj);

      //Checking whether any other cut depends on this one
      bool hasDependence=false;
      TIterator * itr2 = fCuts->MakeIterator(false);
      TObject * obj2;
      while (!hasDependence && (obj2 = itr2->Next())) {
        TString cutDef2 = ((TObjString*)obj2)->String();
        //Parsing the cut expression
        TQCompiledCut::parseCutDefinition(cutDef2, cutName2, baseCutName2, cutExpr2, weightExpr2);
        if ((*baseCutName2)==(*cutName)) {
          hasDependence=true;
          // cout << "\tCut " << (*cutName2) << " is orphaned" << endl;
        }
      }
      delete itr2;
    }
    if (found) break;
  }
  delete itr;

  delete cutName ;
  delete baseCutName ;
  delete cutExpr ;
  delete weightExpr ;
  delete cutName2 ;
  delete baseCutName2 ;
  delete cutExpr2 ;
  delete weightExpr2 ;

  return cutDef;
}


//______________________________________________________________________________________________

void TQCutFactory::setTreeObservableTemplates(TList * treeObservableTemplates) {
  WARNclass("this functionality is deprecated. please use 'TQObservable::addObservable(myObservable,name)' instead");
  fTreeObservableTemplates = treeObservableTemplates;
  TQObservableIterator itr(fTreeObservableTemplates);
  while(itr.hasNext()){
    TQObservable* obs = itr.readNext();
    if(obs) TQObservable::addObservable(obs);
  }
}

//______________________________________________________________________________________________

void TQCutFactory::print() {
  for (int icut = 0; icut < fCuts->GetEntries(); ++icut) {
    std::cout << ((TObjString*) fCuts->At(icut))->GetString() << std::endl;
  }
}


//______________________________________________________________________________________________

void TQCutFactory::orderCutDefs() {

  TList * outputCuts = new TList(); //Will contain the ordered list of cut definitions
 
  TString * cutName = new TString();
  TString * baseCutName = new TString();
  TString * cutExpr = new TString();
  TString * weightExpr = new TString();
  TString * cutName2 = new TString();
  TString * baseCutName2= new TString();
  TString * cutExpr2 = new TString();
  TString * weightExpr2 = new TString();

  while (fCuts->GetSize()>0) {
    TIterator * itr = fCuts->MakeIterator(false);
    TObject * obj;
    while ((obj = itr->Next())) {
      TString cutDef = ((TObjString*)obj)->String();
      //Parsing the cut expression
      (*cutName)=""; (*baseCutName)="";
      TQCompiledCut::parseCutDefinition(cutDef, cutName, baseCutName, cutExpr, weightExpr);

      //Checking whether any other cut depends on this one
      bool hasDependence=false;
      TIterator * itr2 = fCuts->MakeIterator(false);
      TObject * obj2;
      while (!hasDependence && (obj2 = itr2->Next())) {
        TString cutDef2 = ((TObjString*)obj2)->String();
        //Parsing the cut expression
        TQCompiledCut::parseCutDefinition(cutDef2, cutName2, baseCutName2, cutExpr2, weightExpr2);
        if ((*baseCutName2)==(*cutName))
          hasDependence=true;
      }
      if (!hasDependence) {
        outputCuts->AddFirst(new TObjString(cutDef));
        fCuts->Remove(obj);
      }
      delete itr2;
    }
    delete itr;
  }
  delete cutName ;
  delete baseCutName ;
  delete cutExpr ;
  delete weightExpr ;
  delete cutName2 ;
  delete baseCutName2 ;
  delete cutExpr2 ;
  delete weightExpr2 ;


  //Replace the input list with the ordered one
  delete fCuts;
  fCuts = outputCuts; 
}

//______________________________________________________________________________________________

TQCompiledCut * TQCutFactory::compileCutsWithoutEvaluation() {

  TQIterator itr(fCuts);
  TQCompiledCut * baseCut = 0;
  while(itr.hasNext()){
    TObjString* obj = dynamic_cast<TObjString*>(itr.readNext());
    if(!obj) continue;
    TString cutDefCompiled = obj->String();
 
    if (baseCut) {
      baseCut->addCut(cutDefCompiled);
    } else { 
      baseCut = TQCompiledCut::createCut(cutDefCompiled);
    }
  }

  return baseCut;

}


//______________________________________________________________________________________________

TQCompiledCut * TQCutFactory::compileCuts(TString parameter) {

  TQIterator itr(fCuts);
  TQCompiledCut * baseCut = 0;
  while(itr.hasNext()){
    TObjString* obj = dynamic_cast<TObjString*>(itr.readNext());
    if(!obj) continue;
    TString cutDefCompiled = evaluate(obj->String(), parameter);
 
    if (baseCut) {
      baseCut->addCut(cutDefCompiled);
    } else { 
      baseCut = TQCompiledCut::createCut(cutDefCompiled);
    }
  }

  return baseCut;

}


//______________________________________________________________________________________________

TString TQCutFactory::evaluateSubExpression(const TString& input_, const TString& parameter) {

  TString input(input_);
  TString resultExpression("");

  /* -- evaluate sub expressions recursively -- */

  while (!input.IsNull()) {

    Ssiz_t pos = input.Index("{");

    if (pos == kNPOS) {

      resultExpression.Append(input);
      input.Remove(0);

    } else if (pos == 0) {

      int depth = 1;
      while (depth > 0 && ++pos < input.Length()) {
        char ch = input(pos);
        if (ch == '{') { depth++; }
        if (ch == '}') { depth--; }
      }

      TString subInput(input(1, pos - 1));
      input.Remove(0, pos + 1);

      TString subExpression(TQStringUtils::trim(evaluate(subInput, parameter)));
      resultExpression.Append(subExpression);

    } else {

      TString subInput = input(0, pos);
      resultExpression.Append(subInput);
      input.Remove(0, pos);

    }

  }
 
  return resultExpression;
 
}


//______________________________________________________________________________________________

TString TQCutFactory::evaluate(const TString& input_, const TString& parameter) {
  TString result = evaluateSubExpression(input_, parameter);

  /* -- replace parameter -- */
 
  TList * tokens = TQStringUtils::tokenize(parameter, ",", true, "", "''");

  result.ReplaceAll("'", "\"");

  for (int iPar = 0; iPar < tokens->GetEntries(); iPar++) {

    TString * key = new TString();
    TString * value = new TString();

    if (TQUtils::parseAssignment(((TObjString*)tokens->At(iPar))->String(), key, value)) {

      key->ToUpper();
      value->ReplaceAll("'", "\"");
 
      result.ReplaceAll(TString::Format("$%s", key->Data()), *value);

    }

    delete key;
    delete value;

  }

  /* delete the token list again */
  tokens->Delete();
  delete tokens;


  /* -- evaluate this expression -- */

  Ssiz_t posIf = result.Index("?");

  if (posIf == kNPOS) {
 
    return TQStringUtils::trim(result);

  } else {

    /* extract the formula from the "if expression" */
    TString ifExpr = result(0, posIf);

    TFormula * formula = new TFormula("", ifExpr);

    /* extract the "then-else expression" ... */
    TString thenElseExpr = result(posIf + 1, result.Length());
 
    /* ... and split it into "then" and "else expression" */
    TString thenExpr, elseExpr;
    Ssiz_t posElse = thenElseExpr.Index(":");
    if (posElse == kNPOS) {
      thenExpr = thenElseExpr;
    } else {
      thenExpr = thenElseExpr(0, posElse);
      elseExpr = thenElseExpr(posElse + 1, thenElseExpr.Length());
    }

    /* let the formula decide which expression to return */
    if (formula->Eval(0.)) {
      delete formula;
      return TQStringUtils::trim(thenExpr);
    } else {
      delete formula;
      return TQStringUtils::trim(elseExpr);
    }

  }

}


//______________________________________________________________________________________________

TQCutFactory::~TQCutFactory() {
  delete fCuts;
}

//______________________________________________________________________________________________

bool TQCutFactory::isEmpty() {
  return (this->fCuts->GetEntries() < 1);
}


