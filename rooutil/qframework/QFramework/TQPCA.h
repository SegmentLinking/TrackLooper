//this file looks like plain C, but it's actually -*- c++ -*-
#ifndef __TQPCA__
#define __TQPCA__

#include "QFramework/TQNamedTaggable.h"
#include <TMatrixDSym.h>
#include <TMatrixD.h>
#include <TVectorD.h>

class TQPCA : public TQNamedTaggable {
protected:
  unsigned int nDim;
  double sumWeights;
  unsigned int nEntries;
  TMatrixDSym covMatrixA;
  std::vector<double> covMatrixB;

  bool calculated;
  bool besselCorrection;
  TVectorD* vEigenValues; //!
  TMatrixD* mEigenVectors; //!

  void printMatrix(const TMatrixD& matrix, const TString& title);

public:

  TQPCA();
  TQPCA(const TString& name, int nDim=2);
  ~TQPCA();

  void init(int nDim);
  void clear();
  int getNDim();
  double getSumWeights();
  int getNEntries();

  void useBesselCorrection(bool val);

  TString getDetailsAsString(int option = 1);
  void print(const TString& opts = "");

  TVectorD getEigenValues();
  TMatrixD getEigenVectors();
  TVectorD getEigenVector(size_t i);
  double getEigenValue(size_t i);
  void calculate();
  void calculateFromCorrelationMatrix();

  void fill(double weight, const double* row);
  bool fill(double weight, const std::vector<double>& row);
  bool add(const TQPCA& other, double factor=1);
  bool add(const TQPCA* other, double factor=1);
  void scale(double factor=1);
  TMatrixDSym getCovarianceMatrix();
  TMatrixDSym getCorrelationMatrix();

  void clearResults();

  void printRawData();
  void printResults();
  void printCovarianceMatrix();
  void printCorrelationMatrix();

  TString getProjection(size_t index, bool normalize = false);
  TString getReverseProjection(size_t index, bool normalize = false);
  double evaluateProjection(size_t index, const std::vector<double>& values, bool normalize = false);
  double evaluateReverseProjection(size_t index, const std::vector<double>& values, bool normalize = false);

  TString getHistogramDefinition2D(size_t iX, size_t iY, size_t nBinsXplus, size_t nBinsXminus, size_t nBinsYplus, size_t nBinsYminus);
  TString getHistogramDefinition(size_t i, size_t nBinsPlus, size_t nBinsMinus);
  TString getNormalizedHistogramDefinition2D(size_t iX, size_t iY, size_t nBinsX, double xMin, double xMax, size_t nBinsY, double yMin, double yMax);
  TString getNormalizedHistogramDefinition(size_t i, size_t nBins, double min, double max);

  bool exportEigenSystemTags (TQTaggable& tags, size_t iX, size_t iY, double nSigmaX, double nSigmaY, bool normalize=false);
  TQTaggable* getEigenSystemTags (size_t iX, size_t iY, double nSigmaX, double nSigmaY, bool normalize=false);
  TString getEigenSystemTagsAsString (size_t iX, size_t iY, double nSigmaX, double nSigmaY, bool normalize=false);
  bool exportEigenSystemTags (TQTaggable& tags, size_t iX, size_t iY, double nSigmaX, double nSigmaY, double normX, double normY);
  TQTaggable* getEigenSystemTags (size_t iX, size_t iY, double nSigmaX, double nSigmaY, double normX, double normY);
  TString getEigenSystemTagsAsString (size_t iX, size_t iY, double nSigmaX, double nSigmaY, double normX, double normY);

  double getVariance(size_t i);
  double getMean(size_t i);
  double getPCVariance(size_t i);
  double getPCMean(size_t i);

  ClassDefOverride(TQPCA,3) // enhanced version of the ROOT TPrincipal class
};

#endif //TQPCA
