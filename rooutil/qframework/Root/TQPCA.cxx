#include "QFramework/TQPCA.h"
#include <TMatrixDEigen.h>
#include <TMatrixDSymEigen.h>
#include <iostream>
#include <math.h>
#include "QFramework/TQStringUtils.h"

////////////////////////////////////////////////////////////////////////////////////////////////
//
// TQPCA
// 
// The TQPCA is a improved version of the standard root class TPrincipal. 
// It provides extended functionality, including management of metadata and event weights.
// It is integrated to the HWWAnalysisCode and used by the TQPCAAnalysisJob.
// Interfaces for adding and scaling TQPCA objects are provided and intertwined with 
// the TQSampleDataReader facilities.
// 
////////////////////////////////////////////////////////////////////////////////////////////////

ClassImp(TQPCA)

TQPCA::TQPCA() :
TQNamedTaggable("TQPCA"),
  nDim(0),
  sumWeights(0),
  nEntries(0),
  calculated(false),
  besselCorrection(false),
  vEigenValues(NULL),
  mEigenVectors(NULL)
{
  // default constructor
}

TQPCA::TQPCA(const TString& name, int nDim) :
  TQNamedTaggable(name),
  nDim(nDim),
  sumWeights(0),
  nEntries(0),
  covMatrixA(nDim),
  covMatrixB(nDim),
  calculated(false),
  besselCorrection(false),
  vEigenValues(NULL),
  mEigenVectors(NULL)
{
  // default constructor
}

TQPCA::~TQPCA(){
  // destructor
  this->clear();
}

void TQPCA::useBesselCorrection(bool val){
  // control the use of the Bessel correction
  // for the calculation of the (co)variance,
  // i.e. the use of 1/(n-1) instead of 1/n
  // in the (co)variance formula
  this->besselCorrection = val;
}

void TQPCA::clear(){
  // clear all data
  // the user needs to call TQPCA::init to reinitialize
  // before re-use of the instance is sensible
  this->calculated = false;
  this->covMatrixA.Clear();
  this->covMatrixB.clear();
}

int TQPCA::getNDim(){
  // retrieve the number of dimensions
  return this->nDim;
}

double TQPCA::getSumWeights(){
  // retrieve the total sum of weights
  return this->sumWeights;
}

int TQPCA::getNEntries(){
  // retrieve the total number of entries
  return this->nEntries;
}

TString TQPCA::getDetailsAsString(int/*option*/) {
  // retrieve an info-string 
  std::vector<TString> varnames;
  TString retval = TString::Format("%g entries",this->getSumWeights());
  if(this->getTag("varname",varnames) > 0){
    retval += " in " + TQStringUtils::concat(varnames,",");
  } else {
    retval += TString::Format(" in %d dimensions",this->getNDim());
  }
  return retval;
}

void TQPCA::init(int nDim){
  // initialize the object based on given dimensionality
  this->clear();
  this->nDim = nDim;
  this->nEntries = 0;
  this->sumWeights = 0;
  this->covMatrixA.ResizeTo(this->nDim,this->nDim);
  this->covMatrixB.reserve(this->nDim);

  for(size_t i=0; i<this->nDim; i++){
    covMatrixB[i] = 0;
    for(size_t j=0; j<=this->nDim; j++){
      covMatrixA[i][j] = 0;
    }
  }
}

void TQPCA::print(const TString&/*opts*/){
  // print some information about the object
  std::cout << this->getDetailsAsString() << std::endl;
}

bool TQPCA::fill(double weight, const std::vector<double>& row){
  // fill values from a std::vector<double>
  if(row.size() != this->nDim) return false;
  this->fill(weight,&(row[0]));
  return true;
}
 

void TQPCA::printRawData(){
  this->printMatrix(this->covMatrixA,"covMatrixA");
  std::cout << TQStringUtils::makeBoldWhite("covMatrixB") << " ";
  for(size_t i=0; i<this->nDim; i++){
    std::cout << this->covMatrixB[i] << " ";
  }
  std::cout << std::endl;
}
 

void TQPCA::fill(double weight, const double* row){
  // fill values from an array
  this->sumWeights += weight;
  this->nEntries++;
  for(size_t i=0; i<this->nDim; i++){
    covMatrixB[i] += weight*row[i];
    for(size_t j=0; j<=i; j++){
      covMatrixA[j][i] += weight*row[i]*row[j]; 
    }
  }
}

TMatrixDSym TQPCA::getCovarianceMatrix(){
  // retrieve the covariance matrix
  TMatrixDSym covMatrix(this->nDim);
  double sw = this->sumWeights - this->besselCorrection;
  for(size_t i=0; i<this->nDim; i++){
    for(size_t j=0; j<=i; j++){
      covMatrix[i][j] = (this->covMatrixA[j][i] - (covMatrixB[i]*covMatrixB[j]/sw))/sw;
    }
    for(size_t j=i; j<this->nDim; j++){
      covMatrix[i][j] = (this->covMatrixA[i][j] - (covMatrixB[i]*covMatrixB[j]/sw))/sw;
    }
  }
  return covMatrix;
}


TMatrixDSym TQPCA::getCorrelationMatrix(){
  // retrieve the correlation matrix
  TMatrixDSym corrMatrix = this->getCovarianceMatrix();
  TMatrixDSym covMatrix = this->getCovarianceMatrix();
  // W = 1/W^2 as normalization factor for averages
  for(size_t i=0; i<this->nDim; ++i) {
    for(size_t j=0; j<this->nDim; ++j) {
      // corr[i,j] = cov[i,j] / (s_i * s_j)
      // with s_k = sqrt(cov[k,k]) the standard deviation
      // of variable k
      corrMatrix[i][j] /= sqrt(covMatrix[i][i] * covMatrix[j][j]);
    }
  }
  return corrMatrix;
}

TVectorD TQPCA::getEigenValues(){
  // retrieve the TVectorD of all eigenvectors
  if(!calculated) this->calculate();
  return *this->vEigenValues;
}

double TQPCA::getEigenValue(size_t i){
  // retrieve the i-th eigenvalue as a double
  if(!calculated) this->calculate();
  return (*(this->vEigenValues))[i];
}

TMatrixD TQPCA::getEigenVectors(){
  // retrieve the TMatrixD of all eigenvectors
  if(!calculated) this->calculate();
  return *this->mEigenVectors;
}

TVectorD TQPCA::getEigenVector(size_t i){
  // retrieve the i-th eigenvector as a TVectorD
  if(!calculated) this->calculate();
  TVectorD ev;
  for(size_t j=0; j<this->nDim; j++){
    ev[j] = (*(this->mEigenVectors))[j][i];
  }
  return ev;
}

void TQPCA::clearResults(){
  // delete all matrix operation results and reset the calculation status
  if(this->mEigenVectors) delete this->mEigenVectors;
  if(this->vEigenValues) delete this->vEigenValues;
  this->mEigenVectors = NULL;
  this->vEigenValues = NULL;
  this->calculated = false;
}

void TQPCA::calculate(){
  // calculate the eigenvectors and eigenvalues
  this->clearResults();
  TMatrixDSymEigen covMatrix = this->getCovarianceMatrix(); 
  this->mEigenVectors = new TMatrixD(covMatrix.GetEigenVectors());
  this->vEigenValues = new TVectorD(covMatrix.GetEigenValues());
  this->calculated = true;
}

void TQPCA::calculateFromCorrelationMatrix(){
  // calculate the eigenvectors and eigenvalues
  // from the correlation matrix instead of the covariance matrix
  this->clearResults();
  TMatrixDSymEigen corrMatrix(this->getCorrelationMatrix()); 
  this->mEigenVectors = new TMatrixD(corrMatrix.GetEigenVectors());
  this->vEigenValues = new TVectorD(corrMatrix.GetEigenValues());
  this->calculated = true;
}

bool TQPCA::add(const TQPCA* other, double /*factor*/){
  // add another TQPCA instance to this one
  if(!other) return false;
  return this->add(*other);
}

bool TQPCA::add(const TQPCA& other, double factor){
  // add two objects of type TQPCA
  this->clearResults();
  if(other.nDim != this->nDim) return false;

  this->sumWeights += other.sumWeights*factor;
  this->nEntries += other.nEntries;

  for(size_t i=0; i<this->nDim; i++){
    covMatrixB[i] += other.covMatrixB[i]*factor;
    for(size_t j=0; j<=i; j++){
      covMatrixA[i][j] += other.covMatrixA[i][j]*factor;
    }
  }
  return true;
}

void TQPCA::scale(double factor){
  // scale the entries with some given factor
  this->clearResults();
  this->sumWeights *= factor;

  for(size_t i=0; i<this->nDim; i++){
    this->covMatrixB[i] *= factor;
    for(size_t j=0; j<=i; j++){
      covMatrixA[i][j] *= factor;
    }
  }
}

void TQPCA::printMatrix(const TMatrixD& mat, const TString& title){
  // print any matrix to std::cout, nicely formatted
  std::cout << TQStringUtils::makeBoldWhite(TQStringUtils::fixedWidth(title,12,"c"));
  for(size_t i=0; i<this->nDim; i++){
    std::cout << TQStringUtils::fixedWidth(this->getTagStringDefault(TString::Format("varname.%d",(int)i),TString::Format("var%d",(int)i)),12,"r") << " ";
  }
  std::cout << std::endl;
  for(size_t i=0; i<this->nDim; i++){
    std::cout <<TQStringUtils::fixedWidth(this->getTagStringDefault(TString::Format("varname.%d",(int)i),TString::Format("var%d",(int)i)),12,"r") << " ";
    for(size_t j=0; j<this->nDim; j++){
      std::cout << TQStringUtils::fixedWidth(TString::Format("%g",mat[i][j]),12,"r");
    }
    std::cout << std::endl;
  }
}

void TQPCA::printCovarianceMatrix(){
  // print the covariance matrix to std::cout, nicely formatted
  TMatrixDSym cov(this->getCovarianceMatrix());
  this->printMatrix(cov,"cov(i,j)");
}

void TQPCA::printCorrelationMatrix(){
  // print the correlation matrix to std::cout, nicely formatted
  TMatrixDSym corr(this->getCorrelationMatrix());
  this->printMatrix(corr,"corr(i,j)");
}

void TQPCA::printResults(){
  // print the results to std::cout, nicely formatted
  if(!this->calculated) this->calculate();
  std::cout << TQStringUtils::makeBoldWhite(TQStringUtils::fixedWidth("PCA",12,"r"));
  for(size_t j=0; j<this->nDim; j++){
    std::cout << TQStringUtils::fixedWidth(TString::Format("#%d",(int)j),12,"c") << " ";
  }
  std::cout << std::endl;
  std::cout << TQStringUtils::fixedWidth("eigenvalue",12,"r");
  for(size_t j=0; j<this->nDim; j++){
    std::cout << TQStringUtils::fixedWidth(TString::Format("%g",(*(this->vEigenValues))[j]),12,"r") << " ";
  }
  std::cout << std::endl;
  for(size_t i=0; i<this->nDim; i++){
    std::cout <<TQStringUtils::fixedWidth(this->getTagStringDefault(TString::Format("varname.%d",(int)i),TString::Format("var%d",(int)i)),12,"r") << " ";
    for(size_t j=0; j<this->nDim; j++){
      std::cout << TQStringUtils::fixedWidth(TString::Format("%g",(*(this->mEigenVectors))[i][j]),12,"r") << " ";
    }
    std::cout << std::endl;
  }
}

double TQPCA::getVariance(size_t i){
  // retrieve the gaussian variance on variable i
  if(i>=this->nDim) return std::numeric_limits<double>::quiet_NaN();
  TMatrixD covMatrix(this->getCovarianceMatrix());
  return covMatrix[i][i];
}

double TQPCA::getMean(size_t i){
  // retrieve the weighted mean
  if(i>=this->nDim) return std::numeric_limits<double>::quiet_NaN();
  return this->covMatrixB[i]/this->sumWeights;
}

double TQPCA::getPCVariance(size_t i){
  // retrieve the gaussian variance on principal component i
  if(i>=this->nDim) return std::numeric_limits<double>::quiet_NaN();
  if(!this->calculated) this->calculate();
  return (*(this->vEigenValues))[i];
}

double TQPCA::getPCMean(size_t i){
  // retrieve the mean of the principal component i
  if(i>=this->nDim) return std::numeric_limits<double>::quiet_NaN();
  if(!this->calculated) this->calculate();
  double mean = 0;
  for(size_t j=0; j<this->nDim; j++){
    mean += this->getMean(j) * (*(this->mEigenVectors))[j][i];
  }
  return mean;
}

TString TQPCA::getProjection(size_t index, bool normalize){
  // retrieve the arithmetic expression for the projection
  // onto the principal component with the given index 
  // if normalize is true, the prefactors will be normalized
  // to their respective dimensional weighted averages
  if(index >= this->nDim) return "";
  if(!this->calculated) this->calculate();
  TString retval;
  for(size_t i=0; i<this->nDim; i++){
    if(normalize){
      retval += TString::Format("%g",(*(this->mEigenVectors))[i][index] / sqrt(this->getPCVariance(index)));
    } else {
      retval += TString::Format("%g",(*(this->mEigenVectors))[i][index]);
    }
    retval += " * ";
    if(normalize){
      retval += "( ";
      retval += this->getTagStringDefault(TString::Format("varexpression.%d",(int)i),"Var1");
      retval += TString::Format(" - %g",this->getMean(i));
      retval += " )";
    } else {
      retval += this->getTagStringDefault(TString::Format("varexpression.%d",(int)i),"Var1");
    }
    if(i < this->nDim-1) retval += " + ";
  }
  return retval;
}

double TQPCA::evaluateProjection(size_t index, const std::vector<double>& values, bool normalize){
  // evaluate the projection of the given coordinates
  // onto the principal component with the given index 
  // if normalize is true, the prefactors will be normalized
  // to their respective dimensional weighted averages
  if(index >= this->nDim) return std::numeric_limits<double>::quiet_NaN();
  if(!this->calculated) this->calculate();
  double retval = 0;
  for(size_t i=0; i<this->nDim; i++){
    if(normalize){
      retval += (*(this->mEigenVectors))[i][index] / sqrt(this->getPCVariance(index)) * (values[i] - this->getMean(i));
    } else {
      retval += (*(this->mEigenVectors))[i][index] * values[i];
    }
  }
  return retval;
}


TString TQPCA::getReverseProjection(size_t index, bool normalize){
  // retrieve the arithmetic expression for the projection
  // onto the input variable with the given index
  // if normalize is true, the prefactors will be normalized
  // to their respective dimensional weighted averages
  if(index >= this->nDim) return "";
  if(!this->calculated) this->calculate();
  TString retval;
  for(size_t i=0; i<this->nDim; i++){
    if(normalize){
      retval += TString::Format("%g",(*(mEigenVectors))[index][i] * sqrt(this->getPCVariance(i)));
    } else {
      retval += TString::Format("%g",(*(mEigenVectors))[index][i]);
    }
    retval += TString::Format(" * e%d",(int)i);
    if(i < this->nDim-1) retval += " + ";
  }
  return retval;
}

double TQPCA::evaluateReverseProjection(size_t index, const std::vector<double>& values, bool normalize){
  // evaluate the projection of the given coordinates
  // onto the input variable with the given index
  // if normalize is true, the prefactors will be normalized
  // to their respective dimensional weighted averages
  if(index >= this->nDim) return std::numeric_limits<double>::quiet_NaN();
  if(!this->calculated) this->calculate();
  double retval = normalize ? this->getMean(index) : 0;
  for(size_t i=0; i<this->nDim; i++){
    if(normalize){
      retval += (*(mEigenVectors))[index][i] * sqrt(this->getPCVariance(i)) * values[i];
    } else {
      retval += (*(mEigenVectors))[index][i] * values[i];
    }
  }
  return retval;
}


TString TQPCA::getHistogramDefinition2D(size_t iX, size_t iY, size_t nBinsXplus, size_t nBinsXminus, size_t nBinsYplus, size_t nBinsYminus){
  // retrieve the definition string for a 2d histogram
  return TString::Format("TH2F('pc_%d_%d','',%d,%g,%g,%d,%g,%g) << (%s : 'Principal Component %d', %s : 'Principal Component %d')",
                         (int)iX,(int)iY,
                         (int)(nBinsXplus+nBinsXminus),
                         this->getMean(iX) - nBinsXminus*sqrt(this->getVariance(iX)),
                         this->getMean(iX) + nBinsXplus *sqrt(this->getVariance(iX)),
                         (int)(nBinsXplus+nBinsXminus),
                         this->getMean(iY) - nBinsYminus*sqrt(this->getVariance(iY)),
                         this->getMean(iY) + nBinsYplus *sqrt(this->getVariance(iY)),
                         this->getProjection(iX,false).Data(),
                         (int)iX,
                         this->getProjection(iY,false).Data(),
                         (int)iY);
}

TString TQPCA::getNormalizedHistogramDefinition2D(size_t iX, size_t iY, size_t nBinsX, double xMin, double xMax, size_t nBinsY, double yMin, double yMax){
  // retrieve the definition string for a 2d histogram in normalized coordinates
  return TString::Format("TH2F('pc_%d_%d','',%d,%g,%g,%d,%g,%g) << (%s : 'Principal Component %d', %s : 'Principal Component %d')",
                         (int)iX,(int)iY,
                         (int)(nBinsX),
                         xMin,
                         xMax,
                         (int)(nBinsY),
                         yMin,
                         yMax,
                         this->getProjection(iX,true).Data(),
                         (int)iX,
                         this->getProjection(iY,true).Data(),
                         (int)iY);
}

TString TQPCA::getHistogramDefinition(size_t i, size_t nBinsPlus, size_t nBinsMinus){
  // retrieve the definition string for a histogram
  return TString::Format("TH1F('pc_%d','',%d,%g,%g) << (%s : 'Principal Component %d')",
                         (int)i,
                         (int)(nBinsPlus+nBinsMinus),
                         this->getMean(i) - nBinsMinus*sqrt(this->getVariance(i)),
                         this->getMean(i) + nBinsPlus *sqrt(this->getVariance(i)),
                         this->getProjection(i,false).Data(),
                         (int)i);
}

TString TQPCA::getNormalizedHistogramDefinition(size_t i, size_t nBins, double min, double max){
  // retrieve the definition string for a histogram in normalized coordinates
  return TString::Format("TH1F('pc_%d','',%d,%g,%g) << (%s : 'Principal Component %d')",
                         (int)i,
                         (int)(nBins),
                         min,
                         max,
                         this->getProjection(i,true).Data(),
                         (int)i);
}

bool TQPCA::exportEigenSystemTags(TQTaggable& tags, size_t iX, size_t iY, double nSigma1, double nSigma2, bool normalize){
  // export the TQPlotter tags required to draw the Eigen coordinate system
  if(std::max(iX,iY) >= this->nDim) return false;
  if(!this->calculated) this->calculate();

  std::vector<double> vals(this->nDim, 0.0);

  tags.setTagBool("axis.0.show",true);
  tags.setTagString("axis.0.title",TString::Format("PC %d",(int)iX));
  vals[iX] = 0; 
  tags.setTagDouble("axis.0.xMin",this->evaluateReverseProjection(iX,vals,normalize));
  tags.setTagDouble("axis.0.yMin",this->evaluateReverseProjection(iY,vals,normalize));
  vals[iX] = nSigma1; 
  tags.setTagDouble("axis.0.xMax",this->evaluateReverseProjection(iX,vals,normalize));
  tags.setTagDouble("axis.0.yMax",this->evaluateReverseProjection(iY,vals,normalize));
  vals[iX] = 0;
  tags.setTagDouble("axis.0.wMin",0);
  tags.setTagDouble("axis.0.wMax",nSigma1);
 
  tags.setTagBool("axis.1.show",true);
  tags.setTagString("axis.1.title",TString::Format("PC %d",(int)iY));
  vals[iY] = 0; 
  tags.setTagDouble("axis.1.xMin",this->evaluateReverseProjection(iX,vals,normalize));
  tags.setTagDouble("axis.1.yMin",this->evaluateReverseProjection(iY,vals,normalize));
  vals[iY] = nSigma2; 
  tags.setTagDouble("axis.1.xMax",this->evaluateReverseProjection(iX,vals,normalize));
  tags.setTagDouble("axis.1.yMax",this->evaluateReverseProjection(iY,vals,normalize));
  vals[iY] = 0;
  tags.setTagDouble("axis.1.wMin",0);
  tags.setTagDouble("axis.1.wMax",nSigma2);

  return true;
}

bool TQPCA::exportEigenSystemTags(TQTaggable& tags, size_t iX, size_t iY, double nSigma1, double nSigma2, double normX, double normY){
  // export the TQPlotter tags required to draw the Eigen coordinate system
  if(std::max(iX,iY) >= this->nDim) return false;
  if(!this->calculated) this->calculate();

  std::vector<double> vals(this->nDim, 0.0);
  bool normalize = false;

  tags.setTagBool("axis.0.show",true);
  tags.setTagString("axis.0.title",TString::Format("PC %d",(int)iX));
  vals[iX] = 0; 
  tags.setTagDouble("axis.0.xMin",this->evaluateReverseProjection(iX,vals,normalize) * normX);
  tags.setTagDouble("axis.0.yMin",this->evaluateReverseProjection(iY,vals,normalize) * normY);
  vals[iX] = nSigma1; 
  tags.setTagDouble("axis.0.xMax",this->evaluateReverseProjection(iX,vals,normalize) * normX);
  tags.setTagDouble("axis.0.yMax",this->evaluateReverseProjection(iY,vals,normalize) * normY);
  vals[iX] = 0;
  tags.setTagDouble("axis.0.wMin",0);
  tags.setTagDouble("axis.0.wMax",fabs(nSigma1));
 
  tags.setTagBool("axis.1.show",true);
  tags.setTagString("axis.1.title",TString::Format("PC %d",(int)iY));
  vals[iY] = 0; 
  tags.setTagDouble("axis.1.xMin",this->evaluateReverseProjection(iX,vals,normalize) * normX);
  tags.setTagDouble("axis.1.yMin",this->evaluateReverseProjection(iY,vals,normalize) * normY);
  vals[iY] = nSigma2; 
  tags.setTagDouble("axis.1.xMax",this->evaluateReverseProjection(iX,vals,normalize) * normX);
  tags.setTagDouble("axis.1.yMax",this->evaluateReverseProjection(iY,vals,normalize) * normY);
  vals[iY] = 0;
  tags.setTagDouble("axis.1.wMin",0);
  tags.setTagDouble("axis.1.wMax",fabs(nSigma2));

  return true;
}

TQTaggable* TQPCA::getEigenSystemTags(size_t iX, size_t iY, double nSigma1, double nSigma2, bool normalize){
  // export the TQPlotter tags required to draw the Eigen coordinate system
  TQTaggable* tags = new TQTaggable();
  this->exportEigenSystemTags(*tags,iX,iY,nSigma1,nSigma2, normalize);
  return tags;
}

TString TQPCA::getEigenSystemTagsAsString(size_t iX, size_t iY, double nSigma1,double nSigma2,bool normalize){
  // export the TQPlotter tags required to draw the Eigen coordinate system
  TQTaggable tags;
  this->exportEigenSystemTags(tags,iX,iY,nSigma1,nSigma2,normalize);
  return tags.exportTagsAsString();
}

TQTaggable* TQPCA::getEigenSystemTags(size_t iX, size_t iY, double nSigma1, double nSigma2, double normX, double normY){
  // export the TQPlotter tags required to draw the Eigen coordinate system
  TQTaggable* tags = new TQTaggable();
  this->exportEigenSystemTags(*tags,iX,iY,nSigma1,nSigma2, normX, normY);
  return tags;
}

TString TQPCA::getEigenSystemTagsAsString(size_t iX, size_t iY, double nSigma1,double nSigma2, double normX, double normY){
  // export the TQPlotter tags required to draw the Eigen coordinate system
  TQTaggable tags;
  this->exportEigenSystemTags(tags,iX,iY,nSigma1,nSigma2, normX, normY);
  return tags.exportTagsAsString();
}
