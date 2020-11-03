#include "QFramework/TQObservable.h"

class ObsName : public TQObservable {
protected:
  // put here any data members your class might need
  
public:
  virtual double getValue() const override;
  virtual TObjArray* getBranchNames() const override;
  virtual Long64_t getCurrentEntry() const override;
