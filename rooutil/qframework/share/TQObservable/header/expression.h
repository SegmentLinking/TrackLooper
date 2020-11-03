protected:
  TString fExpression = "";

public:
  virtual bool hasExpression() const override;
  virtual const TString& getExpression() const override;
  virtual void setExpression(const TString& expr) override;

  ObsName();
  ObsName(const TString& expression);
  virtual ~ObsName();
