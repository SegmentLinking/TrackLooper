#ifndef TiltedGeometry_h
#define TiltedGeometry_h

#include <vector>
#include <map>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <stdexcept>

namespace SDL {
  class TiltedGeometry {
  private:
    std::map<unsigned int, float> drdzs_;  // dr/dz slope
    std::map<unsigned int, float> dxdys_;  // dx/dy slope

  public:
    TiltedGeometry() = default;
    TiltedGeometry(std::string filename);
    ~TiltedGeometry() = default;

    void load(std::string);

    float getDrDz(unsigned int detid);
    float getDxDy(unsigned int detid);
  };

  namespace globals {
    extern TiltedGeometry tiltedGeometry;
  }
}  // namespace SDL

#endif
