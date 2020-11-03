#ifndef TiltedGeometry_h
#define TiltedGeometry_h

#include <vector>
#include <map>
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>

namespace SDL
{
    class TiltedGeometry
    {

        private:
            std::map<unsigned int, float> drdzs_;
            std::map<unsigned int, float> slopes_;

        public:
            TiltedGeometry();
            TiltedGeometry(std::string filename);
            ~TiltedGeometry();

            void load(std::string);

            float getDrDz(unsigned int detid);
            float getSlope(unsigned int detid);

    };

        extern TiltedGeometry tiltedGeometry;
}

#endif
