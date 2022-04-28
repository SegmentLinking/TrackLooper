#ifndef ModuleConnectionMap_h
#define ModuleConnectionMap_h

#include <iostream>
#include <fstream>
#include <vector>
#include <map>
#include <sstream>
#include <algorithm>

namespace SDL
{
    class ModuleConnectionMap
    {

        private:
            std::map<unsigned int, std::vector<unsigned int>> moduleConnections_;

        public:
            ModuleConnectionMap();
            ModuleConnectionMap(std::string filename);
            ~ModuleConnectionMap();

            void load(std::string);
            void add(std::string);
            void print();

            const std::vector<unsigned int>& getConnectedModuleDetIds(unsigned int detid);
            /*const*/ int size();

    };

    extern ModuleConnectionMap moduleConnectionMap;
    extern ModuleConnectionMap moduleConnectionMap_pLStoLayer1Subdet5;
    extern ModuleConnectionMap moduleConnectionMap_pLStoLayer2Subdet5;
    extern ModuleConnectionMap moduleConnectionMap_pLStoLayer3Subdet5;
    extern ModuleConnectionMap moduleConnectionMap_pLStoLayer1Subdet4;
    extern ModuleConnectionMap moduleConnectionMap_pLStoLayer2Subdet4;
    extern ModuleConnectionMap moduleConnectionMap_pLStoLayer3Subdet4;
    extern ModuleConnectionMap moduleConnectionMap_pLStoLayer4Subdet4;
    extern ModuleConnectionMap moduleConnectionMap_pLStoLayer1Subdet5_pos;
    extern ModuleConnectionMap moduleConnectionMap_pLStoLayer2Subdet5_pos;
    extern ModuleConnectionMap moduleConnectionMap_pLStoLayer3Subdet5_pos;
    extern ModuleConnectionMap moduleConnectionMap_pLStoLayer1Subdet4_pos;
    extern ModuleConnectionMap moduleConnectionMap_pLStoLayer2Subdet4_pos;
    extern ModuleConnectionMap moduleConnectionMap_pLStoLayer3Subdet4_pos;
    extern ModuleConnectionMap moduleConnectionMap_pLStoLayer4Subdet4_pos;
    extern ModuleConnectionMap moduleConnectionMap_pLStoLayer1Subdet5_neg;
    extern ModuleConnectionMap moduleConnectionMap_pLStoLayer2Subdet5_neg;
    extern ModuleConnectionMap moduleConnectionMap_pLStoLayer3Subdet5_neg;
    extern ModuleConnectionMap moduleConnectionMap_pLStoLayer1Subdet4_neg;
    extern ModuleConnectionMap moduleConnectionMap_pLStoLayer2Subdet4_neg;
    extern ModuleConnectionMap moduleConnectionMap_pLStoLayer3Subdet4_neg;
    extern ModuleConnectionMap moduleConnectionMap_pLStoLayer4Subdet4_neg;
}

#endif
