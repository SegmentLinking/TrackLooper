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

            const std::vector<unsigned int>& getConnectedModuleDetIds(unsigned int detid) const;
            int size() const;

    };

    class ModuleConnectionMapManager {
    public:
        static ModuleConnectionMapManager& getInstance();
        ModuleConnectionMap& getmoduleConnectionMap();
        ModuleConnectionMap& getmoduleConnectionMap_pLStoLayer1Subdet5();
        ModuleConnectionMap& getmoduleConnectionMap_pLStoLayer2Subdet5();
        ModuleConnectionMap& getmoduleConnectionMap_pLStoLayer1Subdet4();
        ModuleConnectionMap& getmoduleConnectionMap_pLStoLayer2Subdet4();
        ModuleConnectionMap& getmoduleConnectionMap_pLStoLayer1Subdet5_pos();
        ModuleConnectionMap& getmoduleConnectionMap_pLStoLayer2Subdet5_pos();
        ModuleConnectionMap& getmoduleConnectionMap_pLStoLayer1Subdet4_pos();
        ModuleConnectionMap& getmoduleConnectionMap_pLStoLayer2Subdet4_pos();
        ModuleConnectionMap& getmoduleConnectionMap_pLStoLayer1Subdet5_neg();
        ModuleConnectionMap& getmoduleConnectionMap_pLStoLayer2Subdet5_neg();
        ModuleConnectionMap& getmoduleConnectionMap_pLStoLayer1Subdet4_neg();
        ModuleConnectionMap& getmoduleConnectionMap_pLStoLayer2Subdet4_neg();

    private:
        ModuleConnectionMapManager(); // Private constructor to enforce Singleton pattern
        ModuleConnectionMap moduleConnectionMap;
        ModuleConnectionMap moduleConnectionMap_pLStoLayer1Subdet5;
        ModuleConnectionMap moduleConnectionMap_pLStoLayer2Subdet5;
        ModuleConnectionMap moduleConnectionMap_pLStoLayer1Subdet4;
        ModuleConnectionMap moduleConnectionMap_pLStoLayer2Subdet4;
        ModuleConnectionMap moduleConnectionMap_pLStoLayer1Subdet5_pos;
        ModuleConnectionMap moduleConnectionMap_pLStoLayer2Subdet5_pos;
        ModuleConnectionMap moduleConnectionMap_pLStoLayer1Subdet4_pos;
        ModuleConnectionMap moduleConnectionMap_pLStoLayer2Subdet4_pos;
        ModuleConnectionMap moduleConnectionMap_pLStoLayer1Subdet5_neg;
        ModuleConnectionMap moduleConnectionMap_pLStoLayer2Subdet5_neg;
        ModuleConnectionMap moduleConnectionMap_pLStoLayer1Subdet4_neg;
        ModuleConnectionMap moduleConnectionMap_pLStoLayer2Subdet4_neg;
    };

}

#endif
