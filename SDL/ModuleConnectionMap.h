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
    static SDL::ModuleConnectionMap moduleConnectionMap() {static SDL::ModuleConnectionMap moduleConnectionMap_; return moduleConnectionMap_;}   
//    static SDL::ModuleConnectionMap moduleConnectionMap_pLStoLayer1Subdet5() {static SDL::ModuleConnectionMap moduleConnectionMap_pLStoLayer1Subdet5_; return moduleConnectionMap_pLStoLayer1Subdet5_;}   
    static SDL::ModuleConnectionMap moduleConnectionMap_pLStoLayer2Subdet5() {static SDL::ModuleConnectionMap moduleConnectionMap_pLStoLayer2Subdet5_; return moduleConnectionMap_pLStoLayer2Subdet5_;}   
    static SDL::ModuleConnectionMap moduleConnectionMap_pLStoLayer3Subdet5() {static SDL::ModuleConnectionMap moduleConnectionMap_pLStoLayer3Subdet5_; return moduleConnectionMap_pLStoLayer3Subdet5_;}   
    static SDL::ModuleConnectionMap moduleConnectionMap_pLStoLayer1Subdet4() {static SDL::ModuleConnectionMap moduleConnectionMap_pLStoLayer1Subdet4_; return moduleConnectionMap_pLStoLayer1Subdet4_;}   
    static SDL::ModuleConnectionMap moduleConnectionMap_pLStoLayer2Subdet4() {static SDL::ModuleConnectionMap moduleConnectionMap_pLStoLayer2Subdet4_; return moduleConnectionMap_pLStoLayer2Subdet4_;}   
    static SDL::ModuleConnectionMap moduleConnectionMap_pLStoLayer3Subdet4() {static SDL::ModuleConnectionMap moduleConnectionMap_pLStoLayer3Subdet4_; return moduleConnectionMap_pLStoLayer3Subdet4_;}   
    static SDL::ModuleConnectionMap moduleConnectionMap_pLStoLayer4Subdet4() {static SDL::ModuleConnectionMap moduleConnectionMap_pLStoLayer4Subdet4_; return moduleConnectionMap_pLStoLayer4Subdet4_;}   
    static SDL::ModuleConnectionMap moduleConnectionMap_pLStoLayer1Subdet5_pos() {static SDL::ModuleConnectionMap moduleConnectionMap_pLStoLayer1Subdet5_pos_; return moduleConnectionMap_pLStoLayer1Subdet5_pos_;}   
    static SDL::ModuleConnectionMap moduleConnectionMap_pLStoLayer2Subdet5_pos() {static SDL::ModuleConnectionMap moduleConnectionMap_pLStoLayer2Subdet5_pos_; return moduleConnectionMap_pLStoLayer2Subdet5_pos_;}   
    static SDL::ModuleConnectionMap moduleConnectionMap_pLStoLayer3Subdet5_pos() {static SDL::ModuleConnectionMap moduleConnectionMap_pLStoLayer3Subdet5_pos_; return moduleConnectionMap_pLStoLayer3Subdet5_pos_;}   
    static SDL::ModuleConnectionMap moduleConnectionMap_pLStoLayer1Subdet4_pos() {static SDL::ModuleConnectionMap moduleConnectionMap_pLStoLayer1Subdet4_pos_; return moduleConnectionMap_pLStoLayer1Subdet4_pos_;}   
    static SDL::ModuleConnectionMap moduleConnectionMap_pLStoLayer2Subdet4_pos() {static SDL::ModuleConnectionMap moduleConnectionMap_pLStoLayer2Subdet4_pos_; return moduleConnectionMap_pLStoLayer2Subdet4_pos_;}   
    static SDL::ModuleConnectionMap moduleConnectionMap_pLStoLayer3Subdet4_pos() {static SDL::ModuleConnectionMap moduleConnectionMap_pLStoLayer3Subdet4_pos_; return moduleConnectionMap_pLStoLayer3Subdet4_pos_;}   
    static SDL::ModuleConnectionMap moduleConnectionMap_pLStoLayer4Subdet4_pos() {static SDL::ModuleConnectionMap moduleConnectionMap_pLStoLayer4Subdet4_pos_; return moduleConnectionMap_pLStoLayer4Subdet4_pos_;}   

    static SDL::ModuleConnectionMap moduleConnectionMap_pLStoLayer1Subdet5_neg() {static SDL::ModuleConnectionMap moduleConnectionMap_pLStoLayer1Subdet5_neg_; return moduleConnectionMap_pLStoLayer1Subdet5_neg_;}   
    static SDL::ModuleConnectionMap moduleConnectionMap_pLStoLayer2Subdet5_neg() {static SDL::ModuleConnectionMap moduleConnectionMap_pLStoLayer2Subdet5_neg_; return moduleConnectionMap_pLStoLayer2Subdet5_neg_;}   
    static SDL::ModuleConnectionMap moduleConnectionMap_pLStoLayer3Subdet5_neg() {static SDL::ModuleConnectionMap moduleConnectionMap_pLStoLayer3Subdet5_neg_; return moduleConnectionMap_pLStoLayer3Subdet5_neg_;}   
    static SDL::ModuleConnectionMap moduleConnectionMap_pLStoLayer1Subdet4_neg() {static SDL::ModuleConnectionMap moduleConnectionMap_pLStoLayer1Subdet4_neg_; return moduleConnectionMap_pLStoLayer1Subdet4_neg_;}   
    static SDL::ModuleConnectionMap moduleConnectionMap_pLStoLayer2Subdet4_neg() {static SDL::ModuleConnectionMap moduleConnectionMap_pLStoLayer2Subdet4_neg_; return moduleConnectionMap_pLStoLayer2Subdet4_neg_;}   
    static SDL::ModuleConnectionMap moduleConnectionMap_pLStoLayer3Subdet4_neg() {static SDL::ModuleConnectionMap moduleConnectionMap_pLStoLayer3Subdet4_neg_; return moduleConnectionMap_pLStoLayer3Subdet4_neg_;}   
    static SDL::ModuleConnectionMap moduleConnectionMap_pLStoLayer4Subdet4_neg() {static SDL::ModuleConnectionMap moduleConnectionMap_pLStoLayer4Subdet4_neg_; return moduleConnectionMap_pLStoLayer4Subdet4_neg_;}   

//    extern ModuleConnectionMap moduleConnectionMap;
    extern ModuleConnectionMap moduleConnectionMap_pLStoLayer1Subdet5;
//    extern ModuleConnectionMap moduleConnectionMap_pLStoLayer2Subdet5;
//    extern ModuleConnectionMap moduleConnectionMap_pLStoLayer3Subdet5;
//    extern ModuleConnectionMap moduleConnectionMap_pLStoLayer1Subdet4;
//    extern ModuleConnectionMap moduleConnectionMap_pLStoLayer2Subdet4;
//    extern ModuleConnectionMap moduleConnectionMap_pLStoLayer3Subdet4;
//    extern ModuleConnectionMap moduleConnectionMap_pLStoLayer4Subdet4;
//    extern ModuleConnectionMap moduleConnectionMap_pLStoLayer1Subdet5_pos;
//    extern ModuleConnectionMap moduleConnectionMap_pLStoLayer2Subdet5_pos;
//    extern ModuleConnectionMap moduleConnectionMap_pLStoLayer3Subdet5_pos;
//    extern ModuleConnectionMap moduleConnectionMap_pLStoLayer1Subdet4_pos;
//    extern ModuleConnectionMap moduleConnectionMap_pLStoLayer2Subdet4_pos;
//    extern ModuleConnectionMap moduleConnectionMap_pLStoLayer3Subdet4_pos;
//    extern ModuleConnectionMap moduleConnectionMap_pLStoLayer4Subdet4_pos;
//    extern ModuleConnectionMap moduleConnectionMap_pLStoLayer1Subdet5_neg;
//    extern ModuleConnectionMap moduleConnectionMap_pLStoLayer2Subdet5_neg;
//    extern ModuleConnectionMap moduleConnectionMap_pLStoLayer3Subdet5_neg;
//    extern ModuleConnectionMap moduleConnectionMap_pLStoLayer1Subdet4_neg;
//    extern ModuleConnectionMap moduleConnectionMap_pLStoLayer2Subdet4_neg;
//    extern ModuleConnectionMap moduleConnectionMap_pLStoLayer3Subdet4_neg;
//    extern ModuleConnectionMap moduleConnectionMap_pLStoLayer4Subdet4_neg;
}

#endif
