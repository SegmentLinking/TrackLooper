# include "Module.cuh"
std::map <unsigned int, unsigned int> *SDL::detIdToIndex;

void SDL::createModulesInUnifiedMemory(struct modules& modulesInGPU,unsigned int nModules)
{
    /* modules stucture object will be created in Event.cu*/
    cudaMallocManaged(&(modulesInGPU.detIds),nModules * sizeof(unsigned int));
    cudaMallocManaged(&modulesInGPU.moduleMap,nModules * 40 * sizeof(unsigned int));
    cudaMallocManaged(&modulesInGPU.nConnectedModules,nModules * sizeof(unsigned int));
    cudaMallocManaged(&modulesInGPU.drdzs,nModules * sizeof(float));
    cudaMallocManaged(&modulesInGPU.slopes,nModules * sizeof(float));
    cudaMallocManaged(&modulesInGPU.nModules,sizeof(unsigned int));
    cudaMallocManaged(&modulesInGPU.nLowerModules,sizeof(unsigned int));
    cudaMallocManaged(&modulesInGPU.layers,nModules * sizeof(short));
    cudaMallocManaged(&modulesInGPU.rings,nModules * sizeof(short));
    cudaMallocManaged(&modulesInGPU.modules,nModules * sizeof(short));
    cudaMallocManaged(&modulesInGPU.rods,nModules * sizeof(short));
    cudaMallocManaged(&modulesInGPU.subdets,nModules * sizeof(short));
    cudaMallocManaged(&modulesInGPU.sides,nModules * sizeof(short));
    cudaMallocManaged(&modulesInGPU.isInverted, nModules * sizeof(bool));
    cudaMallocManaged(&modulesInGPU.isLower, nModules * sizeof(bool));

    cudaMallocManaged(&modulesInGPU.hitRanges,nModules * 2 * sizeof(int));
    cudaMallocManaged(&modulesInGPU.mdRanges,nModules * 2 * sizeof(int));
    cudaMallocManaged(&modulesInGPU.segmentRanges,nModules * 2 * sizeof(int));
    cudaMallocManaged(&modulesInGPU.trackletRanges,nModules * 2 * sizeof(int));
    cudaMallocManaged(&modulesInGPU.tripletRanges,nModules * 2 * sizeof(int));
    cudaMallocManaged(&modulesInGPU.trackCandidateRanges, nModules * 2 * sizeof(int));

    cudaMallocManaged(&modulesInGPU.moduleType,nModules * sizeof(ModuleType));
    cudaMallocManaged(&modulesInGPU.moduleLayerType,nModules * sizeof(ModuleLayerType));

    *modulesInGPU.nModules = nModules;

}

void SDL::freeModulesInUnifiedMemory(struct modules& modulesInGPU)
{
  cudaFree(modulesInGPU.detIds);
  cudaFree(modulesInGPU.moduleMap);
  cudaFree(modulesInGPU.nConnectedModules);
  cudaFree(modulesInGPU.drdzs);
  cudaFree(modulesInGPU.slopes);
  cudaFree(modulesInGPU.nModules);
  cudaFree(modulesInGPU.nLowerModules);
  cudaFree(modulesInGPU.layers);
  cudaFree(modulesInGPU.rings);
  cudaFree(modulesInGPU.modules);
  cudaFree(modulesInGPU.rods);
  cudaFree(modulesInGPU.subdets);
  cudaFree(modulesInGPU.sides);
  cudaFree(modulesInGPU.isInverted);
  cudaFree(modulesInGPU.isLower);
  cudaFree(modulesInGPU.hitRanges);
  cudaFree(modulesInGPU.mdRanges);
  cudaFree(modulesInGPU.segmentRanges);
  cudaFree(modulesInGPU.trackletRanges);
  cudaFree(modulesInGPU.tripletRanges);
  cudaFree(modulesInGPU.trackCandidateRanges);
  cudaFree(modulesInGPU.moduleType);
  cudaFree(modulesInGPU.moduleLayerType);
  cudaFree(modulesInGPU.lowerModuleIndices);
  cudaFree(modulesInGPU.reverseLookupLowerModuleIndices);
}

void SDL::createLowerModuleIndexMap(struct modules& modulesInGPU, unsigned int nLowerModules, unsigned int nModules)
{
    //FIXME:some hacks to get the pixel module in hte lower modules index without incrementing nLowerModules counter!
    //Reproduce these hacks in the explicit memory for identical results (or come up with a better method)
    cudaMallocManaged(&modulesInGPU.lowerModuleIndices,(nLowerModules + 1) * sizeof(unsigned int));
    cudaMallocManaged(&modulesInGPU.reverseLookupLowerModuleIndices,nModules * sizeof(int));

    unsigned int lowerModuleCounter = 0;
    for(auto it = (*detIdToIndex).begin(); it != (*detIdToIndex).end(); it++)
    {
        unsigned int index = it->second;
        unsigned int detId = it->first;
        if(modulesInGPU.isLower[index])
        {
            modulesInGPU.lowerModuleIndices[lowerModuleCounter] = index;
            modulesInGPU.reverseLookupLowerModuleIndices[index] = lowerModuleCounter;
            lowerModuleCounter++;
        }
        else
        {
            modulesInGPU.reverseLookupLowerModuleIndices[index] = -1;
        }
    }
    //hacky stuff "beyond the index" for the pixel module. nLowerModules will *NOT* cover the pixel module!
    modulesInGPU.lowerModuleIndices[nLowerModules] = (*detIdToIndex)[1];
    modulesInGPU.reverseLookupLowerModuleIndices[(*detIdToIndex)[1]] = nLowerModules;

}

void SDL::loadModulesFromFile(struct modules& modulesInGPU, unsigned int& nModules)
{
    detIdToIndex = new std::map<unsigned int, unsigned int>;

    /*modules structure object will be created in Event.cu*/
    /* Load the whole text file into the unordered_map first*/

    std::ifstream ifile;
    ifile.open("data/centroid.txt");
    if(!ifile.is_open())
    {
        std::cout<<"ERROR! module list file not present!"<<std::endl;
    }
    std::string line;
    unsigned int counter = 0;
    unsigned int lowerModuleCounter = 0;

    while(std::getline(ifile,line))
    {
        std::stringstream ss(line);
        std::string token;
        bool flag = 0;

        while(std::getline(ss,token,','))
        {
            if(flag == 1) break;
            (*detIdToIndex)[stoi(token)] = counter;
            flag = 1;
            counter++;
        }
    }
    //MANUAL INSERTION OF PIXEL MODULE!
    (*detIdToIndex)[1] = counter; //pixel module is the last module in the module list
    counter++;
    nModules = counter;
    std::cout<<"Number of modules = "<<nModules<<std::endl;

    createModulesInUnifiedMemory(modulesInGPU,nModules);

    for(auto it = (*detIdToIndex).begin(); it != (*detIdToIndex).end(); it++)
    {
        unsigned int detId = it->first;
        unsigned int index = it->second;
        modulesInGPU.detIds[index] = detId;
        if(detId == 1)
        {
            modulesInGPU.layers[index] = 0;
            modulesInGPU.rings[index] = 0;
            modulesInGPU.rods[index] = 0;
            modulesInGPU.modules[index] = 0;
            modulesInGPU.subdets[index] = SDL::InnerPixel;
            modulesInGPU.sides[index] = 0;
            modulesInGPU.isInverted[index] = 0;
            modulesInGPU.isLower[index] = false;
            modulesInGPU.moduleType[index] = PixelModule;
            modulesInGPU.moduleLayerType[index] = SDL::InnerPixelLayer;
            modulesInGPU.slopes[index] = 0;
            modulesInGPU.drdzs[index] = 0;
        }
        else
        {
            unsigned short layer,ring,rod,module,subdet,side;
            setDerivedQuantities(detId,layer,ring,rod,module,subdet,side);
            modulesInGPU.layers[index] = layer;
            modulesInGPU.rings[index] = ring;
            modulesInGPU.rods[index] = rod;
            modulesInGPU.modules[index] = module;
            modulesInGPU.subdets[index] = subdet;
            modulesInGPU.sides[index] = side;

            modulesInGPU.isInverted[index] = modulesInGPU.parseIsInverted(index);
            modulesInGPU.isLower[index] = modulesInGPU.parseIsLower(index);

            modulesInGPU.moduleType[index] = modulesInGPU.parseModuleType(index);
            modulesInGPU.moduleLayerType[index] = modulesInGPU.parseModuleLayerType(index);

            modulesInGPU.slopes[index] = (subdet == Endcap) ? endcapGeometry.getSlopeLower(detId) : tiltedGeometry.getSlope(detId);
            modulesInGPU.drdzs[index] = (subdet == Barrel) ? tiltedGeometry.getDrDz(detId) : 0;
        }
        if(modulesInGPU.isLower[index]) lowerModuleCounter++;
    }

    *modulesInGPU.nLowerModules = lowerModuleCounter;
    std::cout<<"number of lower modules (without fake pixel module)= "<<*modulesInGPU.nLowerModules<<std::endl;
    createLowerModuleIndexMap(modulesInGPU,lowerModuleCounter, nModules);
    fillConnectedModuleArray(modulesInGPU,nModules);
    resetObjectRanges(modulesInGPU,nModules);
}

void SDL::fillConnectedModuleArray(struct modules& modulesInGPU, unsigned int nModules)
{
    for(auto it = (*detIdToIndex).begin(); it != (*detIdToIndex).end(); ++it)
    {
        unsigned int detId = it->first;
        unsigned int index = it->second;
        auto& connectedModules = moduleConnectionMap.getConnectedModuleDetIds(detId);
        modulesInGPU.nConnectedModules[index] = connectedModules.size();
        for(unsigned int i = 0; i< modulesInGPU.nConnectedModules[index];i++)
        {
            modulesInGPU.moduleMap[index * 40 + i] = (*detIdToIndex)[connectedModules[i]];
        }
    }
}

void SDL::setDerivedQuantities(unsigned int detId, unsigned short& layer, unsigned short& ring, unsigned short& rod, unsigned short& module, unsigned short& subdet, unsigned short& side)
{
    subdet = (detId & (7 << 25)) >> 25;
    side = (subdet == Endcap) ? (detId & (3 << 23)) >> 23 : (detId & (3 << 18)) >> 18;
    layer = (subdet == Endcap) ? (detId & (7 << 18)) >> 18 : (detId & (7 << 20)) >> 20;
    ring = (subdet == Endcap) ? (detId & (15 << 12)) >> 12 : 0;
    module = (detId & (127 << 2)) >> 2;
    rod = (subdet == Endcap) ? 0 : (detId & (127 << 10)) >> 10;
}

//auxilliary functions - will be called as needed
bool SDL::modules::parseIsInverted(unsigned int index)
{
    if (subdets[index] == Endcap)
    {
        if (sides[index] == NegZ)
        {
            return modules[index] % 2 == 1;
        }
        else if (sides[index] == PosZ)
        {
            return modules[index] % 2 == 0;
        }
        else
        {
            return 0;
        }
    }
    else if (subdets[index] == Barrel)
    {
        if (sides[index] == Center)
        {
            if (layers[index] <= 3)
            {
                return modules[index] % 2 == 1;
            }
            else if (layers[index] >= 4)
            {
                return modules[index] % 2 == 0;
            }
            else
            {
                return 0;
            }
        }
        else if (sides[index] == NegZ or sides[index] == PosZ)
        {
            if (layers[index] <= 2)
            {
                return modules[index] % 2 == 1;
            }
            else if (layers[index] == 3)
            {
                return modules[index] % 2 == 0;
            }
            else
            {
                return 0;
            }
        }
        else
        {
            return 0;
        }
    }
    else
    {
        return 0;
    }
}

bool SDL::modules::parseIsLower(unsigned int index)
{
    return (isInverted[index]) ? !(detIds[index] & 1) : (detIds[index] & 1);
}

unsigned int SDL::modules::partnerModuleIndex(unsigned int index)
{
    /*We need to ensure modules with successive det Ids are right next to each other
    or we're dead*/
    if(isLower[index])
    {
        return (isInverted[index] ? index - 1: index + 1);
    }
    else
    {
        return (isInverted[index] ? index + 1 : index - 1);
    }
}

SDL::ModuleType SDL::modules::parseModuleType(unsigned int index)
{
    if(subdets[index] == Barrel)
    {
        if(layers[index] <= 3)
        {
            return PS;
        }
        else
        {
            return TwoS;
        }
    }
    else
    {
        if(layers[index] <= 2)
        {
            if(rings[index] <= 10)
            {
                return PS;
            }
            else
            {
                return TwoS;
            }
        }
        else
        {
            if(rings[index] <= 7)
            {
                return PS;
            }
            else
            {
                return TwoS;
            }
        }
    }
}

SDL::ModuleLayerType SDL::modules::parseModuleLayerType(unsigned int index)
{
    if(moduleType[index] == TwoS)
    {
        return Strip;
    }
    if(isInverted[index])
    {
        if(isLower[index])
        {
            return Strip;
        }
        else
        {
            return Pixel;
        }
    }
    else
   {
        if(isLower[index])
        {
            return Pixel;
        }
        else
        {
            return Strip;
        }
    }
}

void SDL::resetObjectRanges(struct modules& modulesInGPU, unsigned int nModules)
{

#pragma omp parallel for default(shared)
    for(size_t i = 0; i<nModules *2; i++)
    {
        modulesInGPU.hitRanges[i] = -1;
        modulesInGPU.mdRanges[i] = -1;
        modulesInGPU.segmentRanges[i] = -1;
        modulesInGPU.trackletRanges[i] = -1;
        modulesInGPU.tripletRanges[i] = -1;
        modulesInGPU.trackCandidateRanges[i] = -1;
    }

}
