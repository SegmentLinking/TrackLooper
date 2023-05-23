#ifndef NeuralNetworkWeights_cuh
#define NeuralNetworkWeights_cuh

#ifdef __CUDACC__
#define CUDA_CONST_VAR __device__
#else
#define CUDA_CONST_VAR
#endif


namespace T5DNN
{
    extern CUDA_CONST_VAR const float bias_0[32];
    extern CUDA_CONST_VAR const float wgtT_0[38][32];
    extern CUDA_CONST_VAR const float bias_2[32];
    extern CUDA_CONST_VAR const float wgtT_2[32][32];
    extern CUDA_CONST_VAR const float bias_4[1];
    extern CUDA_CONST_VAR const float wgtT_4[32][1];
    extern CUDA_CONST_VAR const float working_point;
}
#endif
