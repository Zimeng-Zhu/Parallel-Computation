#include <assert.h>
#include <math.h>
#include "../src/utils.h"
#include "../src/types.h"
#include "mytypes.h"
using namespace std;

#include <stdio.h>

#ifdef NAIVE
__global__ void matMul(int N, _FTYPE_ *C, _FTYPE_ *A, _FTYPE_ *B) {

    int I =  blockIdx.y*blockDim.y + threadIdx.y;
    int J =  blockIdx.x*blockDim.x + threadIdx.x;

    if((I < N) && (J < N)){
        _FTYPE_ _c = 0;
        for (unsigned int k = 0; k < N; k++) {
            _FTYPE_ a = A[I * N + k];
            _FTYPE_ b = B[k * N + J];
            _c += a * b;
        }
        C[I * N + J] = _c;
    }
}

#else
__global__ void matMul(int N, _FTYPE_ *C, _FTYPE_ *A, _FTYPE_ *B) {
    // //using shared memory, no tiling

    // extern __shared__ _FTYPE_ sMem[];
    // _FTYPE_ (*As0)[TILEDIM_K] = (_FTYPE_ (*)[TILEDIM_K])sMem;
    // _FTYPE_ (*Bs0)[TILEDIM_N] = (_FTYPE_ (*)[TILEDIM_N])&sMem[TILEDIM_M * TILEDIM_K];

    // int ii = threadIdx.y, jj = threadIdx.x;

    // int idxi = blockIdx.y * TILEDIM_M + ii, idxj = blockIdx.x * TILEDIM_N + jj;

    // _FTYPE_ Cij = 0;

    // #pragma unroll
    // for (int k = 0; k < N; k += TILEDIM_K)
    // {
    //     int kb = (N - k) < TILEDIM_K? (N - k) : TILEDIM_K;

    //     if (jj < kb)
    //         As0[ii][jj] = idxi < N ? A[idxi * N + k + jj] : 0;
    //     if (ii < kb)
    //         Bs0[ii][jj] = idxj < N ? B[(k + ii) * N + idxj] : 0;
    //     __syncthreads();

    //     #pragma unroll
    //     for (int kk = 0; kk < kb; kk++)
    //         Cij += As0[ii][kk] * Bs0[kk][jj];
    //     __syncthreads();
    // }
    // if (idxi < N && idxj < N)
    //     C[idxi * N + idxj] = Cij;


////////////////////////////////////////////////////////////////////////////////////////////////////////  
    // //using shared memory and register, 2-D tiling, 2x2

    // extern __shared__ _FTYPE_ sMem[];
    // _FTYPE_ (*As0)[TILEDIM_K] = (_FTYPE_ (*)[TILEDIM_K])sMem;
    // _FTYPE_ (*Bs0)[TILEDIM_N] = (_FTYPE_ (*)[TILEDIM_N])&sMem[TILEDIM_M * TILEDIM_K];
    // _FTYPE_ (*As1)[TILEDIM_K] = (_FTYPE_ (*)[TILEDIM_K])&sMem[TILEDIM_M * TILEDIM_K + TILEDIM_K * TILEDIM_N];
    // _FTYPE_ (*Bs1)[TILEDIM_N] = (_FTYPE_ (*)[TILEDIM_N])&sMem[TILEDIM_M * TILEDIM_K * 2 + TILEDIM_K * TILEDIM_N];

    // int ii = threadIdx.y, jj = threadIdx.x;

    // int idxi[2], idxj[2];
    // idxi[0] = blockIdx.y * TILEDIM_M * 2 + ii;
    // idxi[1] = idxi[0] + TILEDIM_M;
    // idxj[0] = blockIdx.x * TILEDIM_N * 2 + jj;
    // idxj[1] = idxj[0] + TILEDIM_N;

    // _FTYPE_ Ar[2];
    // _FTYPE_ Br[2];
    // _FTYPE_ Cij[4] = {0};

    // #pragma unroll
    // for (int k = 0; k < N; k += TILEDIM_K)
    // {
    //     int kb = (N - k) < TILEDIM_K? (N - k) : TILEDIM_K;

    //     if (jj < kb)
    //     {
    //         As0[ii][jj] = idxi[0] < N ? A[idxi[0] * N + k + jj] : 0;
    //         As1[ii][jj] = idxi[1] < N ? A[idxi[1] * N + k + jj] : 0;
    //     }
    //     if (ii < kb)
    //     {
    //         Bs0[ii][jj] = idxj[0] < N ? B[(k + ii) * N + idxj[0]] : 0;
    //         Bs1[ii][jj] = idxj[1] < N ? B[(k + ii) * N + idxj[1]] : 0;
    //     }
    //     __syncthreads();

    //     #pragma unroll
    //     for (int kk = 0; kk < kb; kk++)
    //     {
    //         Ar[0] = As0[ii][kk];
    //         Ar[1] = As1[ii][kk];
    //         Br[0] = Bs0[kk][jj];
    //         Br[1] = Bs1[kk][jj];

    //         Cij[0] += Ar[0] * Br[0];
    //         Cij[1] += Ar[0] * Br[1];
    //         Cij[2] += Ar[1] * Br[0];
    //         Cij[3] += Ar[1] * Br[1];
    //     }
    //     __syncthreads();
    // }
    // if (idxi[0] < N)
    // {
    //     if (idxj[0] < N)
    //         C[idxi[0] * N + idxj[0]] = Cij[0];
    //     if (idxj[1] < N)
    //         C[idxi[0] * N + idxj[1]] = Cij[1];  
    // }
    // if (idxi[1] < N)
    // {
    //     if (idxj[0] < N)
    //         C[idxi[1] * N + idxj[0]] = Cij[2];
    //     if (idxj[1] < N)
    //         C[idxi[1] * N + idxj[1]] = Cij[3];  
    // } 


////////////////////////////////////////////////////////////////////////////////////////////////////////
    // //using shared memory and register, 2-D tiling, 4x4

    // extern __shared__ _FTYPE_ sMem[];
    // _FTYPE_ (*As0)[TILEDIM_K] = (_FTYPE_ (*)[TILEDIM_K])sMem;
    // _FTYPE_ (*Bs0)[TILEDIM_N] = (_FTYPE_ (*)[TILEDIM_N])&sMem[TILEDIM_M * TILEDIM_K];
    // _FTYPE_ (*As1)[TILEDIM_K] = (_FTYPE_ (*)[TILEDIM_K])&sMem[TILEDIM_M * TILEDIM_K + TILEDIM_K * TILEDIM_N];
    // _FTYPE_ (*Bs1)[TILEDIM_N] = (_FTYPE_ (*)[TILEDIM_N])&sMem[TILEDIM_M * TILEDIM_K * 2 + TILEDIM_K * TILEDIM_N];
    // _FTYPE_ (*As2)[TILEDIM_K] = (_FTYPE_ (*)[TILEDIM_K])&sMem[TILEDIM_M * TILEDIM_K * 2 + TILEDIM_K * TILEDIM_N * 2];
    // _FTYPE_ (*Bs2)[TILEDIM_N] = (_FTYPE_ (*)[TILEDIM_N])&sMem[TILEDIM_M * TILEDIM_K * 3 + TILEDIM_K * TILEDIM_N * 2];
    // _FTYPE_ (*As3)[TILEDIM_K] = (_FTYPE_ (*)[TILEDIM_K])&sMem[TILEDIM_M * TILEDIM_K * 3 + TILEDIM_K * TILEDIM_N * 3];
    // _FTYPE_ (*Bs3)[TILEDIM_N] = (_FTYPE_ (*)[TILEDIM_N])&sMem[TILEDIM_M * TILEDIM_K * 4 + TILEDIM_K * TILEDIM_N * 3]; 

    // int ii = threadIdx.y, jj = threadIdx.x;

    // int idxi[4], idxj[4];
    // idxi[0] = blockIdx.y * TILEDIM_M * 4 + ii;
    // idxi[1] = idxi[0] + TILEDIM_M;
    // idxi[2] = idxi[1] + TILEDIM_M;
    // idxi[3] = idxi[2] + TILEDIM_M;
    // idxj[0] = blockIdx.x * TILEDIM_N * 4 + jj;
    // idxj[1] = idxj[0] + TILEDIM_N;
    // idxj[2] = idxj[1] + TILEDIM_N;
    // idxj[3] = idxj[2] + TILEDIM_N;

    // _FTYPE_ Ar[4];
    // _FTYPE_ Br[4];
    // _FTYPE_ Cij[16] = {0};

    // #pragma unroll
    // for (int k = 0; k < N; k += TILEDIM_K)
    // {
    //     int kb = (N - k) < TILEDIM_K? (N - k) : TILEDIM_K;

    //     if (jj < kb)
    //     {
    //         As0[ii][jj] = idxi[0] < N ? A[idxi[0] * N + k + jj] : 0;
    //         As1[ii][jj] = idxi[1] < N ? A[idxi[1] * N + k + jj] : 0;
    //         As2[ii][jj] = idxi[2] < N ? A[idxi[2] * N + k + jj] : 0;
    //         As3[ii][jj] = idxi[3] < N ? A[idxi[3] * N + k + jj] : 0;
    //     }
    //     if (ii < kb)
    //     {
    //         Bs0[ii][jj] = idxj[0] < N ? B[(k + ii) * N + idxj[0]] : 0;
    //         Bs1[ii][jj] = idxj[1] < N ? B[(k + ii) * N + idxj[1]] : 0;
    //         Bs2[ii][jj] = idxj[2] < N ? B[(k + ii) * N + idxj[2]] : 0;
    //         Bs3[ii][jj] = idxj[3] < N ? B[(k + ii) * N + idxj[3]] : 0;
    //     }
    //     __syncthreads();

    //     #pragma unroll
    //     for (int kk = 0; kk < kb; kk++)
    //     {
    //         Ar[0] = As0[ii][kk];
    //         Ar[1] = As1[ii][kk];
    //         Ar[2] = As2[ii][kk];
    //         Ar[3] = As3[ii][kk];
    //         Br[0] = Bs0[kk][jj];
    //         Br[1] = Bs1[kk][jj];
    //         Br[2] = Bs2[kk][jj];
    //         Br[3] = Bs3[kk][jj];

    //         Cij[0] += Ar[0] * Br[0];
    //         Cij[1] += Ar[0] * Br[1];
    //         Cij[2] += Ar[0] * Br[2];
    //         Cij[3] += Ar[0] * Br[3];
    //         Cij[4] += Ar[1] * Br[0];
    //         Cij[5] += Ar[1] * Br[1];
    //         Cij[6] += Ar[1] * Br[2];
    //         Cij[7] += Ar[1] * Br[3];
    //         Cij[8] += Ar[2] * Br[0];
    //         Cij[9] += Ar[2] * Br[1];
    //         Cij[10] += Ar[2] * Br[2];
    //         Cij[11] += Ar[2] * Br[3];
    //         Cij[12] += Ar[3] * Br[0];
    //         Cij[13] += Ar[3] * Br[1];
    //         Cij[14] += Ar[3] * Br[2];
    //         Cij[15] += Ar[3] * Br[3];
    //     }
    //     __syncthreads();
    // }
    // if (idxi[0] < N)
    // {
    //     if (idxj[0] < N)
    //         C[idxi[0] * N + idxj[0]] = Cij[0];
    //     if (idxj[1] < N)
    //         C[idxi[0] * N + idxj[1]] = Cij[1];
    //     if (idxj[2] < N)
    //         C[idxi[0] * N + idxj[2]] = Cij[2];
    //     if (idxj[3] < N)
    //         C[idxi[0] * N + idxj[3]] = Cij[3];    
    // }
    // if (idxi[1] < N)
    // {
    //     if (idxj[0] < N)
    //         C[idxi[1] * N + idxj[0]] = Cij[4];
    //     if (idxj[1] < N)
    //         C[idxi[1] * N + idxj[1]] = Cij[5];
    //     if (idxj[2] < N)
    //         C[idxi[1] * N + idxj[2]] = Cij[6];
    //     if (idxj[3] < N)
    //         C[idxi[1] * N + idxj[3]] = Cij[7]; 
    // } 
    // if (idxi[2] < N)
    // {
    //     if (idxj[0] < N)
    //         C[idxi[2] * N + idxj[0]] = Cij[8];
    //     if (idxj[1] < N)
    //         C[idxi[2] * N + idxj[1]] = Cij[9];
    //     if (idxj[2] < N)
    //         C[idxi[2] * N + idxj[2]] = Cij[10];
    //     if (idxj[3] < N)
    //         C[idxi[2] * N + idxj[3]] = Cij[11]; 
    // } 
    // if (idxi[3] < N)
    // {
    //     if (idxj[0] < N)
    //         C[idxi[3] * N + idxj[0]] = Cij[12];
    //     if (idxj[1] < N)
    //         C[idxi[3] * N + idxj[1]] = Cij[13];
    //     if (idxj[2] < N)
    //         C[idxi[3] * N + idxj[2]] = Cij[14];
    //     if (idxj[3] < N)
    //         C[idxi[3] * N + idxj[3]] = Cij[15]; 
    // } 


////////////////////////////////////////////////////////////////////////////////////////////////////////
    // // using shared memory and register, 2-D tiling, 8x8

    // extern __shared__ _FTYPE_ sMem[];
    // _FTYPE_ (*As0)[TILEDIM_K] = (_FTYPE_ (*)[TILEDIM_K])sMem;
    // _FTYPE_ (*Bs0)[TILEDIM_N] = (_FTYPE_ (*)[TILEDIM_N])&sMem[TILEDIM_M * TILEDIM_K];
    // _FTYPE_ (*As1)[TILEDIM_K] = (_FTYPE_ (*)[TILEDIM_K])&sMem[TILEDIM_M * TILEDIM_K + TILEDIM_K * TILEDIM_N];
    // _FTYPE_ (*Bs1)[TILEDIM_N] = (_FTYPE_ (*)[TILEDIM_N])&sMem[TILEDIM_M * TILEDIM_K * 2 + TILEDIM_K * TILEDIM_N];
    // _FTYPE_ (*As2)[TILEDIM_K] = (_FTYPE_ (*)[TILEDIM_K])&sMem[TILEDIM_M * TILEDIM_K * 2 + TILEDIM_K * TILEDIM_N * 2];
    // _FTYPE_ (*Bs2)[TILEDIM_N] = (_FTYPE_ (*)[TILEDIM_N])&sMem[TILEDIM_M * TILEDIM_K * 3 + TILEDIM_K * TILEDIM_N * 2];
    // _FTYPE_ (*As3)[TILEDIM_K] = (_FTYPE_ (*)[TILEDIM_K])&sMem[TILEDIM_M * TILEDIM_K * 3 + TILEDIM_K * TILEDIM_N * 3];
    // _FTYPE_ (*Bs3)[TILEDIM_N] = (_FTYPE_ (*)[TILEDIM_N])&sMem[TILEDIM_M * TILEDIM_K * 4 + TILEDIM_K * TILEDIM_N * 3]; 
    // _FTYPE_ (*As4)[TILEDIM_K] = (_FTYPE_ (*)[TILEDIM_K])&sMem[TILEDIM_M * TILEDIM_K * 4 + TILEDIM_K * TILEDIM_N * 4];
    // _FTYPE_ (*Bs4)[TILEDIM_N] = (_FTYPE_ (*)[TILEDIM_N])&sMem[TILEDIM_M * TILEDIM_K * 5 + TILEDIM_K * TILEDIM_N * 4];
    // _FTYPE_ (*As5)[TILEDIM_K] = (_FTYPE_ (*)[TILEDIM_K])&sMem[TILEDIM_M * TILEDIM_K * 5 + TILEDIM_K * TILEDIM_N * 5];
    // _FTYPE_ (*Bs5)[TILEDIM_N] = (_FTYPE_ (*)[TILEDIM_N])&sMem[TILEDIM_M * TILEDIM_K * 6 + TILEDIM_K * TILEDIM_N * 5];
    // _FTYPE_ (*As6)[TILEDIM_K] = (_FTYPE_ (*)[TILEDIM_K])&sMem[TILEDIM_M * TILEDIM_K * 6 + TILEDIM_K * TILEDIM_N * 6];
    // _FTYPE_ (*Bs6)[TILEDIM_N] = (_FTYPE_ (*)[TILEDIM_N])&sMem[TILEDIM_M * TILEDIM_K * 7 + TILEDIM_K * TILEDIM_N * 6];
    // _FTYPE_ (*As7)[TILEDIM_K] = (_FTYPE_ (*)[TILEDIM_K])&sMem[TILEDIM_M * TILEDIM_K * 7 + TILEDIM_K * TILEDIM_N * 7];
    // _FTYPE_ (*Bs7)[TILEDIM_N] = (_FTYPE_ (*)[TILEDIM_N])&sMem[TILEDIM_M * TILEDIM_K * 8 + TILEDIM_K * TILEDIM_N * 7]; 
    
    // int ii = threadIdx.y, jj = threadIdx.x;

    // int idxi[8], idxj[8];
    // idxi[0] = blockIdx.y * TILEDIM_M * 8 + ii;
    // idxi[1] = idxi[0] + TILEDIM_M;
    // idxi[2] = idxi[1] + TILEDIM_M;
    // idxi[3] = idxi[2] + TILEDIM_M;
    // idxi[4] = idxi[3] + TILEDIM_M;
    // idxi[5] = idxi[4] + TILEDIM_M;
    // idxi[6] = idxi[5] + TILEDIM_M;
    // idxi[7] = idxi[6] + TILEDIM_M;
    // idxj[0] = blockIdx.x * TILEDIM_N * 8 + jj;
    // idxj[1] = idxj[0] + TILEDIM_N;
    // idxj[2] = idxj[1] + TILEDIM_N;
    // idxj[3] = idxj[2] + TILEDIM_N;
    // idxj[4] = idxj[3] + TILEDIM_N;
    // idxj[5] = idxj[4] + TILEDIM_N;
    // idxj[6] = idxj[5] + TILEDIM_N;
    // idxj[7] = idxj[6] + TILEDIM_N;

    // _FTYPE_ Ar[8];
    // _FTYPE_ Br[8];
    // _FTYPE_ Cij[64] = {0};

    // #pragma unroll
    // for (int k = 0; k < N; k += TILEDIM_K)
    // {
    //     int kb = (N - k) < TILEDIM_K? (N - k) : TILEDIM_K;

    //     if (jj < kb)
    //     {
    //         As0[ii][jj] = idxi[0] < N ? A[idxi[0] * N + k + jj] : 0;
    //         As1[ii][jj] = idxi[1] < N ? A[idxi[1] * N + k + jj] : 0;
    //         As2[ii][jj] = idxi[2] < N ? A[idxi[2] * N + k + jj] : 0;
    //         As3[ii][jj] = idxi[3] < N ? A[idxi[3] * N + k + jj] : 0;
    //         As4[ii][jj] = idxi[4] < N ? A[idxi[4] * N + k + jj] : 0;
    //         As5[ii][jj] = idxi[5] < N ? A[idxi[5] * N + k + jj] : 0;
    //         As6[ii][jj] = idxi[6] < N ? A[idxi[6] * N + k + jj] : 0;
    //         As7[ii][jj] = idxi[7] < N ? A[idxi[7] * N + k + jj] : 0;
    //     }
    //     if (ii < kb)
    //     {
    //         Bs0[ii][jj] = idxj[0] < N ? B[(k + ii) * N + idxj[0]] : 0;
    //         Bs1[ii][jj] = idxj[1] < N ? B[(k + ii) * N + idxj[1]] : 0;
    //         Bs2[ii][jj] = idxj[2] < N ? B[(k + ii) * N + idxj[2]] : 0;
    //         Bs3[ii][jj] = idxj[3] < N ? B[(k + ii) * N + idxj[3]] : 0;
    //         Bs4[ii][jj] = idxj[4] < N ? B[(k + ii) * N + idxj[4]] : 0;
    //         Bs5[ii][jj] = idxj[5] < N ? B[(k + ii) * N + idxj[5]] : 0;
    //         Bs6[ii][jj] = idxj[6] < N ? B[(k + ii) * N + idxj[6]] : 0;
    //         Bs7[ii][jj] = idxj[7] < N ? B[(k + ii) * N + idxj[7]] : 0;
    //     }
    //     __syncthreads();

    //     #pragma unroll
    //     for (int kk = 0; kk < kb; kk++)
    //     {
    //         Ar[0] = As0[ii][kk];
    //         Ar[1] = As1[ii][kk];
    //         Ar[2] = As2[ii][kk];
    //         Ar[3] = As3[ii][kk];
    //         Ar[4] = As4[ii][kk];
    //         Ar[5] = As5[ii][kk];
    //         Ar[6] = As6[ii][kk];
    //         Ar[7] = As7[ii][kk];            

    //         Br[0] = Bs0[kk][jj];
    //         Br[1] = Bs1[kk][jj];
    //         Br[2] = Bs2[kk][jj];
    //         Br[3] = Bs3[kk][jj];
    //         Br[4] = Bs4[kk][jj];
    //         Br[5] = Bs5[kk][jj];
    //         Br[6] = Bs6[kk][jj];
    //         Br[7] = Bs7[kk][jj];

    //         Cij[0] += Ar[0] * Br[0];
    //         Cij[1] += Ar[0] * Br[1];
    //         Cij[2] += Ar[0] * Br[2];
    //         Cij[3] += Ar[0] * Br[3];
    //         Cij[4] += Ar[0] * Br[4];
    //         Cij[5] += Ar[0] * Br[5];
    //         Cij[6] += Ar[0] * Br[6];
    //         Cij[7] += Ar[0] * Br[7];
    //         Cij[8] += Ar[1] * Br[0];
    //         Cij[9] += Ar[1] * Br[1];
    //         Cij[10] += Ar[1] * Br[2];
    //         Cij[11] += Ar[1] * Br[3];
    //         Cij[12] += Ar[1] * Br[4];
    //         Cij[13] += Ar[1] * Br[5];
    //         Cij[14] += Ar[1] * Br[6];
    //         Cij[15] += Ar[1] * Br[7];
    //         Cij[16] += Ar[2] * Br[0];
    //         Cij[17] += Ar[2] * Br[1];
    //         Cij[18] += Ar[2] * Br[2];
    //         Cij[19] += Ar[2] * Br[3];
    //         Cij[20] += Ar[2] * Br[4];
    //         Cij[21] += Ar[2] * Br[5];
    //         Cij[22] += Ar[2] * Br[6];
    //         Cij[23] += Ar[2] * Br[7];
    //         Cij[24] += Ar[3] * Br[0];
    //         Cij[25] += Ar[3] * Br[1];
    //         Cij[26] += Ar[3] * Br[2];
    //         Cij[27] += Ar[3] * Br[3];
    //         Cij[28] += Ar[3] * Br[4];
    //         Cij[29] += Ar[3] * Br[5];
    //         Cij[30] += Ar[3] * Br[6];
    //         Cij[31] += Ar[3] * Br[7];
    //         Cij[32] += Ar[4] * Br[0];
    //         Cij[33] += Ar[4] * Br[1];
    //         Cij[34] += Ar[4] * Br[2];
    //         Cij[35] += Ar[4] * Br[3];
    //         Cij[36] += Ar[4] * Br[4];
    //         Cij[37] += Ar[4] * Br[5];
    //         Cij[38] += Ar[4] * Br[6];
    //         Cij[39] += Ar[4] * Br[7];
    //         Cij[40] += Ar[5] * Br[0];
    //         Cij[41] += Ar[5] * Br[1];
    //         Cij[42] += Ar[5] * Br[2];
    //         Cij[43] += Ar[5] * Br[3];
    //         Cij[44] += Ar[5] * Br[4];
    //         Cij[45] += Ar[5] * Br[5];
    //         Cij[46] += Ar[5] * Br[6];
    //         Cij[47] += Ar[5] * Br[7];
    //         Cij[48] += Ar[6] * Br[0];
    //         Cij[49] += Ar[6] * Br[1];
    //         Cij[50] += Ar[6] * Br[2];
    //         Cij[51] += Ar[6] * Br[3];
    //         Cij[52] += Ar[6] * Br[4];
    //         Cij[53] += Ar[6] * Br[5];
    //         Cij[54] += Ar[6] * Br[6];
    //         Cij[55] += Ar[6] * Br[7];
    //         Cij[56] += Ar[7] * Br[0];
    //         Cij[57] += Ar[7] * Br[1];
    //         Cij[58] += Ar[7] * Br[2];
    //         Cij[59] += Ar[7] * Br[3];
    //         Cij[60] += Ar[7] * Br[4];
    //         Cij[61] += Ar[7] * Br[5];
    //         Cij[62] += Ar[7] * Br[6];
    //         Cij[63] += Ar[7] * Br[7];
    //     }
    //     __syncthreads();
    // }

    // if (idxi[0] < N)
    // {
    //     if (idxj[0] < N)
    //         C[idxi[0] * N + idxj[0]] = Cij[0];
    //     if (idxj[1] < N)
    //         C[idxi[0] * N + idxj[1]] = Cij[1];
    //     if (idxj[2] < N)
    //         C[idxi[0] * N + idxj[2]] = Cij[2];
    //     if (idxj[3] < N)
    //         C[idxi[0] * N + idxj[3]] = Cij[3]; 
    //     if (idxj[4] < N)
    //         C[idxi[0] * N + idxj[4]] = Cij[4];
    //     if (idxj[5] < N)
    //         C[idxi[0] * N + idxj[5]] = Cij[5];
    //     if (idxj[6] < N)
    //         C[idxi[0] * N + idxj[6]] = Cij[6];
    //     if (idxj[7] < N)
    //         C[idxi[0] * N + idxj[7]] = Cij[7];  
    // }
    // if (idxi[1] < N)
    // {
    //     if (idxj[0] < N)
    //         C[idxi[1] * N + idxj[0]] = Cij[8];
    //     if (idxj[1] < N)
    //         C[idxi[1] * N + idxj[1]] = Cij[9];
    //     if (idxj[2] < N)
    //         C[idxi[1] * N + idxj[2]] = Cij[10];
    //     if (idxj[3] < N)
    //         C[idxi[1] * N + idxj[3]] = Cij[11];  
    //     if (idxj[4] < N)
    //         C[idxi[1] * N + idxj[4]] = Cij[12];
    //     if (idxj[5] < N)
    //         C[idxi[1] * N + idxj[5]] = Cij[13];
    //     if (idxj[6] < N)
    //         C[idxi[1] * N + idxj[6]] = Cij[14];
    //     if (idxj[7] < N)
    //         C[idxi[1] * N + idxj[7]] = Cij[15];  
    // }    
    // if (idxi[2] < N)
    // {
    //     if (idxj[0] < N)
    //         C[idxi[2] * N + idxj[0]] = Cij[16];
    //     if (idxj[1] < N)
    //         C[idxi[2] * N + idxj[1]] = Cij[17];
    //     if (idxj[2] < N)
    //         C[idxi[2] * N + idxj[2]] = Cij[18];
    //     if (idxj[3] < N)
    //         C[idxi[2] * N + idxj[3]] = Cij[19];  
    //     if (idxj[4] < N)
    //         C[idxi[2] * N + idxj[4]] = Cij[20];
    //     if (idxj[5] < N)
    //         C[idxi[2] * N + idxj[5]] = Cij[21];
    //     if (idxj[6] < N)
    //         C[idxi[2] * N + idxj[6]] = Cij[22];
    //     if (idxj[7] < N)
    //         C[idxi[2] * N + idxj[7]] = Cij[23];  
    // }
    // if (idxi[3] < N)
    // {
    //     if (idxj[0] < N)
    //         C[idxi[3] * N + idxj[0]] = Cij[24];
    //     if (idxj[1] < N)
    //         C[idxi[3] * N + idxj[1]] = Cij[25];
    //     if (idxj[2] < N)
    //         C[idxi[3] * N + idxj[2]] = Cij[26];
    //     if (idxj[3] < N)
    //         C[idxi[3] * N + idxj[3]] = Cij[27];  
    //     if (idxj[4] < N)
    //         C[idxi[3] * N + idxj[4]] = Cij[28];
    //     if (idxj[5] < N)
    //         C[idxi[3] * N + idxj[5]] = Cij[29];
    //     if (idxj[6] < N)
    //         C[idxi[3] * N + idxj[6]] = Cij[30];
    //     if (idxj[7] < N)
    //         C[idxi[3] * N + idxj[7]] = Cij[31];  
    // }
    // if (idxi[4] < N)
    // {
    //     if (idxj[0] < N)
    //         C[idxi[4] * N + idxj[0]] = Cij[32];
    //     if (idxj[1] < N)
    //         C[idxi[4] * N + idxj[1]] = Cij[33];
    //     if (idxj[2] < N)
    //         C[idxi[4] * N + idxj[2]] = Cij[34];
    //     if (idxj[3] < N)
    //         C[idxi[4] * N + idxj[3]] = Cij[35]; 
    //     if (idxj[4] < N)
    //         C[idxi[4] * N + idxj[4]] = Cij[36];
    //     if (idxj[5] < N)
    //         C[idxi[4] * N + idxj[5]] = Cij[37];
    //     if (idxj[6] < N)
    //         C[idxi[4] * N + idxj[6]] = Cij[38];
    //     if (idxj[7] < N)
    //         C[idxi[4] * N + idxj[7]] = Cij[39];   
    // }
    // if (idxi[5] < N)
    // {
    //     if (idxj[0] < N)
    //         C[idxi[5] * N + idxj[0]] = Cij[40];
    //     if (idxj[1] < N)
    //         C[idxi[5] * N + idxj[1]] = Cij[41];
    //     if (idxj[2] < N)
    //         C[idxi[5] * N + idxj[2]] = Cij[42];
    //     if (idxj[3] < N)
    //         C[idxi[5] * N + idxj[3]] = Cij[43];  
    //     if (idxj[4] < N)
    //         C[idxi[5] * N + idxj[4]] = Cij[44];
    //     if (idxj[5] < N)
    //         C[idxi[5] * N + idxj[5]] = Cij[45];
    //     if (idxj[6] < N)
    //         C[idxi[5] * N + idxj[6]] = Cij[46];
    //     if (idxj[7] < N)
    //         C[idxi[5] * N + idxj[7]] = Cij[47];  
    // }    
    // if (idxi[6] < N)
    // {
    //     if (idxj[0] < N)
    //         C[idxi[6] * N + idxj[0]] = Cij[48];
    //     if (idxj[1] < N)
    //         C[idxi[6] * N + idxj[1]] = Cij[49];
    //     if (idxj[2] < N)
    //         C[idxi[6] * N + idxj[2]] = Cij[50];
    //     if (idxj[3] < N)
    //         C[idxi[6] * N + idxj[3]] = Cij[51];  
    //     if (idxj[4] < N)
    //         C[idxi[6] * N + idxj[4]] = Cij[52];
    //     if (idxj[5] < N)
    //         C[idxi[6] * N + idxj[5]] = Cij[53];
    //     if (idxj[6] < N)
    //         C[idxi[6] * N + idxj[6]] = Cij[54];
    //     if (idxj[7] < N)
    //         C[idxi[6] * N + idxj[7]] = Cij[55];  
    // }
    // if (idxi[7] < N)
    // {
    //     if (idxj[0] < N)
    //         C[idxi[7] * N + idxj[0]] = Cij[56];
    //     if (idxj[1] < N)
    //         C[idxi[7] * N + idxj[1]] = Cij[57];
    //     if (idxj[2] < N)
    //         C[idxi[7] * N + idxj[2]] = Cij[58];
    //     if (idxj[3] < N)
    //         C[idxi[7] * N + idxj[3]] = Cij[59];  
    //     if (idxj[4] < N)
    //         C[idxi[7] * N + idxj[4]] = Cij[60];
    //     if (idxj[5] < N)
    //         C[idxi[7] * N + idxj[5]] = Cij[61];
    //     if (idxj[6] < N)
    //         C[idxi[7] * N + idxj[6]] = Cij[62];
    //     if (idxj[7] < N)
    //         C[idxi[7] * N + idxj[7]] = Cij[63];  
    // }


////////////////////////////////////////////////////////////////////////////////////////////////////////
    //using shared memory and register, 2-D tiling, axa

    int it, jt;
    extern __shared__ _FTYPE_ sMem[];
    _FTYPE_ (*As[TILESCALE_M])[TILEDIM_K];
    _FTYPE_ (*Bs[TILESCALE_N])[TILEDIM_N];
    #pragma unroll
    for (it = 0; it < TILESCALE_M; it++)
        As[it] = (_FTYPE_ (*)[TILEDIM_K])&sMem[TILEDIM_M * TILEDIM_K * it + TILEDIM_N * TILEDIM_K * it];
    #pragma unroll
    for (jt = 0; jt < TILESCALE_N; jt++)
        Bs[jt] = (_FTYPE_ (*)[TILEDIM_N])&sMem[TILEDIM_M * TILEDIM_K * (jt + 1) + TILEDIM_N * TILEDIM_K * jt];
    
    int ii = threadIdx.y, jj = threadIdx.x;

    int idxi[TILESCALE_M], idxj[TILESCALE_N];
    idxi[0] = blockIdx.y * TILEDIM_M * TILESCALE_M + ii;
    #pragma unroll
    for (it = 1; it < TILESCALE_M; it++)
        idxi[it] = idxi[it - 1] + TILEDIM_M;
    idxj[0] = blockIdx.x * TILEDIM_N * TILESCALE_N + jj;
    #pragma unroll
    for (jt = 1; jt < TILESCALE_N; jt++)
        idxj[jt] = idxj[jt - 1] + TILEDIM_N;

    _FTYPE_ Ar[TILESCALE_M];
    _FTYPE_ Br[TILESCALE_N];
    _FTYPE_ Cij[TILESCALE_M * TILESCALE_N] = {0};

    #pragma unroll
    for (int k = 0; k < N; k += TILEDIM_K)
    {
        int kb = (N - k) < TILEDIM_K? (N - k) : TILEDIM_K;

        if (jj < kb)
        {
            #pragma unroll
            for (it = 0; it < TILESCALE_M; it++)
                As[it][ii][jj] = idxi[it] < N ? A[idxi[it] * N + k + jj] : 0;
        }
        if (ii < kb)
        {
            #pragma unroll
            for (jt = 0; jt < TILESCALE_N; jt++)
                Bs[jt][ii][jj] = idxj[jt] < N ? B[(k + ii) * N + idxj[jt]] : 0;
        }
        __syncthreads();

        #pragma unroll
        for (int kk = 0; kk < kb; kk++)
        {
            #pragma unroll
            for (it = 0; it < TILESCALE_M; it++)
                Ar[it] = As[it][ii][kk];
            #pragma unroll
            for (jt = 0; jt < TILESCALE_N; jt++)
                Br[jt] = Bs[jt][kk][jj];

            #pragma unroll
            for (it = 0; it < TILESCALE_M; it++)
            {
                #pragma unroll
                for (jt = 0; jt < TILESCALE_N; jt++)
                    Cij[it * TILESCALE_N + jt] += Ar[it] * Br[jt];
            }
        }
        __syncthreads();
    }

    #pragma unroll
    for (it = 0; it < TILESCALE_M; it++)
    {
        if (idxi[it] < N)
        {
            #pragma unroll
            for (jt = 0; jt < TILESCALE_N; jt++)
                if (idxj[jt] < N)
                    C[idxi[it] * N + idxj[jt]] = Cij[it * TILESCALE_N + jt];
        }
    }
}
#endif