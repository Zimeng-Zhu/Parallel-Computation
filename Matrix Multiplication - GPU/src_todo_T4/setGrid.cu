#include "mytypes.h"
#include <stdio.h>

void setGrid(int n, dim3 &blockDim, dim3 &gridDim)
{
   //using shared memory and register, 2-D tiling
   gridDim.y = n / (TILEDIM_M * TILESCALE_M);
   gridDim.x = n / (TILEDIM_N * TILESCALE_N);
   
   if (n % (TILEDIM_M * TILESCALE_M) != 0)
      gridDim.y++;
   if (n % (TILEDIM_N * TILESCALE_N) != 0)
      gridDim.x++;
}