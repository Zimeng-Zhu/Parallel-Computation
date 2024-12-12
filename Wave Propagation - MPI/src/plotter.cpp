/// cse260 - hw 3
/// see COPYRIGHT
/// Bryan Chin - University of California San Diego
///
/// provide grpahical output as a netcdf file
///
#include <iostream>
#include <vector>
#include "controlblock.h"
#include "plotter.h"
#ifdef _MPI_
#include <mpi.h>
#endif
#include <cstring>
#include <string>
#include <stdio.h>
#include <netcdf.h>
using namespace std;
#define ERRCODE 2
#define ERR(X) {fprintf(stderr, "Error: %s  %s %d\n", nc_strerror(X), __FILE__, __LINE__); exit(ERRCODE);}

Plotter::Plotter(Buffers &_u, ControlBlock& _cb):
    u(_u),
    cb(_cb), NDIMS(3), tick(0)
{
    if (cb.plot_freq == 0)
	    return;

#ifdef _MPI_
    localPlot.resize(u.M * u.N);
    for (int i = 0; i < u.M; i++)
    {
        for (int j = 0; j < u.N; j++)
            localPlot[i * u.N + j] = 0.0;
    }

    if (u.myRank == 0)
    {
        recvPlot.resize(cb.m * cb.n);

        int total_size = 0;
        for (int row = 0; row < cb.py; row++)
        {
            for (int col = 0; col < cb.px; col++)
            {
                int temp_m = cb.m / cb.py + (u.getExtraRow(row, cb.m, cb.py) ? 1 : 0);
                int temp_n = cb.n / cb.px + (u.getExtraCol(col, cb.n, cb.px) ?  1 : 0);
                blockSizes.push_back(temp_m * temp_n);
                displs.push_back(total_size);
                total_size += temp_m * temp_n;
            }
        }
    }
    MPI_Gatherv(localPlot.data(), u.M * u.N, MPI_DOUBLE, recvPlot.data(), blockSizes.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (u.myRank == 0)
    {
        globalPlot.resize(cb.m * cb.n);
        for (int row = 0, ii = 0; row < cb.py; row++)
        {
            int temp_m = cb.m / cb.py + (u.getExtraRow(row, cb.m, cb.py) ? 1 : 0);
            for (int i = 0; i < temp_m; i++)
            {
                for (int col = 0, jj = 0; col < cb.px; col++)
                {
                    int temp_n = cb.n / cb.px + (u.getExtraCol(col, cb.n, cb.px) ?  1 : 0);
                    for (int j = 0; j < temp_n; j++)
                        globalPlot[(ii + i) * cb.n + (jj + j)] = recvPlot[displs[row * cb.px + col] + i * temp_n + j];
                    jj += temp_n;
                }
            }
            ii += temp_m;
        }
    }
#else
    for (int i = 0; i < cb.m * cb.n; i++)
	    globalPlot.push_back(0.0);

#endif    
    int retval, x_dimid, y_dimid, rec_dimid;
    string outName(cb.programPath.filename());
    outName = outName + ".nc";
    if ((retval = nc_create(outName.c_str(), NC_CLOBBER, &ncid))){
	    ERR(retval);
    }
    if ((retval = nc_def_dim(ncid, "y", cb.m, &y_dimid))){
	    ERR(retval);
    }
    if ((retval = nc_def_dim(ncid, "x", cb.n, &x_dimid))){
	    ERR(retval);
    }
    if ((retval = nc_def_dim(ncid, "time", NC_UNLIMITED, &rec_dimid))){
	    ERR(retval);
    }

    dimids[0] = rec_dimid;
    dimids[1] = y_dimid;
    dimids[2] = x_dimid;
    
    if ((retval = nc_def_var(ncid, "data", NC_DOUBLE, NDIMS, dimids, &varid))){
	    ERR(retval);
    }

    startp[0] = tick;    // this is the time dimension.
    startp[1] = 0;    // this is the row dimension start
    startp[2] = 0;    // this is the col dimension start
    countp[0] = 1;
    countp[1] = cb.m; // # of elements in a col
    countp[2] = cb.n; // # of elements in a row
    
    if ((retval = nc_enddef(ncid))){
	    ERR(retval);
    }
}


///
/// update the plot output
///
void Plotter::updatePlot(int niter, int m, int n){

#ifndef _MPI_
    for (int i = 0; i < u.M; i++){
        for (int j = 0; j < u.N; j++)
            globalPlot[i * cb.n + j] = u.nxtV(i + 1, j + 1);
    }
#else
    for (int i = 0; i < u.M; i++)
    {
        for (int j = 0; j < u.N; j++)
            localPlot[i * u.N + j] = u.nxtV(i + 1, j + 1);
    }

    MPI_Gatherv(localPlot.data(), u.M * u.N, MPI_DOUBLE, recvPlot.data(), blockSizes.data(), displs.data(), MPI_DOUBLE, 0, MPI_COMM_WORLD);

    if (u.myRank == 0)
    {
        globalPlot.resize(cb.m * cb.n);
        for (int row = 0, ii = 0; row < cb.py; row++)
        {
            int temp_m = cb.m / cb.py + (u.getExtraRow(row, cb.m, cb.py) ? 1 : 0);
            for (int i = 0; i < temp_m; i++)
            {
                for (int col = 0, jj = 0; col < cb.px; col++)
                {
                    int temp_n = cb.n / cb.px + (u.getExtraCol(col, cb.n, cb.px) ?  1 : 0);
                    for (int j = 0; j < temp_n; j++)
                        globalPlot[(ii + i) * cb.n + (jj + j)] = recvPlot[displs[row * cb.px + col] + i * temp_n + j];
                    jj += temp_n;
                }
            }
            ii += temp_m;
        }
    }
#endif

#ifdef _MPI_
    if (u.myRank == 0){
#endif
        int retval;
        startp[0] = tick++;
        if ((retval = nc_put_vara_double(ncid, varid, startp, countp, &globalPlot.data()[0]))){
            ERR(retval);
        }
#ifdef _MPI_
    }
#endif
}


///
/// print the global buffer as ascii
///
void Plotter::printGlobal(ControlBlock& cb, int myRank, int iter){
    printf("%d  %5d--------------------------------\n", myRank, iter);
    double *p = &globalPlot.data()[0];
    for (int i = 0; i < cb.m; i++){
        for (int j = 0; j < cb.n; j++){
            printf("%2.3f ", *p++);
        }
        printf("\n");
    }
    printf("--------------------------------\n");
}


Plotter::~Plotter() {
    if (cb.plot_freq == 0)
	    return;

    int retval;
    if (u.myRank == 0 && (retval = nc_close(ncid))){
	    ERR(retval);
    }
    if (u.myRank != 0){
	    return;
    }
}