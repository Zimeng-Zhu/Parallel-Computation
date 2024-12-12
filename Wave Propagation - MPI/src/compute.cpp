#include "stimulus.h"
#include "obstacle.h"
#include "compute.h"
#include "plotter.h"
#include <list>
#include <random>
#include <cstdlib>
#include <math.h>
#include <iostream>
#include <fstream>
#include <assert.h>
using namespace std;

#ifdef _MPI_
#include <mpi.h>
#endif
#include <math.h>

///
/// Compute object
///
/// upon construction, runs a simulation.
///
Compute::Compute(Buffers &u, Plotter *plt, ControlBlock &cb, const int _myRank, const int seedv):
    u(u), plt(plt), cb(cb), myRank(_myRank), seedv(seedv), M(u.M), N(u.N)
{
    int row = myRank / cb.px, col = myRank % cb.px;
    topGlobalEdge = (row == 0 ? true : false);
    botGlobalEdge = (row == cb.py - 1 ? true : false);
    leftGlobalEdge = (col == 0 ? true : false);
    rightGlobalEdge = (col == cb.px - 1 ? true : false);
}


///
/// Simulate
///
/// calls class specific calcU and calcEdgeU
///
void Compute::Simulate()
{
    const unsigned int t = 1;	 // timestep
    const unsigned int h = 1;    // grid space
    const double c = 0.29;   // velocity
    const double kappa = c * t / h;

    mt19937 generator(seedv);

    u.setAlpha((c * t / h) * (c * t / h));

    list<Stimulus *> sList;
    list<Obstacle *> oList;
    int iter = 0;

    ifstream f(cb.configFileName);
    if (cb.config.count("objects")){
        for (int i = 0; i < cb.config["objects"].size(); i++){
            auto const &ob = cb.config["objects"][i];
            if (ob["type"] == "sine"){
                sList.push_back(new StimSine(u, ob["row"], ob["col"],
                                ob["start"], ob["duration"],
                                ob["period"]));
            }
            else if (ob["type"] == "rectobstacle"){
                oList.push_back(new Rectangle(u, ob["row"], ob["col"],
                                ob["height"], ob["width"]));
            }
        }
    }
    else{
        fprintf(stderr, "Using hardcoded stimulus\n");
        Rectangle obstacleA(u, cb.m / 2 + 5, cb.n / 2, 45, 5);
        Rectangle obstacleB(u, cb.m / 2 - 50, cb.n / 2, 45, 5);
        sList.push_back(new StimSine(u, cb.m / 2, cb.n / 3, 0/*start*/, 500/*duration*/, 10/*period*/));
    }

    ///
    /// generate stimulus
    ///
    /// once quiet (non-deterministic),
    /// we exit this loop and go into a loop that
    /// continues until iterations is exhausted
    ///
    int row = myRank / cb.px, col = myRank % cb.px;
    while (!sList.empty() && iter < cb.niters){
        for (auto it = begin(sList); it!= end(sList);){
            if (!(*it)->doit(iter)){
                delete *it;
                it = sList.erase(it);
            }
            else{
                it++;
            }
	    }

        if (!cb.noComm)
        {
#ifdef _MPI_
            if (!topGlobalEdge)
            {
                u.copyToEdgeBuffer(Buffers::TOP);
                MPI_Isend(u.upEdgeSend, u.N, MPI_DOUBLE, (row - 1) * cb.px + col, RecvCBot, MPI_COMM_WORLD, &sndRqst[0]);
                MPI_Irecv(u.upEdgeReceive, u.N, MPI_DOUBLE, (row - 1) * cb.px + col, RecvCTop, MPI_COMM_WORLD, &rcvRqst[0]);
            }
            if (!botGlobalEdge)
            {
                u.copyToEdgeBuffer(Buffers::BOT);
                MPI_Isend(u.botEdgeSend, u.N, MPI_DOUBLE, (row + 1) * cb.px + col, RecvCTop, MPI_COMM_WORLD, &sndRqst[1]);
                MPI_Irecv(u.botEdgeReceive, u.N, MPI_DOUBLE, (row + 1) * cb.px + col, RecvCBot, MPI_COMM_WORLD, &rcvRqst[1]);            
            }
            if (!leftGlobalEdge)
            {
                u.copyToEdgeBuffer(Buffers::LEFT);
                MPI_Isend(u.leftEdgeSend, u.M, MPI_DOUBLE, myRank - 1, RecvCRight, MPI_COMM_WORLD, &sndRqst[2]);
                MPI_Irecv(u.leftEdgeReceive, u.M, MPI_DOUBLE, myRank - 1, RecvCLeft, MPI_COMM_WORLD, &rcvRqst[2]);               
            }
            if (!rightGlobalEdge)
            {
                u.copyToEdgeBuffer(Buffers::RIGHT);
                MPI_Isend(u.rightEdgeSend, u.M, MPI_DOUBLE, myRank + 1, RecvCLeft, MPI_COMM_WORLD, &sndRqst[3]);
                MPI_Irecv(u.rightEdgeReceive, u.M, MPI_DOUBLE, myRank + 1, RecvCRight, MPI_COMM_WORLD, &rcvRqst[3]);                  
            }
#endif
        }

        calcU(u);

        if (!cb.noComm)
        {
#ifdef _MPI_
            if (!topGlobalEdge)
            {
                MPI_Wait(&rcvRqst[0], &recvStatus[0]);
                u.copyToBuffer(Buffers::TOP);
            }
            if (!botGlobalEdge)
            {
                MPI_Wait(&rcvRqst[1], &recvStatus[1]);
                u.copyToBuffer(Buffers::BOT);
            }   
            if (!leftGlobalEdge)
            {
                MPI_Wait(&rcvRqst[2], &recvStatus[2]);
                u.copyToBuffer(Buffers::LEFT);
            }
            if (!rightGlobalEdge)
            {
                MPI_Wait(&rcvRqst[3], &recvStatus[3]);
                u.copyToBuffer(Buffers::RIGHT);
            }
#endif
        }

        calcEdgeU(u, kappa);

        if (cb.plot_freq && iter % cb.plot_freq == 0)
            plt->updatePlot(iter, u.gridM, u.gridN);

        u.AdvBuffers();

        iter++;
    }

    ///
    /// all stimulus done
    /// keep simulating till end
    ///
    for (; iter < cb.niters; iter++){
        if (!cb.noComm)
        {
#ifdef _MPI_
            if (!topGlobalEdge)
            {
                u.copyToEdgeBuffer(Buffers::TOP);
                MPI_Isend(u.upEdgeSend, u.N, MPI_DOUBLE, (row - 1) * cb.px + col, RecvCBot, MPI_COMM_WORLD, &sndRqst[0]);
                MPI_Irecv(u.upEdgeReceive, u.N, MPI_DOUBLE, (row - 1) * cb.px + col, RecvCTop, MPI_COMM_WORLD, &rcvRqst[0]);
            }
            if (!botGlobalEdge)
            {
                u.copyToEdgeBuffer(Buffers::BOT);
                MPI_Isend(u.botEdgeSend, u.N, MPI_DOUBLE, (row + 1) * cb.px + col, RecvCTop, MPI_COMM_WORLD, &sndRqst[1]);
                MPI_Irecv(u.botEdgeReceive, u.N, MPI_DOUBLE, (row + 1) * cb.px + col, RecvCBot, MPI_COMM_WORLD, &rcvRqst[1]);            
            }
            if (!leftGlobalEdge)
            {
                u.copyToEdgeBuffer(Buffers::LEFT);
                MPI_Isend(u.leftEdgeSend, u.M, MPI_DOUBLE, myRank - 1, RecvCRight, MPI_COMM_WORLD, &sndRqst[2]);
                MPI_Irecv(u.leftEdgeReceive, u.M, MPI_DOUBLE, myRank - 1, RecvCLeft, MPI_COMM_WORLD, &rcvRqst[2]);               
            }
            if (!rightGlobalEdge)
            {
                u.copyToEdgeBuffer(Buffers::RIGHT);
                MPI_Isend(u.rightEdgeSend, u.M, MPI_DOUBLE, myRank + 1, RecvCLeft, MPI_COMM_WORLD, &sndRqst[3]);
                MPI_Irecv(u.rightEdgeReceive, u.M, MPI_DOUBLE, myRank + 1, RecvCRight, MPI_COMM_WORLD, &rcvRqst[3]);                  
            }
#endif
        }

        calcU(u);

        if (!cb.noComm)
        {
#ifdef _MPI_
            if (!topGlobalEdge)
            {
                MPI_Wait(&rcvRqst[0], &recvStatus[0]);
                u.copyToBuffer(Buffers::TOP);
            }
            if (!botGlobalEdge)
            {
                MPI_Wait(&rcvRqst[1], &recvStatus[1]);
                u.copyToBuffer(Buffers::BOT);
            }   
            if (!leftGlobalEdge)
            {
                MPI_Wait(&rcvRqst[2], &recvStatus[2]);
                u.copyToBuffer(Buffers::LEFT);
            }
            if (!rightGlobalEdge)
            {
                MPI_Wait(&rcvRqst[3], &recvStatus[3]);
                u.copyToBuffer(Buffers::RIGHT);
            }
#endif
        }

        calcEdgeU(u, kappa);

        if ((cb.plot_freq!=0) && (iter % cb.plot_freq == 0))
            plt->updatePlot(iter, u.gridM, u.gridN);

        u.AdvBuffers();
    }
}

TwoDWave::TwoDWave(Buffers &u, Plotter *plt, ControlBlock &cb, const int _myRank,
		 const int seedv):
    Compute(u, plt, cb, _myRank, seedv){};

void TwoDWave::calcU(Buffers &u)
{
    // interior always starts at 2,2, ends at gridN - 3
    for (int i = 2; i < u.gridM - 2; i++){
        for (int j = 2; j < u.gridN - 2; j++){
            *u.nxt(i, j) =
                u.alpV(i, j) *
                (u.curV(i - 1, j) + u.curV(i + 1, j) +
                u.curV(i, j - 1) + u.curV(i, j + 1) -
                4 * u.curV(i, j)) +
                2 * u.curV(i, j) - u.preV(i, j);
        }
    }
}

void TwoDWave::calcEdgeU(Buffers &u, const double kappa)
{
    // top and bottom edge
    for (int j = 1; j < u.gridN - 1; j++){
        int i = 1;
        *u.nxt(i, j) =
            u.alpV(i, j) *
            (u.curV(i - 1, j) + u.curV(i + 1, j) +
            u.curV(i, j - 1) + u.curV(i, j + 1) -
            4 * u.curV(i, j)) +
            2 * u.curV(i, j) - u.preV(i, j);
        i = u.gridM - 2;
        *u.nxt(i, j) =
            u.alpV(i, j) *
            (u.curV(i - 1, j) + u.curV(i + 1, j) +
            u.curV(i, j - 1) + u.curV(i, j + 1) -
            4 * u.curV(i, j)) +
            2 * u.curV(i, j) - u.preV(i, j);
    }

    // left and right
    for (int i = 1; i < u.gridM - 1; i++){
        int j = 1;
        *u.nxt(i, j) =
            u.alpV(i, j) *
            (u.curV(i - 1, j) + u.curV(i + 1, j) +
            u.curV(i, j - 1) + u.curV(i, j + 1) -
            4 * u.curV(i, j)) +
            2 * u.curV(i, j) - u.preV(i, j);
        j = u.gridN - 2;
        *u.nxt(i, j) =
            u.alpV(i, j) *
            (u.curV(i - 1, j) + u.curV(i + 1, j) +
            u.curV(i, j - 1) + u.curV(i, j + 1) -
            4 * u.curV(i, j)) +
            2 * u.curV(i, j) - u.preV(i, j);
    }

    // set the boundary conditions to absorbing boundary conditions (ABC)
    // du/dx = -1/c du/dt   x=0
    // du/dx = 1/c du/dt    x=N-1
    // conditions for an internal boundary (ie.g. ghost cells)
    // top edge


    // top global edge (instead of ghost cells)
    if (topGlobalEdge){
        // top row absorbing boundary condition
        int i = 0;
        for (int j = 1; j < u.gridN - 1; j++){
            *u.nxt(i, j) = u.curV(i + 1, j) +
            ((kappa - 1) / (kappa + 1)) * (u.nxtV(i + 1, j) - u.curV(i, j));
        }
    }

    // bottom edge (instead of ghost cells)
    if (botGlobalEdge){
        int i = u.gridM - 1;
        for (int j = 1; j < u.gridN - 1; j++){
            *u.nxt(i, j) = u.curV(i - 1, j) +
            ((kappa - 1) / (kappa + 1)) * (u.nxtV(i - 1, j) - u.curV(i, j));
        }
    }

    // left edge
    if (leftGlobalEdge){
        int j = 0;
        for (int i = 1; i < u.gridM - 1; i++){
            *u.nxt(i, j) = u.curV(i, j + 1) +
            ((kappa - 1) / (kappa + 1)) * (u.nxtV(i, j + 1) - u.curV(i, j));
        }
    }
    // right edge
    if (rightGlobalEdge){
        int j = u.gridN - 1;
        for (int i = 1; i < u.gridM - 1; i++){
            *u.nxt(i, j) = u.curV(i, j - 1) +
            ((kappa - 1) / (kappa + 1)) * (u.nxtV(i, j - 1) - u.curV(i, j));
        }
    }
}

//!
//! Use a different propgation model
//! This model shifts values in the horizontal direction
//!
DebugPropagate::DebugPropagate(Buffers &u, Plotter *plt, ControlBlock &cb, const int _myRank,
		 const int seedv):
    Compute(u, plt, cb, _myRank, seedv){};

//!
//! compute the interior cells
//!
void DebugPropagate::calcU(Buffers &u)
{
    // interior always starts at 2,2, ends at gridN-3
    for (int i = 2; i < u.gridM - 2; i++){
        for (int j = 2; j < u.gridN - 2; j++){
            *u.nxt(i,j) = u.curV(i, j - 1);
        }
    }
}

//!
//! compute edges
//! (either interior edges or global edges)
//!
void DebugPropagate::calcEdgeU(Buffers &u, const double kappa)
{
    if (topGlobalEdge){
        // top row absorbing boundary condition
        for (int j=1; j<u.gridN-1; j++){
            *u.nxt(1,j) = 0;
        }
    }else{
        int i = 1;
        for (int j=1; j<u.gridN-1; j++){
            *u.nxt(i,j) = u.curV(i, j-1);
        }
    }

    // bottom edge
    if (botGlobalEdge){
        for (int j=1; j<u.gridN-1; j++){
            *u.nxt(u.gridM-2,j) = 0;
        }
    }else{
        int i=u.gridM-2;
        for (int j=1; j<u.gridN-1; j++){
            *u.nxt(i,j) = u.curV(i, j-1);
        }
    }

    // left edge
    if (leftGlobalEdge){
        for (int i=1; i<u.gridM-1; i++){
            *u.nxt(i,1) = 0.0;
        }
    }else{
        int j=1;
        for (int i=1; i<u.gridM-1; i++){
            *u.nxt(i,j) = u.curV(i, j-1);
        }
    }
    // right edge
    if (rightGlobalEdge){
        for (int i=1; i<u.gridM-1; i++){
            // right column
            *u.nxt(i,u.gridN-2) = 0.0;
        }
    }else{
        int j=u.gridN-2;
        for (int i=1; i<u.gridM-1; i++){
            *u.nxt(i,j) = u.curV(i, j-1);
        }
    }
}