#pragma once
#include "buffers.h"
#include "plotter.h"
#include "controlblock.h"

#ifdef _MPI_
#include <mpi.h>
#endif
#include <array>
#include "json.hpp"

using json = nlohmann::json;
using namespace std;
class Compute {
public:
   Compute(Buffers &u, Plotter *plt, ControlBlock &cb, const int myrank, const int seedv=1);
   void Simulate();

protected:
   virtual void calcU(Buffers &u) = 0;
   virtual void calcEdgeU(Buffers &u, const double kappa) = 0;

#if _MPI_
   MPI_Request sndRqst[4];
   MPI_Request rcvRqst[4];
   MPI_Status recvStatus[4];   // receive status

   const int RecvCTop = 1;    // receive u.curr
   const int RecvCBot = 2;
   const int RecvCLeft = 3;
   const int RecvCRight= 4;
#endif
   Buffers &u;
   Plotter *plt;
   ControlBlock& cb;
   const int myRank;
   const int seedv;

   // am I a border processor?
   bool topGlobalEdge;
   bool botGlobalEdge;
   bool leftGlobalEdge;
   bool rightGlobalEdge;

   int N;
   int M;
};


///
/// TwoDWave propagation model
///
class TwoDWave : public Compute
{
public:
   TwoDWave(Buffers &u, Plotter *plt, ControlBlock &cb, const int myrank, const int seedv=1);

protected:
   void calcU(Buffers &u);
   void calcEdgeU(Buffers &u, const double kappa);
};


///
/// Debug propagation model.
/// Shifts value to the right by oen cell
///
class DebugPropagate : public Compute
{
public:
   DebugPropagate(Buffers &u, Plotter *plt, ControlBlock &cb, const int myrank, const int seedv=1);

protected:
   void calcU(Buffers &u);
   void calcEdgeU(Buffers &u, const double kappa);
};