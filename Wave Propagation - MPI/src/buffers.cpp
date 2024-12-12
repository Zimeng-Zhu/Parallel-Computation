#include "buffers.h"
#include <stdlib.h>
#include <stddef.h>
#include <stdio.h>
#include <vector>

Buffers::Buffers(ControlBlock& _cb, int _myRank):
    cb(_cb),
    myRank(_myRank)
{
    if (cb.px * cb.py == 1){
        // uniprocessor
        M = cb.m;
        N = cb.n;
        // we add extra ghost cells even though we don't need
        // them as this will keep the coordinate system the same
        // in the rest of the code.
        gridM = cb.m + 2;   // extra ghost cells not needed
        gridN = cb.n + 2;   // extra ghost cells not needed

        startRow = startCol = 0;
    }
    else {
        // set M, N and gridM and gridN as approrpiate 
        // for stencil method on MPI
        M = cb.m / cb.py + (getExtraRow() ?  1 : 0);
        N = cb.n / cb.px + (getExtraCol() ?  1 : 0);
        gridM = M + 2;   // add layer of ghost cells on top and bottom
        gridN = N + 2;   // add layer of ghost cells on left and right

        startRow = startCol = 0;
        // calculate global row and column origin for each rank
        for (int row = 0; row < myRank / cb.px; row++){
            startRow += cb.m / cb.py + (getExtraRow(row, cb.m, cb.py) ? 1 : 0);
            fflush(stdout);
        }
        for (int col = 0; col < myRank % cb.px; col++){
            startCol += cb.n / cb.px + (getExtraCol(col, cb.n, cb.px) ?  1 : 0);
            fflush(stdout);
        }    
    }    

#ifdef _MPI_
    upEdgeSend = new double[N];
    botEdgeSend = new double[N];
    leftEdgeSend = new double[M];
    rightEdgeSend = new double[M];

    upEdgeReceive = new double[N];
    botEdgeReceive = new double[N];
    leftEdgeReceive = new double[M];
    rightEdgeReceive = new double[M];
#endif
}

Buffers::~Buffers()
{
#ifdef _MPI_
    delete[] upEdgeSend;
    delete[] botEdgeSend;
    delete[] leftEdgeSend;
    delete[] rightEdgeSend;

    delete[] upEdgeReceive;
    delete[] botEdgeReceive;
    delete[] leftEdgeReceive;
    delete[] rightEdgeReceive;
#endif
}

void Buffers::copyToEdgeBuffer(Edge e){
    switch (e)
    {
    case TOP:
        for (int i = 1; i < gridN - 1; i++)
            upEdgeSend[i - 1] = curV(1, i);
        break;
    case BOT:
        for (int i = 1; i < gridN - 1; i++)
            botEdgeSend[i - 1] = curV(gridM - 2, i);
        break;
    case LEFT:
        for (int i = 1; i < gridM - 1; i++)
            leftEdgeSend[i - 1] = curV(i, 1);
        break;
    case RIGHT:
        for (int i = 1; i < gridM - 1; i++)
            rightEdgeSend[i - 1] = curV(i, gridN - 2);
        break;
    default:
        break;
    }
}

void Buffers::copyToBuffer(Edge e){
    switch (e)
    {
    case TOP:
        for (int i = 0; i < N; i++)
            *cur(0, i + 1) = upEdgeReceive[i];
        break;
    case BOT:
        for (int i = 0; i < N; i++)
            *cur(gridM - 1, i + 1) = botEdgeReceive[i];
        break;
    case LEFT:
        for (int i = 0; i < M; i++)
            *cur(i + 1, 0) = leftEdgeReceive[i];
        break;
    case RIGHT:
        for (int i = 0; i < M; i++)
            *cur(i + 1, gridN - 1) = rightEdgeReceive[i];
        break;
    default:
        break;
    }
}

void Buffers::setAlpha(double aVal){
    for (int i = 0; i < gridM; i++){
        for (int j = 0; j < gridN; j++){
            *alp(i, j) = aVal;
        }
    }
}

///
/// print for debug purposes
///
void Buffers::print(int iter){
    printf("%d  %5d--------------------------------\n", myRank, iter);
    for (int i = 0; i < gridM; i++){
        for (int j = 0; j < gridN; j++){
            printf("%2.3f ", curV(i, j));
        }
        printf("\n");
    }
    printf("--------------------------------\n");
}

///
/// print for debug purposes
///
void Buffers::printAlpha(){
    printf("%d  --------------------------------\n", myRank);
    for (int i = 0; i < gridM; i++){
        for (int j = 0; j < gridN; j++){
            printf("%2.3f ", alpV(i, j));
        }
        printf("\n");
    }
    printf("--------------------------------\n");
}

///
/// printMap for debug purposes
///
/// '.' for 0 cells
/// '-' if magnitude is >0 but less than <1.0
/// '*' if magnitude is >= 1.0
///
void Buffers::printMap(int iter){
    printf("%d  %5d--------------------------------\n", myRank, iter);
    for (int i = 0; i < gridM; i++){
        for (int j = 0; j < gridN; j++){
            double v = curV(i, j);
            char c;
            v = (v < 0.0) ? -1.0 * v : v;
            if (v == 0.0) {
                c = '.';
            }else if (v < 1.0){
                c = '-';
            }else
                c = '*';
            printf("%c", c);
        }
        printf("\n");
    }
    printf("--------------------------------\n");
}

///
/// printActive for debug
///
/// prints coordinates for each cell that is not 0.
///
void Buffers::printActive(int iter){
    for (int i = 1; i < gridM - 1; i++){
        for (int j = 1; j < gridN - 1; j++){
            std::pair<int, int> glob = mapToGlobal(i, j);
            if (nxtV(i,j) != 0.0){
                // print coordinates in native global #s
                // no ghost cells
                printf("%02d %04d, %3d, %3d, %.12f\n",
                        myRank, iter, glob.first, glob.second, nxtV(i, j));
                fflush(stdout);
            }
        }
    }
}

///
/// sumSq
///
/// calculate sum of the squares of each cell
/// between [r,c] and (rend, cend)
///
double Buffers::sumSq(int r, int c, int rend, int cend){
    double sumSq = 0.0;
    for (int i = r; i < rend; i++){
        for (int j = c; j < cend; j++){
            double v = curV(i, j);
            sumSq += v * v;
        }
    }
    return sumSq;
}

///
/// mapToLocal
///
/// map global coord that don't include ghost cells
/// to local coordinates that assume ghost cells.
/// returns -1, -1 if coordinates are not in this buffer
std::pair<int, int> Buffers::mapToLocal(int globr, int globc){
    if (chkBounds(globr, globc))
        return std::pair<int, int>(globr - startRow + 1, globc - startCol + 1);
    else
        return std::pair<int, int>(-1, -1);
}

///
/// mapToGlobal
///
/// map local coord that assumes ghost cells to
/// global coord that has no ghost cells
///
std::pair<int, int> Buffers::mapToGlobal(int r, int c){
    return std::pair<int, int>(startRow + r - 1, startCol + c - 1);
}

///
/// check to see if r and c are contained in this buffer
///
/// r and c are global coordinates
///
bool Buffers::chkBounds(int r, int c){
    if (r >= startRow && r < startRow + M && c >= startCol && c < startCol + N)
        return true;
    else
        return false;
}

///
/// ArrBuff - constructor
///
/// allocate the memory pool
///
ArrBuff::ArrBuff(ControlBlock& _cb, int _myRank) :
    Buffers(_cb, _myRank),
    memoryPool(nullptr),
    alpha(nullptr),
    next(nullptr),
    curr(nullptr),
    prev(nullptr)
{
    memoryPool = new double[3 * gridM * gridN]();
    alpha = new double[gridM * gridN]();
    prev = memoryPool;
    curr = &memoryPool[gridM * gridN];
    next = &memoryPool[2 * gridM * gridN];
}


ArrBuff::~ArrBuff() {
    delete[] memoryPool;
    delete[] alpha;
}

///
/// ArrBuff AdvBuffers - rotate the pointers
///
void ArrBuff::AdvBuffers(){
    double *t;
    t = prev;
    prev = curr;
    curr = next;
    next = t;
}


///
/// Array of Structures
/// 
/// This class groups all the data for each i, j point
/// near each other in memory.
///
AofSBuff::AofSBuff(ControlBlock& _cb, int _myRank) :
    Buffers(_cb, _myRank),
    memoryPool(nullptr),
    alpha(3),
    next(2),
    current(1),
    previous(0)
{
    memoryPool = new point[gridM * gridN]();
}

AofSBuff::~AofSBuff() {
    delete[] memoryPool;
}

void AofSBuff::AdvBuffers(){
    int t = previous;
    previous = current;
    current = next;
    next = t;
}