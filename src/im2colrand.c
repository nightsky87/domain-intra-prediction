#include "mex.h"
#include <string.h>

void mexFunction(int nlhs, mxArray *plhs[], int nrhs, const mxArray *prhs[])
{
    mwSize h, w;
    mwSize bsizeX, bsizeY, psize, numPatches;
    mwIndex a, n, x, y, i;
    
    double *bsize, *X, *R, *G, *B;
    
    // Extract image dimensions
    h = mxGetM(prhs[0]);
    w = mxGetN(prhs[0]) / 3;
    
    // Get the pointer to the data
	X = mxGetPr(prhs[0]);
	bsize = mxGetPr(prhs[1]);
    numPatches = mxGetScalar(prhs[2]);
    
    // Copy the block size parameters
    bsizeY = bsize[0];
    bsizeX = bsize[1];
    psize = bsizeX * bsizeY;
    
    // Create the output matrices
	plhs[0] = mxCreateDoubleMatrix(psize, numPatches, mxREAL);
	plhs[1] = mxCreateDoubleMatrix(psize, numPatches, mxREAL);
	plhs[2] = mxCreateDoubleMatrix(psize, numPatches, mxREAL);
	R = mxGetPr(plhs[0]);
	G = mxGetPr(plhs[1]);
	B = mxGetPr(plhs[2]);

    // Seed the random number generator
    srand(0);
    
    for (n = 0; n < numPatches; n++)
    {
        x = rand() % (w - bsizeX + 1);
        y = rand() % (h - bsizeY + 1);
        
        a = 0;
        for (i = 0; i < bsizeX; i++)
        {
			memcpy(&R[psize*n + a], &X[h*(x + i) + y], bsizeY * sizeof(double));
			memcpy(&G[psize*n + a], &X[h*(x + i) + y + h*w], bsizeY * sizeof(double));
			memcpy(&B[psize*n + a], &X[h*(x + i) + y + 2*h*w], bsizeY * sizeof(double));
			a += bsizeY;
        }
    }
}