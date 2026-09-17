#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        __global__ void gpu_naive_scan(int n, int *d_odata, int *d_idata, int passNum) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x; 
            if(idx >= n) {
                return; 
            }
            int offset = 1 << (passNum -1);
            if (idx >= offset) {
                d_odata[idx] = d_idata[idx - offset] + d_idata[idx];
            }
            else {
                d_odata[idx] = d_idata[idx]; //if not updating, copy over the data from the pingpong buffer
            }
        }

        __global__ void gpu_naive_scan_final_shift(int n, int *d_odata, int *d_idata) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x; 
            if(idx >= n) { //guardrail
                return;
            }
            if (idx == 0) {
                d_odata[0] = 0;
            }
            else {
                d_odata[idx] = d_idata[idx-1]; 
            }
            
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int *d_Adata; 
            int *d_Bdata; 
            cudaMalloc(&d_Adata, n * sizeof(int));
            cudaMalloc(&d_Bdata, n * sizeof(int)); 
            cudaMemcpy(d_Adata, idata, n*sizeof(int), cudaMemcpyHostToDevice); // start with idata = Adata

            dim3 blocks(blockSize);
            dim3 grid((n + blockSize -1)/blockSize);
            timer().startGpuTimer();
            // defining all the sweeps, currently performing inclusive scan
            for(int d = 1; d <= ilog2ceil(n); d++) {
                gpu_naive_scan<<<grid, blocks>>>(n, d_Bdata, d_Adata, d); // start with A in B out
                std::swap(d_Adata, d_Bdata); 
            }
            // converting from inclusive scan to exclusive scan by shifting it
            gpu_naive_scan_final_shift<<<grid, blocks>>>(n, d_Bdata, d_Adata); 
            timer().endGpuTimer();
            cudaMemcpy(odata, d_Bdata, n*sizeof(int), cudaMemcpyDeviceToHost); //d_Bdata is now the most up to date buffer
            cudaFree(d_Adata);
            cudaFree(d_Bdata); 
        }
    }
}
