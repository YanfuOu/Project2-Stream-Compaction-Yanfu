#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        __global__ void upsweep(int n, int *data, int passNum) {
            int idx = blockIdx.x * blockDim.x +  threadIdx.x; 
            if(idx >= n) {
                return;
            }
            if(idx % (1<<(passNum+1)) == 0) {
                data[idx + (1<<(passNum +1)) -1] = data[idx + (1<<(passNum + 1)) -1] + data[idx + (1<<passNum) -1]; 
            }
        }

        __global__ void downsweep(int n, int *data, int passNum) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x; 
            if(idx >= n) {
                return;
            }
            if(idx % (1<<(passNum+1)) == 0) {
                int t = data[idx + (1<<passNum) -1]; // save left child
                data[idx + (1<<passNum) -1] = data[idx + ((1<<(passNum+1)) -1)]; 
                data[idx + (1<<(passNum+1)) -1 ] += t; 

            }

        }
        void scan(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            dim3 blocks(blockSize);
            dim3 grid((n + blockSize - 1)/blockSize);
            int *d_data; //can be done in place, so no need for pingpong buffers
            cudaMalloc(&d_data, n * sizeof(int)); 
            cudaMemcpy(d_data, idata, n*sizeof(int), cudaMemcpyHostToDevice); 

            // Upsweep
            for(int d = 0; d <= (ilog2ceil(n) -1); d++) { //d calculates the passNum (ilog2ceil(n) -1)
                upsweep<<<grid, blocks>>>(n, d_data, d); 
            }

            // Downsweep
            cudaMemset(d_data + n - 1, 0, sizeof(int)); // setting d_data[n-1] = 0
            for(int d = (ilog2ceil(n) - 1); d >= 0; d--) {
                downsweep<<<grid, blocks>>>(n, d_data, d); 
            }
            cudaMemcpy(odata, d_data, n*sizeof(int), cudaMemcpyDeviceToHost); 
            cudaFree(d_data); 
            timer().endGpuTimer();
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            // TODO
            timer().endGpuTimer();
            return -1;
        }
    }
}
