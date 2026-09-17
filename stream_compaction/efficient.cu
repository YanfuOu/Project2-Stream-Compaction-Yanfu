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
            int roundedN = 1 << ilog2ceil(n); // padding to the next power of 2
            cudaMalloc(&d_data, roundedN * sizeof(int)); 
            cudaMemset(d_data, 0, roundedN * sizeof(int)); // padding it with default 0
            cudaMemcpy(d_data, idata, n*sizeof(int), cudaMemcpyHostToDevice); 

            // Upsweep
            for(int d = 0; d <= (ilog2ceil(n) -1); d++) { //d calculates the passNum (ilog2ceil(n) -1)
                upsweep<<<grid, blocks>>>(roundedN, d_data, d); 
            }
            cudaMemset(d_data + roundedN - 1, 0, sizeof(int)); // setting d_data[roundedN - 1] = 0. Aka the last value is 0
            // Downsweep
            cudaMemset(d_data + n - 1, 0, sizeof(int)); // setting d_data[n-1] = 0
            for(int d = (ilog2ceil(n) - 1); d >= 0; d--) {
                downsweep<<<grid, blocks>>>(roundedN, d_data, d); 
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
        __global__ void mask(int n, int *d_odata, int *d_idata) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x; 
            if(idx >= n) return; 
            d_odata[idx] = d_idata[idx] == 0 ? 0 : 1; 
        }

        __global__ void scatter(int n, int *d_odata, int *d_idata, int *d_maskArr, int *d_scannedMaskArr) {
            int idx = blockIdx.x * blockDim.x + threadIdx.x; 
            if(idx >= n) return; 
            if(d_maskArr[idx] == 1) {
                d_odata[d_scannedMaskArr[idx]] = d_idata[idx]; 
            }
        }
        int compact(int n, int *odata, const int *idata) {
            timer().startGpuTimer();
            dim3 blocks(blockSize);
            dim3 grid((n + blockSize - 1)/blockSize);
            int *d_maskArr; 
            int *maskArr = new int[n];
            int *d_idata; 
            int *d_odata; 
            cudaMalloc(&d_maskArr, n*sizeof(int)); 
            cudaMalloc(&d_idata, n*sizeof(int)); 
            cudaMalloc(&d_odata, n*sizeof(int)); 
            cudaMemcpy(d_idata, idata, n*sizeof(int), cudaMemcpyHostToDevice); 
            // 1. create the mask
            mask<<<grid, blocks>>>(n, d_maskArr, d_idata); 
            cudaMemcpy(maskArr, d_maskArr, n*sizeof(int), cudaMemcpyDeviceToHost); 
            // 2. scan the mask
            int *scannedMaskArr = new int[n];
            int *d_scannedMaskArr;
            cudaMalloc(&d_scannedMaskArr, n*sizeof(int));
            timer().endGpuTimer(); // scan() starts its own GPU timer
            scan(n, scannedMaskArr, maskArr); 
            timer().startGpuTimer();
            cudaMemcpy(d_scannedMaskArr, scannedMaskArr, n*sizeof(int), cudaMemcpyHostToDevice);

            // 3. Scatter
            scatter<<<grid, blocks>>>(n, d_odata, d_idata, d_maskArr, d_scannedMaskArr);
            cudaMemcpy(odata, d_odata, n*sizeof(int), cudaMemcpyDeviceToHost); 

            int count = scannedMaskArr[n - 1] + maskArr[n - 1];
            cudaFree(d_maskArr); 
            delete[] maskArr; 
            cudaFree(d_idata); 
            cudaFree(d_odata); 
            delete[] scannedMaskArr; 
            cudaFree(d_scannedMaskArr); 
            timer().endGpuTimer();
            return count;
        }
    }
}
