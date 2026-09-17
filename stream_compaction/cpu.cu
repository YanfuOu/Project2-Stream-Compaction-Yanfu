#include <cstdio>
#include "cpu.h"

#include "common.h"

namespace StreamCompaction {
    namespace CPU {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        /**
         * CPU scan (prefix sum).
         * For performance analysis, this is supposed to be a simple for loop.
         * (Optional) For better understanding before starting moving to GPU, you can simulate your GPU scan in this function first.
         */
        void scan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            int acc = 0; 
            for(int i = 0; i < n; i++) {
                odata[i] = acc;
                acc += idata[i]; 
            }
            timer().endCpuTimer();
        }

        /**
         * CPU stream compaction without using the scan function.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithoutScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            int count = 0;
            for(int i = 0; i < n; i++) {
                if(idata[i] != 0) {
                    odata[count++] = idata[i]; // packing the survivor(aka those without 0's) into odata directly; O(n)
                }
            }
            timer().endCpuTimer();
            return count;
        }

        /**
         * CPU stream compaction using scan and scatter, like the parallel version.
         *
         * @returns the number of elements remaining after compaction.
         */
        int compactWithScan(int n, int *odata, const int *idata) {
            timer().startCpuTimer();
            int *mask = new int[n]; 
            int *scanedArr = new int[n];
            // 1. building the mask
            for(int i = 0; i < n; i++) {
                mask[i] = idata[i] == 0 ? 0 : 1; 
            }

            // 2. scan the mask. Doing the scan again because the timer is being annoying and causing core dumping
            int acc = 0; 
            for(int i = 0; i < n; i++) {
                scanedArr[i] = acc;
                acc += mask[i]; 
            }
             
            int count = 0;
            for(int i = 0; i < n; i++) {
                if(mask[i] == 1) { // if not filtered, directly store it in the final array
                    odata[scanedArr[i]] = idata[i];
                    count++;
                }
            }
            delete[] mask;
            delete[] scanedArr; 
            timer().endCpuTimer();
            return count;
        }
    }
}
