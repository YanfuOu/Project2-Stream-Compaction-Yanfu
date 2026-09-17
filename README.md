CUDA Stream Compaction
======================

**University of Pennsylvania, CIS 565: GPU Programming and Architecture, Project 2**

* Yanfu
  * [LinkedIn](https://www.linkedin.com/in/yanfuou/)
* Tested on: Unbuntu 24.04LTS, Ryzen 7 7840HS @ 2.5GHz 64GB DDR4, RTX 5070 8GB GPU Laptop

## Readme

### Why is my work-efficient GPU scan slower? (Part 5, extra credit +5)
At SIZE=256, we have the following data across CPU scan, Naive scan, and work efficient scan for power of two. 
|  Scan Impl. | ms      |
| ------------------- | ------- |
| CPU                 | 0.00012 |
| Naive Scan          | 0.05824 |
| Work Efficient Scan | 0.17254 |


It boils down the the following reasons:
- Deeper upsweep/downsweep levels have very low occupancy: only threads with `idx % 2^(d+1) == 0` do work (aka 50% → 25% → … → 1 thread). Most threads take the early-out and idle.
- The same full grid (`n / blockSize` blocks) is launched every pass, so later kernels still pay a full launch to run almost no work.
- Those idle threads can be dropped by shrinking the grid each pass: launch `n / 2^(d+1)` threads instead of `n`.
- Compact the remaining threads with an index remap: thread `tid` owns node `k = tid << (d + 1)` (same nodes the `%` test selected). Left/right children stay `k + 2^d - 1` and `k + 2^(d+1) - 1`. Same kernel body, just no modulo filter. This implementation would be similar to the one taught in lecture

### Performance Analysis 

![scan-impls-perf-chart](scan-impls-perf-chart.png)

For small size of n, the CPU performance clearly dominates. All the other implementations performs roughly similar, with the Work Efficient implement performing slightly better amongst the group of non-CPU implementations. 

However, at size 2^18 to 2^20, all 3 implementations have roughly the same performance. 

Beyon which, the Thrust library performs substantially better than all the other implementation. This makes sense because Nvidia probably poured a lot of engineering hours into optimizing the Thrust library. Additionally, in the Nsight System analysis below, we can see that it has a lot fewer kernel launches compared to my implementations, which help save on the kernel launch overhead and focuses more resources on computation instead. The second best performing implementation is Work Efficient, which shows the advantage of the Work Efficient algorithm and its runtime with large data sizes. In third place is the naive scan, which performed poorest amongst the GPU implementations because of its low utilization rate. Lastly, the CPU performed that worst at above size 2^20. This makes sense since the CPU's computation is sequential and relies on a single for loop.  

Ultimately, scan is not compute bound. The mathetical computation is addition is very cheap compared to kernel launches and memory copy to and from the Device. That's probably why work-efficient's lower arithemetic complexity isn't as helpful as we have expected when size n is small. 

| size n     | CPU(ms) | Naive(ms)   | Work-efficient(ms) | Thrust(ms)  |
| ----- | ------------ | ------------ | -------------- | ------------ |
| 1<<5  | 0.000201     | 0.108832     | 0.134016       | 0.052352     |
| 1<<6  | 0.00005      | 0.02         | 0.029664       | 0.01728      |
| 1<<7  | 0.00006      | 0.009888     | 0.046752       | 0.037984     |
| 1<<8  | 0.000091     | 0.012512     | 0.045376       | 0.018784     |
| 1<<9  | 0.00014      | 0.010496     | 0.051008       | 0.019552     |
| 1<<10 | 0.001392     | 0.031232     | 0.042144       | 0.016256     |
| 1<<11 | 0.001412     | 0.012928     | 0.040608       | 0.016576     |
| 1<<12 | 0.001132     | 0.025888     | 0.05328        | 0.018112     |
| 1<<13 | 0.003056     | 0.0152       | 0.044608       | 0.016512     |
| 1<<14 | 0.004849     | 0.016352     | 0.042528       | 0.018016     |
| 1<<15 | 0.009578     | 0.091296     | 0.040416       | 0.020896     |
| 1<<16 | 0.019105     | 0.086432     | 0.041632       | 0.0184       |
| 1<<17 | 0.038192     | 0.096064     | 0.054976       | 0.020288     |
| 1<<18 | 0.074339     | 0.06848      | 0.083392       | 0.119904     |
| 1<<19 | 0.147976     | 0.094592     | 0.144448       | 0.165888     |
| 1<<20 | 0.276466     | 0.245408     | 0.256992       | 0.190208     |
| 1<<21 | 6.928882     | 0.337152     | 0.50928        | 0.17712      |
| 1<<22 | 8.107503     | 1.221024     | 1.010944       | 0.214624     |
| 1<<23 | 16.461376    | 6.244256     | 2.59392        | 0.39184      |
| 1<<24 | 33.179955    | 13.333536    | 7.578912       | 0.659232     |
| 1<<25 | 67.141464    | 28.017471    | 15.537376      | 1.267776     |
| 1<<26 | 130.228943   | 59.076225    | 31.683489      | 2.271776     |
| 1<<27 | 257.637939   | 124.105377   | 60.007999      | 3.342752     |
| 1<<28 | 496.142273   | 242.526855   | 132.665146     | 8.419552     |
| 1<<29 | 979.672546   | 461.196869   | 249.650116     | 16.567104    |
| 1<<30 | 1895.528809  | device OOM   | 504.365509     | device OOM   |
| 1<<31 | INT overflow | INT overflow | INT overflow   | INT overflow |

### Nsight Systems timeline comparison

##### Naive Scan 
We can see a long series of small kernels. 
![naiveScan](naiveScan-nsight.png)

##### Work efficient
We can see a more dense series of kernels
![naiveScan](workefficientScan-nsight.png)

##### Thrust
We don't see that many dense series of kernels as we have see in my Naive and Work Efficient Implementations. Instead, it's replaced by a few dense operations. This helps to reduce kernel launch overhead and increase compute usage instead of being bottlenecked by memory operations.  
![naiveScan](thrustScan-nsight.png)

This evidence further supports my theory above. My implementations(Naive and Work Efficient) has a lot of costly kernel launch overhead. Meanwhile, Thrust performs the entire scan operation in very few steps   

##### Block Size Tuning

I sweep the blockSize across from 8 to 1024 at size n = 2^24. I chose 2^24 because this is a sufficiently large size at which the full advantage of each implementation can be exercised. I noticed that the min for Naive is 10.21MS at blockSize = 512. The min for Work-Efficient implementation is 5.94 at blockSize = 256. As for thrust, the min is 0.53 at blockSize = 256. As a result, the most optimal blockSize should be 256.  

| blockSize | Naive(ms) | Work-efficient(ms) | Thrust(ms) |
| --------- | --------- | ------------------ | ---------- |
| 8         | 55.366623 | 96.489502          | 0.540224   |
| 16        | 27.027905 | 48.726143          | 0.58272    |
| 32        | 13.51312  | 25.301472          | 0.539136   |
| 64        | 10.288256 | 12.660448          | 0.545664   |
| 128       | 10.268864 | 6.7248             | 0.546848   |
| 256       | 10.283296 | 5.94016            | 0.539232   |
| 512       | 10.21744  | 6.21584            | 0.540192   |
| 1024      | 10.84192  | 8.997952           | 0.543744   |
| Min       | 10.21744  | 5.94016            | 0.539136   |

## Full Test Program Output

The following is the output build in Release mode. size = 256

```
****************
** SCAN TESTS **
****************
Input array: 
    [  30   2  53   6  84  19  26   3  43  53   1  90  68 ...  55   0 ]
==== cpu scan, power-of-two ====
Output array: 
   elapsed time: 0.000882ms    (std::chrono Measured)
    [   0  30  32  85  91 175 194 220 223 266 319 320 410 ... 11783 11838 ]
==== cpu scan, non-power-of-two ====
   elapsed time: 0.00012ms    (std::chrono Measured)
    passed 
==== naive scan, power-of-two ====
    [   0  30  32  85  91 175 194 220 223 266 319 320 410 ... 11783 11838 ]
   elapsed time: 0.05824ms    (CUDA Measured)
    passed 
==== naive scan, non-power-of-two ====
   elapsed time: 0.011104ms    (CUDA Measured)
    passed 
==== work-efficient scan, power-of-two ====
   elapsed time: 0.172544ms    (CUDA Measured)
    passed 
==== work-efficient scan, non-power-of-two ====
   elapsed time: 0.036384ms    (CUDA Measured)
    passed 
==== thrust scan, power-of-two ====
   elapsed time: 0.048032ms    (CUDA Measured)
    passed 
==== thrust scan, non-power-of-two ====
   elapsed time: 0.015872ms    (CUDA Measured)
    passed 

*****************************
** STREAM COMPACTION TESTS **
*****************************
    [   2   2   1   2   0   3   2   3   3   1   1   2   0 ...   3   0 ]
==== cpu compact without scan, power-of-two ====
   elapsed time: 0.000421ms    (std::chrono Measured)
    [   2   2   1   2   3   2   3   3   1   1   2   1   2 ...   3   3 ]
    passed 
==== cpu compact without scan, non-power-of-two ====
   elapsed time: 0.000401ms    (std::chrono Measured)
    [   2   2   1   2   3   2   3   3   1   1   2   1   2 ...   2   1 ]
    passed 
==== cpu compact with scan ====
   elapsed time: 0.000982ms    (std::chrono Measured)
    [   2   2   1   2   3   2   3   3   1   1   2   1   2 ...   3   3 ]
    passed 
==== work-efficient compact, power-of-two ====
   elapsed time: 0.016672ms    (CUDA Measured)
    passed 
==== work-efficient compact, non-power-of-two ====
   elapsed time: 0.005056ms    (CUDA Measured)
    passed
```