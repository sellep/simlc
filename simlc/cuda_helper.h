#ifndef __CUDA_HOST_HELPER_H
#define __CUDA_HOST_HELPER_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>

#define X() (blockDim.x * blockIdx.x + threadIdx.x)

__host__ static void print_dev_props(int device)
{
	cudaDeviceProp props;
	cudaGetDeviceProperties(&props, device);

	printf("Major revision number:         %d\n", props.major);
	printf("Minor revision number:         %d\n", props.minor);
	printf("Name:                          %s\n", props.name);
	printf("Total global memory:           %u\n", props.totalGlobalMem);
	printf("Total shared memory per block: %u\n", props.sharedMemPerBlock);
	printf("Total registers per block:     %d\n", props.regsPerBlock);
	printf("Warp size:                     %d\n", props.warpSize);
	printf("Maximum memory pitch:          %u\n", props.memPitch);
	printf("Maximum threads per block:     %d\n", props.maxThreadsPerBlock);
	for (int i = 0; i < 3; ++i)
		printf("Maximum dimension %d of block:  %d\n", i, props.maxThreadsDim[i]);
	for (int i = 0; i < 3; ++i)
		printf("Maximum dimension %d of grid:   %d\n", i, props.maxGridSize[i]);
	printf("Clock rate:                    %d\n", props.clockRate);
	printf("Total constant memory:         %u\n", props.totalConstMem);
	printf("Texture alignment:             %u\n", props.textureAlignment);
	printf("Concurrent copy and execution: %s\n", (props.deviceOverlap ? "Yes" : "No"));
	printf("Number of multiprocessors:     %d\n", props.multiProcessorCount);
	printf("Kernel execution timeout:      %s\n", (props.kernelExecTimeoutEnabled ? "Yes" : "No"));
}

#endif