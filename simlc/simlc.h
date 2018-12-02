#ifndef __SIM_HELPER_H
#define __SIM_HELPER_H

#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static float *front_buf, *back_buf;

#define SWAP_BUFFER(front, back) { \
	float *tmp = front; \
	front = back; \
	back = tmp; \
}

__host__ static inline void init_buffer(const size_t size)
{
	front_buf = (float*)malloc(sizeof(float) * size);
	back_buf = (float*)malloc(sizeof(float) * size);

	memset(front_buf, 0, sizeof(float) * size);
	memset(back_buf, 0, sizeof(float) * size);

	front_buf[0] = 1;
}

__host__ static void deinit_device(float * const front_dev, float * const back_dev)
{
	cudaFree(front_dev);
	cudaFree(back_dev);
}

__host__ static cudaError_t init_device(
	float * * const front_dev,
	float * * const back_dev,
	float const * const front_buf,
	float const * const back_buf,
	const size_t size)
{
	cudaError_t status;

	if ((status = cudaSetDevice(0)) != cudaSuccess)
	{
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		return status;
	}

	if ((status = cudaMalloc((void**)front_dev, size * sizeof(float))) != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed!");
		return status;
	}

	if ((status = cudaMalloc((void**)back_dev, size * sizeof(float))) != cudaSuccess)
	{
		fprintf(stderr, "cudaMalloc failed!");
		return status;
	}

	if ((status = cudaMemcpy(front_dev[0], front_buf, size * sizeof(float), cudaMemcpyHostToDevice)) != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy failed!");
		return status;
	}

	if ((status = cudaMemcpy(back_dev[0], back_buf, size * sizeof(float), cudaMemcpyHostToDevice)) != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy failed!");
		return status;
	}

	return status;
}

__host__ static cudaError_t sync_buffer_to_host(float * io, float const * const io_dev, const size_t size)
{
	cudaError_t status;

	if ((status = cudaMemcpy(io, io_dev, size * sizeof(float), cudaMemcpyDeviceToHost)) != cudaSuccess)
	{
		fprintf(stderr, "cudaMemcpy failed!");
		return status;
	}

	return status;
}

static inline void print_buffer(float const * const io, const size_t plates, const size_t mobiles)
{
	unsigned int i;

	for (i = 0; i < plates; i++)
	{
		printf("[%.2f]", io[i]);
	}

	printf("\n");
}

#endif