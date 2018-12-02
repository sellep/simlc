
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#include <stdio.h>
#include <stdlib.h>
#include "cuda_helper.h"
#include "simlc.h"

__global__ void sim_lc(float * const front, float * const back, const size_t plates, const size_t mobiles)
{
	int i = X();

	if (i >= plates - 1)
		return;

	// move mobile phase
	for (int m = 0; m < mobiles; m++)
	{
		int j = m * plates + i;
		float val = front[j];
		front[j] = 0;
		back[j + 0] += 0.2f * val;
		back[j + 1] += 0.8f * val;
	}
}

__host__ cudaError_t invoke_sim_lc(
	const dim3 grid,
	const dim3 block,
	float * const front,
	float * const back,
	const size_t plates,
	const size_t mobiles)
{
	cudaError_t status;

	sim_lc << <grid, block >> > (front, back, plates, mobiles);

	if ((status = cudaGetLastError()) != cudaSuccess)
	{
		fprintf(stderr, "cudaGetLastError(): %s\n", cudaGetErrorString(status));
		return status;
	}

	if ((status = cudaDeviceSynchronize()) != cudaSuccess)
	{
		fprintf(stderr, "cudaDeviceSynchronize(): %s\n", cudaGetErrorString(status));
		return status;
	}

	return status;
}

int main()
{
    const size_t plates = 20;
	const size_t mobiles = 1;
	const size_t io_size = plates * mobiles;

	dim3 grid(io_size / 32 + 1), block(32);

	init_buffer(io_size);

	float *front_dev, *back_dev;
	if (init_device(&front_dev, &back_dev, front_buf, back_buf, io_size) != cudaSuccess)
		goto exit;

	while (true)
	{
		if ((invoke_sim_lc(grid, block, front_dev, back_dev, plates, mobiles)) != cudaSuccess)
			goto cleanup;

		if (sync_buffer_to_host(front_buf, back_dev, io_size) != cudaSuccess)
			goto cleanup;

		SWAP_BUFFER(front_dev, back_dev);

		print_buffer(front_buf, plates, mobiles);

		getchar();
	}

cleanup:
	deinit_device(front_dev, back_dev);

exit:
    return 0;
}