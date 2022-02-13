#define BLOCK_SIZE 16

__kernel void KernelGemmFloatBlock(int n, int m, int k, __global const float* a, __global const float* b, __global float* c) {
	__local float A[BLOCK_SIZE][BLOCK_SIZE];
	__local float B[BLOCK_SIZE][BLOCK_SIZE];
	float ans = 0;

	int local_row = get_local_id(1);
	int local_col = get_local_id(0);
	int global_row = get_global_id(1);
	int global_col = get_global_id(0);

	int size = m / BLOCK_SIZE;
	for (int i = 0; i < size; ++i) {
		int row = i * BLOCK_SIZE + local_row;
		int col = i * BLOCK_SIZE + local_col;

		A[local_row][local_col] = a[global_row * m + col];
		B[local_row][local_col] = b[row * k + global_col];

		barrier(CLK_LOCAL_MEM_FENCE);
		for (int ii = 0;ii < BLOCK_SIZE; ++ii){
			ans += A[local_row][ii] * B[ii][local_col];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	c[global_row * k + global_col] = ans;
}