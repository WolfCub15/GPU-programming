__kernel void gemmDouble( int n, int m, int k, __global const double* a, __global const double* b, __global double* c) {
	int i = get_global_id(1);
	int j = get_global_id(0);

	double sum = 0;
	for (int q = 0; q < m; ++q) {
		sum += a[i * m + q] * b[q * k + j];
	}
	c[i * k + j] = sum;
}

#define BLOCK_SIZE 16

__kernel void gemmDoubleBlock(int n, int m, int k, __global const double* a, __global const double* b, __global double* c) {
	__local double A[BLOCK_SIZE][BLOCK_SIZE];
	__local double B[BLOCK_SIZE][BLOCK_SIZE];
	double ans = 0;

	int local_row = get_local_id(1);
	int local_col = get_local_id(0);
	int global_row = get_global_id(1);
	int global_col = get_global_id(0);

	int size = m / BLOCK_SIZE;
	for (int i = 0; i < size; ++i) {
		int col = i * BLOCK_SIZE + local_col;
		int row = i * BLOCK_SIZE + local_row;

		A[local_row][local_col] = a[global_row * m + col];
		B[local_row][local_col] = b[row * k + global_col];

		barrier(CLK_LOCAL_MEM_FENCE);
		for (int ii = 0; ii < BLOCK_SIZE; ++ii) {
			ans += A[local_row][ii] * B[ii][local_col];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
	c[global_row * k + global_col] = ans;
}