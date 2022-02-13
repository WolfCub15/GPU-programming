#define BLOCK_SIZE 16

__kernel void KernelGemmFloatImage(int n, int m, int k, __read_only image2d_t a, __read_only image2d_t b, __write_only image2d_t c) {
	__local float A[BLOCK_SIZE][BLOCK_SIZE];
	__local float B[BLOCK_SIZE][BLOCK_SIZE];
	float ans = 0;

	int local_row = get_local_id(1);
	int local_col = get_local_id(0);
	int global_row = get_global_id(1);
	int global_col = get_global_id(0);

	int size = m / BLOCK_SIZE;

	for (int i = 0; i < size; ++i) {
		int col = i * BLOCK_SIZE + local_col;
		int row = i * BLOCK_SIZE + local_row;

		//копируем элемент (i,j) A (I,p)
		//копируем элемент (i,j) B (p,J)
		int2 curr_A = { col, global_row };
		int2 curr_B = { global_col, row };

		A[local_row][local_col] = read_imagef(a, curr_A).x;
		B[local_row][local_col] = read_imagef(b, curr_B).x;

		//__syncthreads ();
		barrier(CLK_LOCAL_MEM_FENCE);
		for (int ii = 0; ii < BLOCK_SIZE; ii++) {
			//произведение i-й строки AТ и j-й строки BТ
			ans += A[local_row][ii] * B[ii][local_col];
		}
		//__syncthreads ();
		barrier(CLK_LOCAL_MEM_FENCE);
	}

	int2 coord = { global_col, global_row };
	write_imagef(c, coord, ans);
}