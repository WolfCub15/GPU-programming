__kernel void KernelGemmDouble( int n, int m, int k, __global const double* a, __global const double* b, __global double* c) {
	int i = get_global_id(1);
	int j = get_global_id(0);

	double sum = 0;
	for (int q = 0; q < m; ++q) {
		sum += a[i * m + q] * b[q * k + j];
	}
	c[i * k + j] = sum;
}