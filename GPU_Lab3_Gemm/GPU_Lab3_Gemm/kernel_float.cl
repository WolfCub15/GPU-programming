__kernel void KernelGemmFloat( int n, int m, int k, __global const float* a, __global const float* b, __global float* c) {
	int i = get_global_id(1);
	int j = get_global_id(0);

	float sum = 0;
	for (int q = 0; q < m; ++q) {
		sum += a[i * m + q] * b[q * k + j];
	}
	c[i * k + j] = sum;
	
}