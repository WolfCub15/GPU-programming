__kernel void jacobiDouble(__global const double* a, __global const double* b, __global double* x0,
	__global double* x1, __global double* norma, int n, int step) {
	const int i = get_global_id(0);
	double ans = 0;

	for (int qq = 0; qq < n; ++qq) {
		ans += a[i * n + qq] * x0[qq] * (double)(qq != (i + step));
	}

	x1[i + step] = (b[i + step] - ans) / a[i * n + i + step];
	norma[i + step] = x1[i + step] - x0[i + step];

	if (x0[i + step] > 0.0001) {
		norma[i + step] /= x0[i + step];
	}
}