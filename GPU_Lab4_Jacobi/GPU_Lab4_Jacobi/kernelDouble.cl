__kernel void jacobiDouble(__global const double* a, __global const double* b, __global double* x0, __global double* x1, __global double* norma) {

	const int i = get_global_id(0);
	const int n = get_global_size(0);

	double ans = 0;

	for (int qq = 0; qq < n; ++qq) {
		ans += a[qq * n + i] * x0[qq] * (double)(qq != i);
	}

	x1[i] = (b[i] - ans) / a[i * n + i];
	if (x0[i] > 9e-5)
		norma[i] = (x0[i] - x1[i]) / x0[i];
}