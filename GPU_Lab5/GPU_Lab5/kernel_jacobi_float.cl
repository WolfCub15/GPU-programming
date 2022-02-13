__kernel void jacobiFloat(__global const float* a, __global const float* b, __global float* x0,
	__global float* x1, __global float* norma, int n, int step) {
	const int i = get_global_id(0);
	float ans = 0;

	for (int qq = 0; qq < n; ++qq) {
		ans += a[i * n + qq] * x0[qq] * (float)(qq != (i + step));
	}

	x1[i + step] = (b[i + step] - ans) / a[i * n + i + step];
	norma[i + step] = x1[i + step] - x0[i + step];

	if (x0[i + step] > 0.0001) {
		norma[i + step] /= x0[i + step];
	}
}