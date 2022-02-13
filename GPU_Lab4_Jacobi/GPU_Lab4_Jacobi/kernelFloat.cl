__kernel void jacobiFloat(__global const float* a, __global const float* b, __global float* x0, __global float* x1,  __global float* norma) {
	const int i = get_global_id(0);
	const int n = get_global_size(0);

	float ans = 0;

	for(int qq = 0; qq < n; ++qq){
		ans += a[qq * n + i] * x0[qq] * (float)(qq != i);
	}

	x1[i] = (b[i] - ans) / a[i * n + i];
	if (x0[i] > 9e-5)
		norma[i] = (x0[i] - x1[i]) / x0[i];
}