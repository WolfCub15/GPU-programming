#include <CL/cl.h>
#include <iostream>
#include <cassert>
#include <random>
#include <iostream>
#include <omp.h>
#include <iomanip> 


double saxpy(int n, float a, float* x, int incx, float* y, int incy) {
	double start = omp_get_wtime();
	for (int i = 0; i < n; ++i) {
		y[i * incy] += a * x[i * incx];
	}
	start = omp_get_wtime() - start;
	return start;
}

double saxpyOMP(int n, float a, float* x, int incx, float* y, int incy) {
	double start = omp_get_wtime();
#pragma omp parallel shared(x,y)
	{
		#pragma omp for
		for (int i = 0; i < n; ++i) {
			y[i * incy] += a * x[i * incx];
		}
	}
	start = omp_get_wtime() - start;
	return start;
}

double daxpy(int n, double a, double* x, int incx, double* y, int incy) {
	double start = omp_get_wtime();
	for (int i = 0; i < n; ++i) {
		y[i * incy] += a * x[i * incx];
	}
	start = omp_get_wtime() - start;
	return start;
}

double daxpyOMP(int n, double a, double* x, int incx, double* y, int incy) {
	double start = omp_get_wtime();
#pragma omp parallel shared(x,y)
	{
		#pragma omp for
		for (int i = 0; i < n; ++i) {
			y[i * incy] += a * x[i * incx];
		}
	}
	start = omp_get_wtime() - start;
	return start;
}

const char* kernelSaxpy =
"__kernel void saxpy( int n,  float a, __global float * x,  int incx, __global float * y,  int incy) { \n"\
"int i = get_global_id(0);	\n"
"if (i < n){ y[i * incy] += a * x[i * incx];}		\n"\
"}";

const char* kernelDaxpy =
"__kernel void daxpy( int n,  double a, __global double * x,  int incx, __global double * y,  int incy) { \n"\
"int i = get_global_id(0);	\n"
"if (i < n){ y[i * incy] += a * x[i * incx];}		\n"\
"}";


double runKernelSaxpy(cl_device_id& device, int len, int incx, float* x, float a, int incy, float* y, size_t g, size_t &block) {

	cl_int ret;
	size_t source_size = strlen(kernelSaxpy);

	cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr);
	cl_queue_properties prop[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
	cl_command_queue command_queue = clCreateCommandQueueWithProperties(context, device, prop, &ret);

	cl_program program = clCreateProgramWithSource(context, 1, (const char**)&kernelSaxpy, (const size_t*)&source_size, nullptr);
	ret = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);

	cl_kernel kernel = clCreateKernel(program, "saxpy", nullptr);

	for (int w = 0; w < len; ++w) {
		y[w] = 0;
	}

	cl_mem 	memobjX = clCreateBuffer(context, CL_MEM_READ_ONLY, len * sizeof(float), nullptr, &ret);
	cl_mem memobjY = clCreateBuffer(context, CL_MEM_READ_WRITE, len * sizeof(float), nullptr, &ret);

	ret = clEnqueueWriteBuffer(command_queue, memobjX, CL_TRUE, 0, len * sizeof(float), x, 0, nullptr, nullptr);
	//ret = clEnqueueWriteBuffer(command_queue, memobjY, CL_TRUE, 0, len * sizeof(float), y, 0, nullptr, nullptr);

	ret = clSetKernelArg(kernel, 0, sizeof(int), (void*)&len);
	ret = clSetKernelArg(kernel, 1, sizeof(float), &a);
	ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), &memobjX);
	ret = clSetKernelArg(kernel, 3, sizeof(int), (void*)&incx);
	ret = clSetKernelArg(kernel, 4, sizeof(cl_mem), &memobjY);
	ret = clSetKernelArg(kernel, 5, sizeof(int), &incy);

	size_t work_size = (len / g + 1) * g;
	size_t global_work_size[1] = { work_size };
	block = global_work_size[1] / g;

	double start = omp_get_wtime();
	ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, nullptr, global_work_size, &g, 0, nullptr, nullptr);
	clFinish(command_queue);
	start = omp_get_wtime() - start;

	clEnqueueReadBuffer(command_queue, memobjY, CL_TRUE, 0, len * sizeof(float), y, 0, nullptr, nullptr);

	clReleaseMemObject(memobjY);
	clReleaseMemObject(memobjX);
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(command_queue);
	clReleaseContext(context);

	return start;
}

double  runKernelDaxpy(cl_device_id& device, int len, int incx, double* x, double a, int incy, double* y, size_t g, size_t &block) {

	cl_int ret;
	size_t source_size = strlen(kernelDaxpy);

	cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, nullptr);
	cl_queue_properties prop[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
	cl_command_queue command_queue = clCreateCommandQueueWithProperties(context, device, prop, &ret);

	cl_program program = clCreateProgramWithSource(context, 1, (const char**)&kernelSaxpy, (const size_t*)&source_size, nullptr);
	ret = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);

	cl_kernel kernel = clCreateKernel(program, "daxpy", nullptr);

	for (int w = 0; w < len; ++w) {
		y[w] = 0;
	}

	cl_mem 	memobjX = clCreateBuffer(context, CL_MEM_READ_ONLY, len * sizeof(double), nullptr, &ret);
	cl_mem memobjY = clCreateBuffer(context, CL_MEM_READ_WRITE, len * sizeof(double), nullptr, &ret);

	ret = clEnqueueWriteBuffer(command_queue, memobjX, CL_TRUE, 0, len * sizeof(double), x, 0, nullptr, nullptr);

	ret = clSetKernelArg(kernel, 0, sizeof(int), (void*)&len);
	ret = clSetKernelArg(kernel, 1, sizeof(double), &a);
	ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), &memobjX);
	ret = clSetKernelArg(kernel, 3, sizeof(int), (void*)&incx);
	ret = clSetKernelArg(kernel, 4, sizeof(cl_mem), &memobjY);
	ret = clSetKernelArg(kernel, 5, sizeof(int), &incy);

	size_t work_size = (len / g + 1) * g;
	size_t global_work_size[1] = { work_size };
	block = global_work_size[1] / g;

	double start = omp_get_wtime();
	ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, nullptr, global_work_size, &g, 0, nullptr, nullptr);
	clFinish(command_queue);
	start = omp_get_wtime() - start;

	clEnqueueReadBuffer(command_queue, memobjY, CL_TRUE, 0, len * sizeof(double), y, 0, nullptr, nullptr);

	clReleaseMemObject(memobjY);
	clReleaseMemObject(memobjX);
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(command_queue);
	clReleaseContext(context);

	return start;
}

template<typename T>
void genData(T* &input, int & n, T& coeff) {
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> dis(1, 10000);
	coeff = dis(gen);
	input = new T[n];
	for (int i = 0; i < n; ++i) {
		input[i] = dis(gen);
	}
}


const int mi = 1e6;
const int ma = 3e6;
const int step = 1e6;

void runGPU(cl_platform_id& platform) {
	cl_uint deviceCount = 0;
	clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &deviceCount);
	std::cout << "device count :    " << deviceCount << std::endl;

	cl_device_id* devices = new cl_device_id[deviceCount];
	clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, devices, &deviceCount);

	for (cl_uint j = 0; j < deviceCount; ++j) {
		char deviceName[128];
		clGetDeviceInfo(devices[j], CL_DEVICE_NAME, 128, deviceName, nullptr);
		std::cout << std::string(deviceName) << '\n';

		for (int g = 8; g <= 256; g <<= 1) {
			for (int len = mi; len < ma; len += step) {
				float* x;
				int n = len;
				float a;
				genData(x, n, a);
				int incx = 1, incy = 1;
				size_t b;

				float* y = new float[n];
				for (int ii = 0; ii < n; ++ii) {
					y[ii] = 0;
				}

				double time = runKernelSaxpy(devices[j], n, incx, x, a, incy, y, g, b);

				std::cout << "GPU OpenCL: float :   len = " << len <<"   " << g  << " time:   " << time << std::fixed << std::setprecision(20) << '\n';
			}
		}
		for (int g = 8; g <= 256; g <<= 1) {
			for (int len = mi; len < ma; len += step) {
				double* x;
				int n = len;
				double a;
				genData(x, n, a);

				int incx = 1, incy = 1;
				size_t b;

				double* y = new double[n];
				for (int ii = 0; ii < n; ++ii) {
					y[ii] = 0;
				}
				double time = runKernelDaxpy(devices[j], n, incx, x, a, incy, y, g, b);

				std::cout << "GPU OpenCL: double :   len = " << len << "   " << g <<  " time:   " << time << std::fixed << std::setprecision(20) << '\n';

			}
		}
	}
}

void runCPU(cl_platform_id& platform) {
	cl_uint deviceCount = 0;
	clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 0, nullptr, &deviceCount);
	std::cout << "device count :    " << deviceCount << std::endl;

	cl_device_id* devices = new cl_device_id[deviceCount];
	clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, devices, &deviceCount);

	for (cl_uint j = 0; j < deviceCount; ++j) {
		char deviceName[128];
		clGetDeviceInfo(devices[j], CL_DEVICE_NAME, 128, deviceName, nullptr);
		std::cout << std::string(deviceName) << '\n';

		for (int g = 8; g <= 256; g <<= 1) {
			for (int len = mi; len < ma; len += step) {
				float* x;
				int n = len;
				float a;
				genData(x, n, a);
				int incx = 1, incy = 1;
				size_t b;
				float* y = new float[n];
				for (int ii = 0; ii < n; ++ii) {
					y[ii] = 0;
				}

				double time = runKernelSaxpy(devices[j], n, incx, x, a, incy, y, g, b);

				std::cout << "CPU OpenCL: float :   len = " << len << "   " << g <<  " time:   " << time << std::fixed << std::setprecision(20) << '\n';
			}
		}
		for (int g = 8; g <= 256; g <<= 1) {
			for (int len = mi; len < ma; len += step) {
				double* x;
				int n = len;
				double a;
				genData(x, n, a);

				int incx = 1, incy = 1;
				size_t b;

				double* y = new double[n];
				for (int ii = 0; ii < n; ++ii) {
					y[ii] = 0;
				}
				double time = runKernelDaxpy(devices[j], n, incx, x, a, incy, y, g, b);

				std::cout << "CPU OpenCL: double :   len = " << len << "   " << g << " time:   " << time << std::fixed << std::setprecision(20) << '\n';

			}
		}
	}
}


int main() {
	cl_uint platformCount = 0;

	clGetPlatformIDs(0, nullptr, &platformCount);

	if (platformCount > 0) {
		cl_platform_id* platforms = new cl_platform_id[platformCount];
		clGetPlatformIDs(platformCount, platforms, nullptr);

		for (cl_uint i = 0; i < platformCount; ++i) {
			char platformName[128];
			clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 128, platformName, nullptr);
			std::cout << platformName << std::endl;

			runGPU(platforms[i]);
			runCPU(platforms[i]);
			
		}
	}

	for (int len = mi; len < ma; len += step) {
		float* x;
		int n = len;
		float a;
		genData(x, n, a);
		int incx = 1, incy = 1;

		float* y = new float[n];
		for (int ii = 0; ii < n; ++ii) {
			y[ii] = 0;
		}
		double time = saxpy(n, a, x, incx, y, incy);
		std::cout << "Sequent: float :   len = " << len << " time:   " << time << std::fixed << std::setprecision(20) << '\n';

	}
	
	for (int len = mi; len < ma; len += step) {
		double* x;
		int n = len;
		double a;
		genData(x, n, a);

		int incx = 1, incy = 1;

		double* y = new double[n];
		for (int ii = 0; ii < n; ++ii) {
			y[ii] = 0;
		}
		double time = daxpy(n, a, x, incx, y, incy);

		std::cout << "Sequent: double :   len = " << len << " time:   " << time << std::fixed << std::setprecision(20) << '\n';

	}
	for (int len = mi; len < ma; len += step) {
		float* x;
		int n = len;
		float a;
		genData(x, n, a);
		int incx = 1, incy = 1;

		float* y = new float[n];
		for (int ii = 0; ii < n; ++ii) {
			y[ii] = 0;
		}
		double time = saxpyOMP(n, a, x, incx, y, incy);
		std::cout << "OpenMP: float :   len = " << len << " time:   " << time << std::fixed << std::setprecision(20) << '\n';

	}
	for (int len = mi; len < ma; len += step) {
		double* x;
		int n = len;
		double a;
		genData(x, n, a);

		int incx = 1, incy = 1;

		double* y = new double[n];
		for (int ii = 0; ii < n; ++ii) {
			y[ii] = 0;
		}
		double time = daxpyOMP(n, a, x, incx, y, incy);

		std::cout <<"OpenMP: double :   len = " << len << " time:   " << time << std::fixed << std::setprecision(20) << '\n';

	}
	

	return 0;
}