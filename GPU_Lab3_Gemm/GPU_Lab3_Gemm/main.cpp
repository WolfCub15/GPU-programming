#include <CL/cl.h>
#include <iostream>
#include <omp.h>
#include <random>
#include <chrono>
#include <cassert>
#include <cassert>
#include <random>
#include <omp.h>
#include <iomanip> 
#include <vector>
#include <string>
#include <istream>
#include <fstream>

#define BLOCK_SIZE 16

int n = BLOCK_SIZE * 80;
int m = BLOCK_SIZE * 80;
int k = BLOCK_SIZE * 80;

//Matrix
template<typename T>
void GenMatrix(T*& input, int n) {
	std::random_device rd;
	std::mt19937 gen(rd());
	std::uniform_real_distribution<> dis(-100.0, 100.0);
	for (int i = 0; i < n; ++i) {
		input[i] = dis(gen);
	}
}

template<typename T>
void PrintMatrix(T* A, int n, int  m) {
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < m; ++j) {
			std::cout << A[i * n + j] << ' ';
		}
		std::cout << '\n';
	}
}

template<typename T>
void NullMatrix(int n, T* a) {
	for (int i = 0; i < n; ++i) {
		a[i] = T(0);
	}
}

//Gemm
template<typename T>
double SequentialGemm(int n, int m, int k, const T const * a, const T const * b, T* c) {
	double time = omp_get_wtime();
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < k; ++j) {
			T sum = 0;
			for (int q = 0; q < m; ++q) {
				sum += a[i * m + q] * b[q * k + j];
			}
			c[i * k + j] = sum;
		}
	}
	time = omp_get_wtime() - time;
	return time;
}

template<typename T>
double OmpGemm(int n, int m, int k, const T const* a, const T const* b, T* c) {
	int i, j, q;
	T sum;
	double time = omp_get_wtime();
#pragma omp parallel for shared(n, m, k, a, b, c) private(i, j, q, sum)
	for (i = 0; i < n; ++i) {
		for (q = 0; q < m; ++q) {
			sum = 0.0;
			for (j = 0; j < k; ++j) {
				sum += a[i * m + q] * b[q * k + j];
			}
			c[i * k + j] = sum;
		}
	}
	time = omp_get_wtime() - time;
	return time;
}

template<typename T>
double OmpGemmBlock(int n, int m, int k, const T const* a, const T const* b, T* c) {
	int block_i, block_j, block_q, i, j, q;
	int block_count_n = n / BLOCK_SIZE;
	int block_count_k = k / BLOCK_SIZE;
	int block_count_m = m / BLOCK_SIZE;
	double time = omp_get_wtime();
#pragma omp parallel for shared(block_count_n, block_count_k, block_count_m, a, b, c) private(block_i, block_j, block_q, i, j, q) collapse(6)
	for (block_i = 0; block_i < block_count_n; ++block_i) {
		for (block_q = 0; block_q < block_count_m; ++block_q) {
			for (block_j = 0; block_j < block_count_k; ++block_j) {
				for (i = block_i * BLOCK_SIZE; i < (block_i + 1) * BLOCK_SIZE; ++i) {
					for (q = block_q * BLOCK_SIZE; q < (block_q + 1) * BLOCK_SIZE; ++q) {
						for (j = block_j * BLOCK_SIZE; j < (block_j + 1) * BLOCK_SIZE; ++j) {
							c[i * k + j] += a[i * m + q] * b[q * k + j];
						}
					}
				}
			}
		}
	}

	time = omp_get_wtime() - time;
	return time;
}

//Kernel
void CheckReturnCode(cl_int ret, std::string msg) {
	if (ret != CL_SUCCESS) {
		std::cout << msg << "   return code: " << ret << '\n';
		exit(1);
	}
}

std::string ReadKernel(char* filename) {
	std::ifstream is(filename);
	std::string kernel, s;
	while (getline(is, s)) {
		kernel += s;
		kernel += '\n';
	}
	return kernel;
}

template<typename T>
double  runKernel(cl_device_id& device, T* a, T* b, T* c, int n, int m, int k, char* filename, char* kernel_name) {
	std::string kernel_gemm = ReadKernel(filename);
	size_t kernel_size = kernel_gemm.size();
	
	cl_int ret;
	cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &ret);
	//CheckReturnCode(ret, "clCreateContext");

	cl_command_queue command_queue = clCreateCommandQueueWithProperties(context, device, nullptr, &ret);
	//CheckReturnCode(ret, "clCreateCommandQueueWithProperties");

	cl_program program = clCreateProgramWithSource(context, 1, (const char**)&kernel_gemm, &kernel_size, &ret);
	//CheckReturnCode(ret, "clCreateProgramWithSource");

	ret = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);

	cl_kernel kernel = clCreateKernel(program, kernel_name, &ret);
	//CheckReturnCode(ret, "clCreateKernel");

	cl_mem memObjA = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(T) * n * m, nullptr, &ret);
	//CheckReturnCode(ret, "clCreateBuffer A");

	cl_mem memObjB = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(T) * m * k, nullptr, &ret);
	//CheckReturnCode(ret, "clCreateBuffer B");

	cl_mem memObjC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(T) * n * k, nullptr, &ret);
	//CheckReturnCode(ret, "clCreateBuffer C");

	ret = clEnqueueWriteBuffer(command_queue, memObjA, CL_TRUE, 0, sizeof(T) * n * m, a, 0, nullptr, nullptr);
	//CheckReturnCode(ret, "clEnqueueWriteBuffer A");

	ret = clEnqueueWriteBuffer(command_queue, memObjB, CL_TRUE, 0, sizeof(T) * m * k, b, 0, nullptr, nullptr);
	//CheckReturnCode(ret, "clEnqueueWriteBuffer B");

	ret = clSetKernelArg(kernel, 0, sizeof(int), &n);
	//CheckReturnCode(ret, "clSetKernelArg 0");

	ret = clSetKernelArg(kernel, 1, sizeof(int), &m);
	//CheckReturnCode(ret, "clSetKernelArg 1");

	ret = clSetKernelArg(kernel, 2, sizeof(int), &k);
	//CheckReturnCode(ret, "clSetKernelArg 2");

	ret = clSetKernelArg(kernel, 3, sizeof(cl_mem), &memObjA);
	//CheckReturnCode(ret, "clSetKernelArg 3");

	ret = clSetKernelArg(kernel, 4, sizeof(cl_mem), &memObjB);
	//CheckReturnCode(ret, "clSetKernelArg 4");

	ret = clSetKernelArg(kernel, 5, sizeof(cl_mem), &memObjC);
	//CheckReturnCode(ret, "clSetKernelArg 5");

	size_t global_work_size[2] = { n, k };
	size_t group_size[2] = { BLOCK_SIZE, BLOCK_SIZE };

	double time = omp_get_wtime();
	ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, nullptr, global_work_size, group_size, 0, nullptr, nullptr);
	clFinish(command_queue);
	time = omp_get_wtime() - time;
	//CheckReturnCode(ret, "clEnqueueNDRangeKernel");

	ret = clEnqueueReadBuffer(command_queue, memObjC, CL_TRUE, 0, sizeof(T) * n * k, c, 0, nullptr, nullptr);
	//CheckReturnCode(ret, "clEnqueueReadBuffer");

	clReleaseMemObject(memObjA);
	clReleaseMemObject(memObjB);
	clReleaseMemObject(memObjC);
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(command_queue);
	clReleaseContext(context);
	return time;
}

template<typename T>
double  runKernelImage(cl_device_id& device, T* a, T* b, T* c, int n, int m, int k, char* filename, char* kernel_name) {
	std::string kernel_gemm = ReadKernel(filename);
	size_t kernel_size = kernel_gemm.size();

	cl_int ret;
	cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &ret);
	//CheckReturnCode(ret, "clCreateContext");

	cl_command_queue command_queue = clCreateCommandQueueWithProperties(context, device, nullptr, &ret);
	//CheckReturnCode(ret, "clCreateCommandQueueWithProperties");

	cl_program program = clCreateProgramWithSource(context, 1, (const char**)&kernel_gemm, &kernel_size, &ret);
	//CheckReturnCode(ret, "clCreateProgramWithSource");
	ret = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);

	cl_kernel kernel = clCreateKernel(program, kernel_name, &ret);
	//CheckReturnCode(ret, "clCreateKernel");

	const cl_image_format format = { CL_R, CL_FLOAT };
	const cl_image_desc descA = { CL_MEM_OBJECT_IMAGE2D, m, n, 1, 1, 0, 0, 0, 0 };
	const cl_image_desc descB = { CL_MEM_OBJECT_IMAGE2D, k, m, 1, 1, 0, 0, 0, 0 };
	const cl_image_desc descC = { CL_MEM_OBJECT_IMAGE2D, k, n, 1, 1, 0, 0, 0, 0 };

	cl_mem bufferA = clCreateImage(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &format, &descA, (void*)a, &ret);
	//CheckReturnCode(ret, "clCreateImage A");

	cl_mem bufferB = clCreateImage(context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, &format, &descB, (void*)b, &ret);
	//CheckReturnCode(ret, "clCreateImage B");

	cl_mem bufferC = clCreateImage(context, CL_MEM_WRITE_ONLY | CL_MEM_COPY_HOST_PTR, &format, &descC, (void*)c, &ret);
	//CheckReturnCode(ret, "clCreateImage C");

	ret = clSetKernelArg(kernel, 0, sizeof(int), &n);
	//CheckReturnCode(ret, "clSetKernelArg 0");

	ret = clSetKernelArg(kernel, 1, sizeof(int), &m);
	//CheckReturnCode(ret, "clSetKernelArg 1");

	ret = clSetKernelArg(kernel, 2, sizeof(int), &k);
	//CheckReturnCode(ret, "clSetKernelArg 2");

	ret = clSetKernelArg(kernel, 3, sizeof(cl_mem), &bufferA);
	//CheckReturnCode(ret, "clSetKernelArg 3");

	ret = clSetKernelArg(kernel, 4, sizeof(cl_mem), &bufferB);
	//CheckReturnCode(ret, "clSetKernelArg 4");

	ret = clSetKernelArg(kernel, 5, sizeof(cl_mem), &bufferC);
	//CheckReturnCode(ret, "clSetKernelArg 5");

	size_t global_work_size[2] = { n, k };
	size_t group_size[2] = { BLOCK_SIZE, BLOCK_SIZE };

	double time = omp_get_wtime();
	ret = clEnqueueNDRangeKernel(command_queue, kernel, 2, nullptr, global_work_size, group_size, 0, nullptr, nullptr);
	clFinish(command_queue);
	time = omp_get_wtime() - time;
	//CheckReturnCode(ret, "clEnqueueNDRangeKernel");

	const size_t origin[] = { 0, 0, 0 };
	const size_t region[] = { k, n, 1 };

	ret = clEnqueueReadImage(command_queue, bufferC, CL_TRUE, origin, region, 0, 0, c, 0, nullptr, nullptr);
	//CheckReturnCode(ret, "clEnqueueReadImage");

	clReleaseMemObject(bufferA);
	clReleaseMemObject(bufferB);
	clReleaseMemObject(bufferC);
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(command_queue);
	clReleaseContext(context);
	return time;
}

template<typename T>
void run(cl_device_id& device, std::string type) {
	T *A = new T[n * m];
	T *B = new T[m * k];
	int C_size = n * k;
	T *C_sequential = new T[C_size];
	T *C_omp = new T[C_size];
	T *C_omp_block = new T[C_size];
	T *C_opencl = new T[C_size];
	T *C_opencl_block = new T[C_size];
	T *C_opencl_image = new T[C_size];
	char* file, *file_block, *file_image, *kernel, *kernel_block, *kernel_image;

	GenMatrix(A, n * m);
	GenMatrix(B, m * k);

	NullMatrix(C_size, C_sequential);
	NullMatrix(C_size, C_omp);
	NullMatrix(C_size, C_omp_block);
	NullMatrix(C_size, C_opencl);
	NullMatrix(C_size, C_opencl_block);
	NullMatrix(C_size, C_opencl_image);
	
	if (type == "float") {
		file = (char*)"kernel_float.cl";
		kernel = (char*)"KernelGemmFloat";
		file_block = (char*)"kernel_block_float.cl";
		kernel_block = (char*)"KernelGemmFloatBlock";
		file_image = (char*)"kernel_image_float.cl";
		kernel_image = (char*)"KernelGemmFloatImage";
	}
	else {
		file = (char*)"kernel_double.cl";
		kernel = (char*)"KernelGemmDouble";
		file_block = (char*)"kernel_block_double.cl";
		kernel_block = (char*)"KernelGemmDoubleBlock";
		file_image = (char*)"kernel_image_float.cl";
		kernel_image = (char*)"KernelGemmFloatImage";
	}

	double sequential_time = SequentialGemm(n, m, k, A, B, C_sequential);
	std::cout << "Sequential gemm: \t" << sequential_time << '\n';

	double omp_time = OmpGemm(n, m, k, A, B, C_omp);
	std::cout << "OMP gemm: \t\t" << omp_time << '\n';
	
	//double omp_block_time = OmpGemmBlock(n, m, k, A, B, C_omp_block);
	//std::cout << "OMP gemm block: \t" << omp_block_time << '\n';
	
	double opencl_time = runKernel<T>(device, A, B, C_opencl, n, m, k, file, kernel);
	std::cout << "OpenCl gemm: \t\t" << opencl_time << '\n';
	
	double opencl_block_time = runKernel<T>(device, A, B, C_opencl_block, n, m, k, file_block, kernel_block);
	std::cout << "OpenCl gemm block: \t" << opencl_block_time << '\n';

	if (type == "float") {
		double opencl_image_time = runKernelImage(device, A, B, C_opencl_image, n, m, k, file_image, kernel_image);
		std::cout << "OpenCl gemm image: \t" << opencl_image_time << '\n';
	}
}

void runGPU(cl_platform_id& platform) {
	cl_uint deviceCount = 0;
	clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &deviceCount);
	//std::cout << "device count :    " << deviceCount << std::endl;

	cl_device_id* devices = new cl_device_id[deviceCount];
	clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, devices, &deviceCount);

	for (cl_uint j = 0; j < deviceCount; ++j) {
		char deviceName[128];
		clGetDeviceInfo(devices[j], CL_DEVICE_NAME, 128, deviceName, nullptr);
		std::cout << "\n******************************" << std::string(deviceName) << "******************************" << '\n';

		std::cout << "GPU Float: \n\n";
		run<float>(devices[j], "float");
		std::cout << "\nGPU Double: \n\n";
		run<double>(devices[j], "double");
	}
}

void runCPU(cl_platform_id& platform) {
	cl_uint deviceCount = 0;
	clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 0, nullptr, &deviceCount);
	//std::cout << "device count :    " << deviceCount << std::endl;

	cl_device_id* devices = new cl_device_id[deviceCount];
	clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, 1, devices, &deviceCount);

	for (cl_uint j = 0; j < deviceCount; ++j) {
		char deviceName[128];
		clGetDeviceInfo(devices[j], CL_DEVICE_NAME, 128, deviceName, nullptr);
		std::cout << "\n******************************" << std::string(deviceName) << "******************************" << '\n';

		std::cout << "CPU Float: \n\n";
		run<float>(devices[j], "float");
		std::cout << "\nCPU Double: \n\n";
		run<double>(devices[j], "double");
	}
}

int main(){
	std::cout << std::fixed << std::setprecision(20);
	cl_uint platformCount = 0;

	clGetPlatformIDs(0, nullptr, &platformCount);

	if (platformCount > 0) {
		cl_platform_id* platforms = new cl_platform_id[platformCount];
		clGetPlatformIDs(platformCount, platforms, nullptr);

		for (cl_uint i = 0; i < platformCount; ++i) {
			char platformName[128];
			clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 128, platformName, nullptr);
			std::cout << "\n_______________________________________________________________________________\n";
			std::cout << platformName;
			std::cout << "\n_______________________________________________________________________________\n";

			runGPU(platforms[i]);
			runCPU(platforms[i]);
		}
	}
	return 0;
}

