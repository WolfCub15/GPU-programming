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
#include <cassert>

std::mt19937 gen(std::chrono::high_resolution_clock::now().time_since_epoch().count());
#define BLOCK_SIZE 64
const int n = BLOCK_SIZE * 180;

//Kernel
void CheckReturnCode(cl_int ret, std::string msg) {
	if (ret != CL_SUCCESS) {
		std::cout << msg << "   return code: " << ret << '\n';
		exit(1);
	}
}

std::string ReadKernel(char* str) {
	std::ifstream is(str);
	std::string kernel, s;
	while (getline(is, s)) {
		kernel += s;
		kernel += '\n';
	}
	return kernel;
}
template<typename T>
void GenerateMatrix(T * a){
	for(int i = 0; i < n; ++i){
		for(int j = 0; j < n; ++j){
			T tmp = (gen() % 100);
			tmp /= n;
			if (i == j) {
				a[i * n + j] = 1000;
			}
			else {
				a[i * n + j] = tmp;
			}
		}
	}
}

template<typename T>
void GenerateVector(T* b) {
	for (int i = 0; i < n; i++) {
		T tmp = (gen() % 100);
		tmp /= n;
		b[i] = tmp;
	}
}

template<typename T>
bool CheckSum(T* a){
	for(int i = 0; i < n; ++i){
		T sum = 0;
		for (int j = 0; j < n; ++j) {
			sum += a[i * n + j];
		}
		sum -= a[i * n + i];
		if (sum > a[i * n + i]) return false;
	}
	return true;
}

template<typename T>
bool Check(T* a, T* b, T* x) {
	T* tmp = new T[n];
	T epsilon = 9e-5;
	for(int i = 0; i < n; ++i){
		tmp[i] = 0.0;
		for(int j = 0; j < n; ++j){
			//std::cout << a[i * n + j] << ' ' << x[j] << '\n';
			tmp[i] += a[i * n + j] * x[j];
		}
	}
	T ans = 0.0;
	for(int i = 0; i < n; ++i){
		//std::cout << tmp[i] << ' ' << b[i] << '\n';
		if (tmp[i] > epsilon) ans += (std::abs(tmp[i] - b[i]) / tmp[i]);
	}
	std::cout << ans << '\n';
	return ans < epsilon;
}

template<typename T>
void PrintMatrix(int n, int m, T* matrix, std::string type) {
	for (int i = 0; i < n; ++i) {
		for (int j = 0; j < m; ++j) {
			std::cout << ' ' << matrix[i * m + j];
		}
		std::cout << '\n';
	}
}

template<typename T>
double OpenCLJacobi(cl_device_id& device, int n, T* a, T* b, T* x0, T* x1, T* norma, char* file_name, char* kernel_name, int & counter, T & norma_sum) {
	std::string kernel_code = ReadKernel(file_name);
	size_t kernel_size = kernel_code.size();
	cl_int ret;

	cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &ret);
	//CheckReturnCode(ret, "clCreateContext");

	cl_command_queue command_queue = clCreateCommandQueueWithProperties(context, device, nullptr, &ret);
	//CheckReturnCode(ret, "clCreateCommandQueueWithProperties");

	cl_program program = clCreateProgramWithSource(context, 1, (const char**)&kernel_code, &kernel_size, &ret);
	//CheckReturnCode(ret, "clCreateProgramWithSource");

	ret = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
	//CheckReturnCode(ret, "clBuildProgram");

	cl_kernel kernel = clCreateKernel(program, kernel_name, &ret);
	//CheckReturnCode(ret, "clCreateKernel");

	cl_mem memObjA = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(T) * n * n, nullptr, &ret);
	//CheckReturnCode(ret, "clCreateBuffer A");
	cl_mem memObjB = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(T) * n, nullptr, &ret);
	//CheckReturnCode(ret, "clCreateBuffer B");
	cl_mem memObjX0 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(T) * n, nullptr, &ret);
	//CheckReturnCode(ret, "clCreateBuffer X0");
	cl_mem memObjX1 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(T) * n, nullptr, &ret);
	//CheckReturnCode(ret, "clCreateBuffer X1");
	cl_mem memObjNorma = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(T) * n, nullptr, &ret);
	//CheckReturnCode(ret, "clCreateBuffer norma");
	
	ret = clEnqueueWriteBuffer(command_queue, memObjA, CL_TRUE, 0, sizeof(T) * n * n, a, 0, nullptr, nullptr);
	//CheckReturnCode(ret, "clEnqueueWriteBuffer A");
	ret = clEnqueueWriteBuffer(command_queue, memObjB, CL_TRUE, 0, sizeof(T) * n, b, 0, nullptr, nullptr);
	//CheckReturnCode(ret, "clEnqueueWriteBuffer B");
	ret = clEnqueueWriteBuffer(command_queue, memObjX0, CL_TRUE, 0, sizeof(T) * n, x0, 0, nullptr, nullptr);
	//CheckReturnCode(ret, "clEnqueueWriteBuffer X0");
	ret = clEnqueueWriteBuffer(command_queue, memObjX1, CL_TRUE, 0, sizeof(T) * n, x1, 0, nullptr, nullptr);
	//CheckReturnCode(ret, "clEnqueueWriteBuffer X1");
	ret = clEnqueueWriteBuffer(command_queue, memObjNorma, CL_TRUE, 0, sizeof(T) * n, norma, 0, nullptr, nullptr);
	//CheckReturnCode(ret, "clEnqueueWriteBuffer norma");

	ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), &memObjA);
	//CheckReturnCode(ret, "clSetKernelArg 0");
	ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), &memObjB);
	//CheckReturnCode(ret, "clSetKernelArg 1");
	ret = clSetKernelArg(kernel, 4, sizeof(cl_mem), &memObjNorma);
	//CheckReturnCode(ret, "set kernel arg 4");

	size_t global_work_size[1] = { n };
	size_t group_size = BLOCK_SIZE;

	const T epsilon = 9e-5;
	cl_event event;

	double time = omp_get_wtime();

	do {
		ret = clSetKernelArg(kernel, 2, sizeof(cl_mem), &memObjX0);
		//CheckReturnCode(ret, "clSetKernelArg 2");
		ret = clSetKernelArg(kernel, 3, sizeof(cl_mem), &memObjX1);
		//CheckReturnCode(ret, "clSetKernelArg 3");

		ret = clEnqueueNDRangeKernel(command_queue, kernel, 1, nullptr, global_work_size, &group_size, 0, nullptr, &event);
		//CheckReturnCode(ret, "clEnqueueNDRangeKernel");
		clWaitForEvents(1, &event);

		ret = clEnqueueReadBuffer(command_queue, memObjNorma, CL_TRUE, 0, sizeof(T) * n, norma, 0, nullptr, nullptr);
		//CheckReturnCode(ret, "clEnqueueReadBuffer");

		norma_sum = 0;

		for (int i = 0; i < n; ++i) {
			//std::cout << "norma++" << '\n';
			norma_sum += (std::abs(norma[i])/x0[i]);
		}

		std::swap(x0, x1);
		std::swap(memObjX0, memObjX1);

	} while (counter-- && norma_sum > epsilon);
	
	time = omp_get_wtime() - time;

	ret = clEnqueueReadBuffer(command_queue, memObjX0, CL_TRUE, 0, sizeof(T) * n, x0, 0, nullptr, nullptr);
	//CheckReturnCode(ret, "clEnqueueReadBuffer");
	ret = clEnqueueReadBuffer(command_queue, memObjX1, CL_TRUE, 0, sizeof(T) * n, x1, 0, nullptr, nullptr);
	//CheckReturnCode(ret, "clEnqueueReadBuffer");

	clFinish(command_queue);

	clReleaseMemObject(memObjA);
	clReleaseMemObject(memObjB);
	clReleaseMemObject(memObjX0);
	clReleaseMemObject(memObjX1);
	clReleaseMemObject(memObjNorma);
	clReleaseProgram(program);
	clReleaseKernel(kernel);
	clReleaseCommandQueue(command_queue);
	clReleaseContext(context);
	return time;
}

template<typename T>
void run(cl_device_id& device, std::string type) {
	T* a = new T[n * n];
	T* b = new T[n];
	T* x0 = new T[n];
	T* x1 = new T[n];
	T* norma = new T[n];

	int counter = 400;
	T norma_sum = 0;

	GenerateMatrix(a);
	GenerateVector(b);
	
	char* file_name;
	char* kernel_name;
	for (int i = 0; i < n; ++i) {
		x0[i] = gen();
		x1[i] = 0;
	}
	if (type == "float") {
		file_name = (char*)"kernelFloat.cl";
		kernel_name = (char*)"jacobiFloat";
	}
	else {
		file_name = (char*)"kernelDouble.cl";
		kernel_name = (char*)"jacobiDouble";
	}
	

	if (!CheckSum(a)) {
		std::cout << "Не выполняется условие сходимости\n";
		return;
	}

	double time = OpenCLJacobi(device, n, a, b, x0, x1, norma, file_name, kernel_name, counter, norma_sum);

	if (Check(a, b, x1)) {
		std::cout << "Iter count =     " << counter << '\n';
		std::cout << "norma =          " << norma_sum << '\n';
		std::cout << "Time =           " << time << '\n';
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

int main() {
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

