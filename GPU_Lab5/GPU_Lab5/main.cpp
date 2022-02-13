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
//#include "gemm.h"
#include "jacobi.h"
#include "utils.h"

/*-------------------------------------------------------------------------------------------------------------------*/
/*-------------------------------------------------------------------------------------------------------------------*/
/*-------------------------------------------------------------------------------------------------------------------*/
/*--------------------------------------------------------GEMM-------------------------------------------------------*/
/*-------------------------------------------------------------------------------------------------------------------*/
/*-------------------------------------------------------------------------------------------------------------------*/
/*-------------------------------------------------------------------------------------------------------------------*/

//int n = 32 * 200, m = 16 * 100, k = 16 * 100;

template<typename T>
void runGemm(std::string type){
	std::cout << "**********************************" << type << "**********************************\n";
	char* file_name = (char*)"";
	char* kernel_name = (char*)"";
	char* kernel_name_block = (char*)"";
	char* kernel_name_image = (char*)"";

	if (type == "float") {
		file_name = (char*)"kernel_gemm_float.cl";
		kernel_name = (char*)"gemmFloat";
		kernel_name_block = (char*)"gemmFloatBlock";
		kernel_name_image = (char*)"gemmFloatImage";
	}
	else if (type == "double") {
		file_name = (char*)"kernel_gemm_double.cl";
		kernel_name = (char*)"gemmDouble";
		kernel_name_block = (char*)"gemmDoubleBlock";
		kernel_name_image = (char*)"gemmFloatImage";
	}
	
	int C_size = n * k;

	T* a = new T[n * m];
	T* b = new T[m * k];
	T* c_seq = new T[C_size];
	T* c_opencl_cpu = new T[C_size];
	T* c_opencl_cpu_block = new T[C_size];
	T* c_opencl_gpu_block = new T[C_size];
	T* c_opencl_cpu_image = new T[C_size];
	T* c_opencl_gpu_image = new T[C_size];
	T* c_opencl_gpu = new T[C_size];
	T* c_opencl_block = new T[C_size];
	T* c_opencl_image = new T[C_size];
	T* c_opencl_cpu_gpu = new T[C_size];

	GenerateMatrix(a, b, n, m, k);

	std::cout << "/________________________________________________________________________/" << '\n';
	std::cout << "\nIntel(R) Core(TM) i7-8550U CPU @ 1.80GHz VS NVIDIA GeForce MX150\n";
	std::cout << "/________________________________________________________________________/" << '\n';

	auto cpu_time = opencl_gemm(1, 1, 0, 0, n, m, k, a, b, c_opencl_cpu, file_name, kernel_name, 1);
	std::cout << "CPU \t\t\t\t" << cpu_time << '\n';

	auto gpu_time = opencl_gemm(1, 1, 0, 0, n, m, k, a, b, c_opencl_gpu, file_name, kernel_name, 0);
	std::cout << "GPU \t\t\t\t" << gpu_time << '\n';

	if (gpu_time < cpu_time) {
		int qwe = (gpu_time / cpu_time) * 100;
		qwe %= 100;
		double cpu_count = ((double)qwe * 0.01);
		std::cout << cpu_count << " * CPU + " << (1.0 - cpu_count) << " * GPU\n";

		auto cpu_gpu_time = opencl_gemm(1, 1, 0, 0, n, m, k, a, b, c_opencl_cpu_gpu, file_name, kernel_name, cpu_count);
		std::cout << cpu_count << " * CPU + " << 1 - cpu_count << " * GPU \t" << cpu_gpu_time;
		if (cpu_gpu_time < gpu_time) std::cout << "  < GPU" << '\n';
		else std::cout << '\n';

	}
	else {
		int qwe = (cpu_time / gpu_time) * 100;
		qwe %= 100;
		double cpu_count = 1 - ((double)qwe * 0.01);
		std::cout << cpu_count << " * CPU + " << (1.0 - cpu_count) << " * GPU\n";

		auto cpu_gpu_time = opencl_gemm(1, 1, 0, 0, n, m, k, a, b, c_opencl_cpu_gpu, file_name, kernel_name, cpu_count);
		std::cout << cpu_count << " * CPU + " << 1 - cpu_count << " * GPU \t" << cpu_gpu_time;
		if (cpu_gpu_time < cpu_time) std::cout << "  < CPU" << '\n';
		else std::cout << '\n';

	}
	
	//block
	std::cout << "block: \n";

	auto cpu_block_time = opencl_gemm(1, 1, 0, 0, n, m, k, a, b, c_opencl_cpu_block, file_name, kernel_name_block, 1);
	std::cout << "CPU \t\t\t\t" << cpu_block_time << '\n';

	auto gpu_block_time = opencl_gemm(1, 1, 0, 0, n, m, k, a, b, c_opencl_gpu_block, file_name, kernel_name_block, 0);
	std::cout << "GPU \t\t\t\t" << gpu_block_time << '\n';

	if (gpu_block_time < cpu_block_time) {
		int qwe = (gpu_block_time / cpu_block_time) * 100;
		qwe %= 100;
		double cpu_count = ((double)qwe * 0.01);
		std::cout << cpu_count << " * CPU + " << (1.0 - cpu_count) << " * GPU\n";

		auto cpu_gpu_block_time = opencl_gemm(1, 1, 0, 0, n, m, k, a, b, c_opencl_block, file_name, kernel_name_block, cpu_count);
		std::cout << cpu_count << " * CPU + " << 1 - cpu_count << " * GPU \t" << cpu_gpu_block_time;
		if (cpu_gpu_block_time < gpu_block_time) std::cout << "  < GPU" << '\n';
		else std::cout << '\n';
	}
	else {
		int qwe = (cpu_block_time / gpu_block_time) * 100;
		qwe %= 100;
		double cpu_count = 1 - ((double)qwe * 0.01);
		std::cout << cpu_count << " * CPU + " << (1.0 - cpu_count) << " * GPU\n";

		auto cpu_gpu_block_time = opencl_gemm(1, 1, 0, 0, n, m, k, a, b, c_opencl_block, file_name, kernel_name_block, cpu_count);
		std::cout << cpu_count << " * CPU + " << 1 - cpu_count << " * GPU \t" << cpu_gpu_block_time;
		if (cpu_gpu_block_time < cpu_block_time) std::cout << "  < CPU" << '\n';
		else std::cout << '\n';
	}

	

	std::cout << "/________________________________________________________________________/" << '\n';
	std::cout << "\nIntel(R) Core(TM) i7-8550U CPU @ 1.80GHz VS Intel(R) UHD Graphics 620\n";
	std::cout << "/________________________________________________________________________/" << '\n';

	cpu_time = opencl_gemm(1, 1, 1, 0, n, m, k, a, b, c_opencl_cpu, file_name, kernel_name, 1);
	std::cout << "CPU \t\t\t\t" << cpu_time << '\n';

	gpu_time = opencl_gemm(1, 1, 1, 0, n, m, k, a, b, c_opencl_gpu, file_name, kernel_name, 0);
	std::cout << "GPU \t\t\t\t" << gpu_time << '\n';

	if (gpu_time < cpu_time) {
		int qwe = (gpu_time / cpu_time) * 100;
		qwe %= 100;
		double cpu_count = ((double)qwe * 0.01);
		std::cout << cpu_count << " * CPU + " << (1.0 - cpu_count) << " * GPU\n";

		auto cpu_gpu_time = opencl_gemm(1, 1, 1, 0, n, m, k, a, b, c_opencl_cpu_gpu, file_name, kernel_name, cpu_count);
		std::cout << cpu_count << " * CPU + " << 1 - cpu_count << " * GPU \t" << cpu_gpu_time;
		if (cpu_gpu_time < gpu_time) std::cout << "  < GPU" << '\n';
		else std::cout << '\n';

	}
	else {
		int qwe = (cpu_time / gpu_time) * 100;
		qwe %= 100;
		double cpu_count = 1 - ((double)qwe * 0.01);
		std::cout << cpu_count << " * CPU + " << (1.0 - cpu_count) << " * GPU\n";

		auto cpu_gpu_time = opencl_gemm(1, 1, 1, 0, n, m, k, a, b, c_opencl_cpu_gpu, file_name, kernel_name, cpu_count);
		std::cout << cpu_count << " * CPU + " << 1 - cpu_count << " * GPU \t" << cpu_gpu_time;
		if (cpu_gpu_time < cpu_time) std::cout << "  < CPU" << '\n';
		else std::cout << '\n';

	}

	//block
	std::cout << "block: \n";

	cpu_block_time = opencl_gemm(1, 1, 1, 0, n, m, k, a, b, c_opencl_cpu_block, file_name, kernel_name_block, 1);
	std::cout << "CPU \t\t\t\t" << cpu_block_time << '\n';

	gpu_block_time = opencl_gemm(1, 1, 1, 0, n, m, k, a, b, c_opencl_gpu_block, file_name, kernel_name_block, 0);
	std::cout << "GPU \t\t\t\t" << gpu_block_time << '\n';

	if (gpu_block_time < cpu_block_time) {
		int qwe = (gpu_block_time / cpu_block_time) * 100;
		qwe %= 100;
		double cpu_count = ((double)qwe * 0.01);
		std::cout << cpu_count << " * CPU + " << (1.0 - cpu_count) << " * GPU\n";

		auto cpu_gpu_block_time = opencl_gemm(1, 1, 1, 0, n, m, k, a, b, c_opencl_block, file_name, kernel_name_block, cpu_count);
		std::cout << cpu_count << " * CPU + " << 1 - cpu_count << " * GPU \t" << cpu_gpu_block_time;
		if (cpu_gpu_block_time < gpu_block_time) std::cout << "  < GPU" << '\n';
		else std::cout << '\n';
	}
	else {
		int qwe = (cpu_block_time / gpu_block_time) * 100;
		qwe %= 100;
		double cpu_count = 1 - ((double)qwe * 0.01);
		std::cout << cpu_count << " * CPU + " << (1.0 - cpu_count) << " * GPU\n";

		auto cpu_gpu_block_time = opencl_gemm(1, 1, 1, 0, n, m, k, a, b, c_opencl_block, file_name, kernel_name_block, cpu_count);
		std::cout << cpu_count << " * CPU + " << 1 - cpu_count << " * GPU \t" << cpu_gpu_block_time;
		if (cpu_gpu_block_time < cpu_block_time) std::cout << "  < CPU" << '\n';
		else std::cout << '\n';
	}

}

/*-------------------------------------------------------------------------------------------------------------------*/
/*-------------------------------------------------------------------------------------------------------------------*/
/*-------------------------------------------------------------------------------------------------------------------*/
/*--------------------------------------------------------JACOBI-----------------------------------------------------*/
/*-------------------------------------------------------------------------------------------------------------------*/
/*-------------------------------------------------------------------------------------------------------------------*/
/*-------------------------------------------------------------------------------------------------------------------*/

template<typename T>
void runJacobi(std::string type) {
	std::cout << "**********************************" << type << "**********************************\n";
	char* file_name = (char*)"";
	char* kernel_name = (char*)"";
	T* a = new T[n * n];
	T* b = new T[n];
	T* x0 = new T[n];
	T* x1 = new T[n];
	T* norma = new T[n];
	double cpu_time;
	double gpu_time;
	double cpu_gpu_time;
	T eps;
	int counter = 300;

	if (type == "float") {
		eps = 1e-6;
		file_name = (char*)"kernel_jacobi_float.cl";
		kernel_name = (char*)"jacobiFloat";
	}
	else if (type == "double") {
		eps = 1e-12;
		file_name = (char*)"kernel_jacobi_double.cl";
		kernel_name = (char*)"jacobiDouble";
	}
	generateA(a);
	generateB(b);

	if (!check(a)) {
		std::cout << "Не выполняется условие сходимости\n";
		return;
	}

	std::cout << "/________________________________________________________________________/" << '\n';
	std::cout << "\nIntel(R) Core(TM) i7-8550U CPU @ 1.80GHz VS NVIDIA GeForce MX150\n";
	std::cout << "/________________________________________________________________________/" << '\n';

	for (int i = 0; i < n; ++i) {
		x0[i] = 0;
		x1[i] = 0;
	}
	cpu_time = Jacobi(1, 1, 0, 0, n, a, b, x0, x1, norma, file_name, kernel_name, 1, eps, counter);
	std::cout << "CPU \t\t\t\t" << cpu_time << '\n';
	
	for (int i = 0; i < n; ++i) {
		x0[i] = 0;
		x1[i] = 0;
	}
	gpu_time = Jacobi(1, 1, 0, 0, n, a, b, x0, x1, norma, file_name, kernel_name, 0, eps, counter);
	std::cout << "GPU \t\t\t\t" << gpu_time << '\n';
	
	if (gpu_time < cpu_time) {
		int qwe = (gpu_time / cpu_time) * 100;
		qwe %= 100;
		double cpu_count = ((double)qwe * 0.01);

		std::cout << cpu_count << " * CPU + " << (1.0 - cpu_count) << " * GPU\n";
		for (int i = 0; i < n; ++i) {
			x0[i] = 0;
			x1[i] = 0;
		}
		cpu_gpu_time = Jacobi(1, 1, 0, 0, n, a, b, x0, x1, norma, file_name, kernel_name, cpu_count, eps, counter);
		std::cout << cpu_count << " * CPU + " << (1.0 - cpu_count) << " * GPU\t" << cpu_gpu_time;
		if (cpu_gpu_time < gpu_time) std::cout << "  <  GPU    ";
		std::cout << '\n';
	}
	else {
		int qwe = (cpu_time / gpu_time) * 100;
		qwe %= 100;
		double cpu_count = 1 - ((double)qwe * 0.01);

		std::cout << cpu_count << " * CPU + " << (1.0 - cpu_count) << " * GPU\n";
		for (int i = 0; i < n; ++i) {
			x0[i] = 0;
			x1[i] = 0;
		}
		cpu_gpu_time = Jacobi(1, 1, 0, 0, n, a, b, x0, x1, norma, file_name, kernel_name, cpu_count, eps, counter);
		std::cout << cpu_count << " * CPU + " << (1.0 - cpu_count) << " * GPU\t" << cpu_gpu_time;
		if (cpu_gpu_time < cpu_time) std::cout << "  <  CPU    ";
		std::cout << '\n';

	}

	std::cout << "/________________________________________________________________________/" << '\n';
	std::cout << "\nIntel(R) Core(TM) i7-8550U CPU @ 1.80GHz VS Intel(R) UHD Graphics 620\n";
	std::cout << "/________________________________________________________________________/" << '\n';

	for (int i = 0; i < n; ++i) {
		x0[i] = 0;
		x1[i] = 0;
	}
	cpu_time = Jacobi(1, 1, 1, 0, n, a, b, x0, x1, norma, file_name, kernel_name, 1, eps, counter);
	std::cout << "CPU \t\t\t\t" << cpu_time << '\n';
	
	for (int i = 0; i < n; ++i) {
		x0[i] = 0;
		x1[i] = 0;
	}
	gpu_time = Jacobi(1, 1, 1, 0,  n, a, b, x0, x1, norma, file_name, kernel_name, 0, eps, counter);
	std::cout << "GPU \t\t\t\t" << gpu_time << '\n';
	
	if (gpu_time < cpu_time) {
		int qwe = (gpu_time / cpu_time) * 100;
		qwe %= 100;
		double cpu_count = ((double)qwe * 0.01);

		std::cout << cpu_count << " * CPU + " << (1.0 - cpu_count) << " * GPU\n";
		for (int i = 0; i < n; ++i) {
			x0[i] = 0;
			x1[i] = 0;
		}
		cpu_gpu_time = Jacobi(1, 1, 1, 0, n, a, b, x0, x1, norma, file_name, kernel_name, cpu_count, eps, counter);
		std::cout << cpu_count << " * CPU + " << (1.0 - cpu_count) << " * GPU\t" << cpu_gpu_time;
		if (cpu_gpu_time < gpu_time) std::cout << "   <  GPU    ";
		std::cout << '\n';
	}
	else {
		int qwe = (cpu_time / gpu_time) * 100;
		qwe %= 100;
		double cpu_count = 1 - ((double)qwe * 0.01);

		std::cout << cpu_count << " * CPU + " << (1.0 - cpu_count) << " * GPU\n";
		for (int i = 0; i < n; ++i) {
			x0[i] = 0;
			x1[i] = 0;
		}
		cpu_gpu_time = Jacobi(1, 1, 1, 0, n, a, b, x0, x1, norma, file_name, kernel_name, cpu_count, eps, counter);
		std::cout << cpu_count << " * CPU + " << (1.0 - cpu_count) << " * GPU\t" << cpu_gpu_time;
		if (cpu_gpu_time < cpu_time) std::cout << "  <  CPU    ";
		std::cout << '\n';

	}
	
	std::cout << '\n';
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
		std::cout << "\n" << std::string(deviceName) << '\n';
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
		std::cout << "\n"<< std::string(deviceName) << '\n';
	}
}

int main() {
	std::cout << std::fixed << std::setprecision(4);

	cl_uint platformCount = 0;

	clGetPlatformIDs(0, nullptr, &platformCount);

	if (platformCount > 0) {
		cl_platform_id* platforms = new cl_platform_id[platformCount];
		clGetPlatformIDs(platformCount, platforms, nullptr);

		for (cl_uint i = 0; i < platformCount; ++i) {
			char platformName[128];
			clGetPlatformInfo(platforms[i], CL_PLATFORM_NAME, 128, platformName, nullptr);
			std::cout << platformName;
			runGPU(platforms[i]);
			runCPU(platforms[i]);
		}
	}

	/*int number;
	std::cout << "1 - gemm, 2 - jacobi" << '\n';
	std::cin >> number;*/

	//if (number == 1) 
	{
		std::cout << "/************************************************************************/" << '\n';
		std::cout << "/***********************************GEMM*********************************/" << '\n';
		std::cout << "/************************************************************************/" << '\n';

		runGemm<float>("float");
		std::cout << '\n' << '\n';
		//runGemm<double>("double");
	}
	//else 
	{
		std::cout << "/************************************************************************/" << '\n';
		std::cout << "/**********************************JACOBI********************************/" << '\n';
		std::cout << "/************************************************************************/" << '\n';

		runJacobi<float>("float");
		std::cout << '\n' << '\n';
		//runJacobi<double>("double");
	}
	
	return 0;
}

