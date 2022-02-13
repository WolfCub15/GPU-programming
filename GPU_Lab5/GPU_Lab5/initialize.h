#include <CL/cl.h>
#include <istream>
#include <fstream>
#include <string>
#include <iostream>

void get_device_name(cl_device_id& device) {
	char ans[128];
	clGetDeviceInfo(device, CL_DEVICE_NAME, 128, ans, nullptr);
	//std::cout << ans << '\n';
}

void init(cl_uint platform_index, cl_uint device_index, cl_device_id& device) {
	cl_platform_id* platforms = new cl_platform_id[3];
	clGetPlatformIDs(3, platforms, nullptr);
	char platformName[128];
	//std::cout << platform_index << ' ' << device_index << '\n';
	clGetPlatformInfo(platforms[platform_index], CL_PLATFORM_NAME, 128, platformName, nullptr);
	//clGetPlatformInfo(platforms[2], CL_PLATFORM_NAME, 128, platformName, nullptr);

	//std::cout << platformName << '\n';

	cl_uint deviceCount = 0;

	if (platform_index == 1 && device_index == 1) {
		clGetDeviceIDs(platforms[platform_index], CL_DEVICE_TYPE_CPU, 0, nullptr, &deviceCount);
	}
	else {
		clGetDeviceIDs(platforms[platform_index], CL_DEVICE_TYPE_GPU, 0, nullptr, &deviceCount);

	}

	cl_device_id* devices = new cl_device_id[2];

	if (platform_index == 1 && device_index == 1) {
		clGetDeviceIDs(platforms[platform_index], CL_DEVICE_TYPE_CPU, 1, devices, nullptr);
		device = devices[0];
	}
	else {
		clGetDeviceIDs(platforms[platform_index], CL_DEVICE_TYPE_GPU, 1, devices, nullptr);
		device = devices[0];
	}

	get_device_name(device);
}

std::string ReadKernel(char* filename) {
	std::ifstream is(filename);
	std::string ans, tmp;
	while (getline(is, tmp)) {
		ans += tmp;
		ans += '\n';
	}
	return ans;
}

void check_ret(cl_int ret, const char* message) {
	if (ret != CL_SUCCESS) {
		std::cout << message << '\n';
		std::cout << "RETCODE = " << ret << '\n';
		exit(1);
	}
}