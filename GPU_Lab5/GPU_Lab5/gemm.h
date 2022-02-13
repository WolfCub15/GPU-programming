#include <CL/cl.h>
#include <istream>
#include <fstream>
#include <string>
#include "initialize.h"

#define BLOCK_SIZE 16

struct openclGemm {
	cl_context context;
	cl_command_queue queue;
	cl_program program;
	cl_kernel kernel;
	cl_mem memObjA;
	cl_mem memObjB;
	cl_mem memObjC;

	openclGemm(cl_context _context, cl_command_queue _queue, cl_program _program, cl_kernel _kernel, cl_mem _memA, cl_mem _memB, cl_mem _memC) : 
					context(_context), queue(_queue), program(_program), kernel(_kernel), memObjA(_memA), memObjB(_memB), memObjC(_memC) {}
	~openclGemm(){
		clReleaseMemObject(memObjA);
		clReleaseMemObject(memObjB);
		clReleaseMemObject(memObjC);
		clReleaseProgram(program);
		clReleaseKernel(kernel);
		clReleaseCommandQueue(queue);
		clReleaseContext(context);
	}
};

template<typename T>
openclGemm createGemm(cl_uint platform_index, cl_uint device_index, int n, int m, int k, T* a, T* b, char* file_name, char* kernel_name){
	cl_device_id device;
	init(platform_index, device_index, device);
	std::string kernel_code = ReadKernel(file_name);
	size_t kernel_size = kernel_code.size();

	cl_int ret;

	cl_context context = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &ret);
	//check_ret(ret, "clCreateContext");
	cl_queue_properties props[] = { CL_QUEUE_PROPERTIES, CL_QUEUE_PROFILING_ENABLE, 0 };
	cl_command_queue command_queue = clCreateCommandQueueWithProperties(context, device, props, &ret);
	//check_ret(ret, "clCreateCommandQueueWithProperties");
	cl_program program = clCreateProgramWithSource(context, 1, (const char**)&kernel_code, &kernel_size, &ret);
	//check_ret(ret, "clCreateProgramWithSource");
	ret = clBuildProgram(program, 1, &device, nullptr, nullptr, nullptr);
	//check_ret(ret, "clBuildProgram");

	cl_kernel kernel = clCreateKernel(program, kernel_name, &ret);
	//check_ret(ret, "clCreateKernel");

	cl_mem memObjA = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(T) * n * m, nullptr, &ret);
	//check_ret(ret, "clCreateBuffer A");
	cl_mem memObjB = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(T) * m * k, nullptr, &ret);
	//check_ret(ret, "clCreateBuffer B");
	cl_mem memObjC = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(T) * n * k, nullptr, &ret);
	//check_ret(ret, "clCreateBuffer C");

	ret = clEnqueueWriteBuffer(command_queue, memObjA, CL_TRUE, 0, sizeof(T) * n * m, a, 0, nullptr, nullptr);
	//check_ret(ret, "EnqueueWriteBuffer A");
	ret = clEnqueueWriteBuffer(command_queue, memObjB, CL_TRUE, 0, sizeof(T) * m * k, b, 0, nullptr, nullptr);
	//check_ret(ret, "EnqueueWriteBuffer B");

	ret = clSetKernelArg(kernel, 0, sizeof(int), &n);
	//zcheck_ret(ret, "clSetKernelArg 0");
	ret = clSetKernelArg(kernel, 1, sizeof(int), &m);
	//check_ret(ret, "clSetKernelArg 1");
	ret = clSetKernelArg(kernel, 2, sizeof(int), &k);
	//check_ret(ret, "clSetKernelArg 2");
	ret = clSetKernelArg(kernel, 3, sizeof(cl_mem), &memObjA);
	//check_ret(ret, "clSetKernelArg 3");
	ret = clSetKernelArg(kernel, 4, sizeof(cl_mem), &memObjB);
	//check_ret(ret, "clSetKernelArg 4");
	ret = clSetKernelArg(kernel, 5, sizeof(cl_mem), &memObjC);
	//check_ret(ret, "clSetKernelArg 5");

	return openclGemm(context, command_queue, program, kernel, memObjA, memObjB, memObjC);
}

template<typename T>
double opencl_gemm(cl_uint cpu_platform_index, cl_uint cpu_device_index, cl_uint gpu_platform_index, cl_uint gpu_device_index, int n,
					int m, int k, T* a, T* b, T* c, char* file_name, char* kernel_name, double count) {
	int cpu_count = n * count;
	int gpu_count = n - cpu_count;

	/*if (cpu_count % BLOCK_SIZE != 0) {
		std::cout << "Error: wrong size!" << '\n';
		exit(1);
	}

	if (gpu_count % BLOCK_SIZE != 0) {
		std::cout << "Error: wrong size!" << '\n';
		exit(1);
	}*/

	size_t cpu_global_work_size[2] = { k, cpu_count };
	size_t gpu_global_work_size[2] = { k, gpu_count };
	size_t group_size[2] = { BLOCK_SIZE, BLOCK_SIZE };
	cl_int ret;

	//CPU
	if (gpu_count == 0){
		openclGemm cpu = createGemm(cpu_platform_index, cpu_device_index, cpu_count, m, k, a, b, file_name, kernel_name);
		cl_event cpu_event;
		ret = clEnqueueNDRangeKernel(cpu.queue, cpu.kernel, 2, nullptr, cpu_global_work_size, group_size, 0, nullptr, &cpu_event);

		clWaitForEvents(1, &cpu_event);
		ret = clEnqueueReadBuffer(cpu.queue, cpu.memObjC, CL_TRUE, 0, sizeof(T) * cpu_count * k, c, 0, nullptr, nullptr);

		cl_ulong start;
		cl_ulong end;

		clGetEventProfilingInfo(cpu_event, CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL);
		clGetEventProfilingInfo(cpu_event, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL);

		double cpu_time = (end - start) / 1e9;
		return cpu_time;
	}

	//GPU
	if (cpu_count == 0){
		openclGemm gpu = createGemm(gpu_platform_index, gpu_device_index, gpu_count, m, k, a, b, file_name, kernel_name);
		cl_event gpu_event;
		ret = clEnqueueNDRangeKernel(gpu.queue, gpu.kernel, 2, nullptr, gpu_global_work_size, group_size, 0, nullptr, &gpu_event);
		clWaitForEvents(1, &gpu_event);

		ret = clEnqueueReadBuffer(gpu.queue, gpu.memObjC, CL_TRUE, 0, sizeof(T) * gpu_count * k, c, 0, nullptr, nullptr);
		cl_ulong start;
		cl_ulong end;

		clGetEventProfilingInfo(gpu_event, CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL);
		clGetEventProfilingInfo(gpu_event, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL);

		double gpu_time = (end - start) / 1e9;
		return gpu_time;
	}

	//CPU + GPU
	openclGemm cpu = createGemm(cpu_platform_index, cpu_device_index, cpu_count, m, k, a, b, file_name, kernel_name);
	openclGemm gpu = createGemm(gpu_platform_index, gpu_device_index, gpu_count, m, k, &a[cpu_count * m], b, file_name, kernel_name);


	cl_event cpu_event;
	cl_event gpu_event;
	ret = clEnqueueNDRangeKernel(cpu.queue, cpu.kernel, 2, nullptr, cpu_global_work_size, group_size, 0, nullptr, &cpu_event);

	ret = clEnqueueNDRangeKernel(gpu.queue, gpu.kernel, 2, nullptr, gpu_global_work_size, group_size, 0, nullptr, &gpu_event);

	clWaitForEvents(1, &cpu_event);
	clWaitForEvents(1, &gpu_event);

	ret = clEnqueueReadBuffer(cpu.queue, cpu.memObjC, CL_TRUE, 0, sizeof(T) * cpu_count * k, c, 0, nullptr, nullptr);
	//check_ret(ret, "clEnqueueReadBuffer");
	ret = clEnqueueReadBuffer(gpu.queue, gpu.memObjC, CL_TRUE, 0, sizeof(T) * gpu_count * k, &c[cpu_count * k], 0, nullptr, nullptr);
	//check_ret(ret, "clEnqueueReadBuffer");

	cl_ulong start;
	cl_ulong end;

	clGetEventProfilingInfo(cpu_event, CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL);
	clGetEventProfilingInfo(cpu_event, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL);
	double cpu_time = (end - start) / 1e9;

	clGetEventProfilingInfo(gpu_event, CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL);
	clGetEventProfilingInfo(gpu_event, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL);
	double gpu_time = (end - start) / 1e9;

	double cpu_gpu_time = std::max(cpu_time, gpu_time);
	return cpu_gpu_time;
}
