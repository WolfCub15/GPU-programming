#include <CL/cl.h>
#include <istream>
#include <fstream>
#include <omp.h>
#include "gemm.h"

#define BLOCK_SIZE 32

struct openclJacobi {
	cl_context context;
	cl_command_queue queue;
	cl_program program;
	cl_kernel kernel;
	cl_mem memObjA;
	cl_mem memObjB;
	cl_mem memObjX0;
	cl_mem memObjX1;
	cl_mem memObjNorma;

	openclJacobi(cl_context _context, cl_command_queue _queue, cl_program _program, cl_kernel _kernel,
						cl_mem _memA, cl_mem _memB,	cl_mem _memX0, cl_mem _memX1, cl_mem _memNorma) : 
						context(_context), queue(_queue), program(_program), kernel(_kernel), memObjA(_memA),
						memObjB(_memB), memObjX0(_memX0), memObjX1(_memX1), memObjNorma(_memNorma) {}

	~openclJacobi() {
		clReleaseMemObject(memObjA);
		clReleaseMemObject(memObjB);
		clReleaseMemObject(memObjX0);
		clReleaseMemObject(memObjX1);
		clReleaseMemObject(memObjNorma);
		clReleaseProgram(program);
		clReleaseKernel(kernel);
		clReleaseCommandQueue(queue);
		clReleaseContext(context);
	}
};

template<typename T>
openclJacobi createJacobi(cl_uint platform_index, cl_uint device_index, int n, int m, int step, T* a, T* b, T* x0,
								T* x1, T* norma, char* file_name, char* kernel_name) {
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
	ret = clEnqueueWriteBuffer(command_queue, memObjA, CL_TRUE, 0, sizeof(T) * n * m, a, 0, nullptr, nullptr);
	//check_ret(ret, "EnqueueWriteBuffer A");
	ret = clSetKernelArg(kernel, 0, sizeof(cl_mem), &memObjA);
	//check_ret(ret, "clSetKernelArg 0");

	cl_mem memObjB = clCreateBuffer(context, CL_MEM_READ_ONLY, sizeof(T) * m, nullptr, &ret);
	//check_ret(ret, "clCreateBuffer B");
	ret = clEnqueueWriteBuffer(command_queue, memObjB, CL_TRUE, 0, sizeof(T) * m, b, 0, nullptr, nullptr);
	//check_ret(ret, "EnqueueWriteBuffer B");
	ret = clSetKernelArg(kernel, 1, sizeof(cl_mem), &memObjB);
	//check_ret(ret, "clSetKernelArg 1");

	cl_mem memObjX0 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(T) * m, nullptr, &ret);
	//check_ret(ret, "clCreateBuffer X0");
	ret = clEnqueueWriteBuffer(command_queue, memObjX0, CL_TRUE, 0, sizeof(T) * m, x0, 0, nullptr, nullptr);
	//check_ret(ret, "EnqueueWriteBuffer X0");

	cl_mem memObjX1 = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(T) * m, nullptr, &ret);
	//check_ret(ret, "clCreateBuffer X1");
	ret = clEnqueueWriteBuffer(command_queue, memObjX1, CL_TRUE, 0, sizeof(T) * m, x1, 0, nullptr, nullptr);
	//check_ret(ret, "EnqueueWriteBuffer X1");
	ret = clSetKernelArg(kernel, 3, sizeof(cl_mem), &memObjX1);
	//check_ret(ret, "clSetKernelArg 3");

	cl_mem memObjNorma = clCreateBuffer(context, CL_MEM_WRITE_ONLY, sizeof(T) * m, nullptr, &ret);
	//check_ret(ret, "clCreateBuffer norma");
	ret = clEnqueueWriteBuffer(command_queue, memObjNorma, CL_TRUE, 0, sizeof(T) * m, norma, 0, nullptr, nullptr);
	//check_ret(ret, "EnqueueWriteBuffer norma");
	ret = clSetKernelArg(kernel, 4, sizeof(cl_mem), &memObjNorma);
	//check_ret(ret, "clSetKernelArg 4");

	ret = clSetKernelArg(kernel, 5, sizeof(int), &m);
	//check_ret(ret, "clSetKernelArg 5");

	ret = clSetKernelArg(kernel, 6, sizeof(int), &step);
	//check_ret(ret, "clSetKernelArg 6");
	
	return openclJacobi(context, command_queue, program, kernel, memObjA, memObjB, memObjX0, memObjX1, memObjNorma);
}

template<typename T>
double Jacobi(cl_uint cpu_platform_index, cl_uint cpu_device_index, cl_uint gpu_platform_index, cl_uint gpu_device_index,
						int n, T* a, T* b, T* x0, T* x1, T* norma, char* file_name, char* kernel_name, double count, double epsilon, int counterN) {
	
	int cpu_count = n * count;
	int gpu_count = n - cpu_count;
	std::pair<double, double> result_time;
	size_t cpu_global_work_size[1] = { cpu_count };
	size_t gpu_global_work_size[1] = { gpu_count };
	size_t group_size = BLOCK_SIZE;
	cl_int ret;
	int counter = 0;
	T norma_sum;
	cl_ulong start;
	cl_ulong end;
	cl_event cpu_event;
	cl_event gpu_event;
	T* buffer = new T[n];
	double time = 0;

	if (cpu_count % BLOCK_SIZE != 0) {
		std::cout << "Error: wrong size!" << '\n';
		exit(1);
	}

	if (gpu_count % BLOCK_SIZE != 0){
		std::cout << "Error: wrong size!" << '\n';
		exit(1);
	}


	//CPU
	if (gpu_count == 0){
		openclJacobi cpu = createJacobi(cpu_platform_index, cpu_device_index, cpu_count, n, 0, a, b, x0, x1, norma, file_name, kernel_name);
		do {
			ret = clSetKernelArg(cpu.kernel, 2, sizeof(cl_mem), &cpu.memObjX0);
			//check_ret(ret, "clSetKernelArg 2");
			ret = clEnqueueNDRangeKernel(cpu.queue, cpu.kernel, 1, nullptr, cpu_global_work_size, &group_size, 0, nullptr, &cpu_event);
			clWaitForEvents(1, &cpu_event);

			clGetEventProfilingInfo(cpu_event, CL_PROFILING_COMMAND_START, sizeof(start), &start, nullptr);
			clGetEventProfilingInfo(cpu_event, CL_PROFILING_COMMAND_END, sizeof(end), &end, nullptr);
			time += (end - start) / 1e9;

			ret = clEnqueueReadBuffer(cpu.queue, cpu.memObjNorma, CL_TRUE, 0, sizeof(T) * n, norma, 0, nullptr, nullptr);
			//check_ret(ret, "clEnqueueReadBuffer");

			norma_sum = 0;
			for (int i = 0; i < cpu_count; i++) {
				norma_sum += fabs(norma[i]);
			}

			ret = clEnqueueReadBuffer(cpu.queue, cpu.memObjX1, CL_TRUE, 0, sizeof(T) * n, buffer, 0, nullptr, nullptr);
			memcpy(x1, buffer, sizeof(T) * cpu_count);
			ret = clEnqueueWriteBuffer(cpu.queue, cpu.memObjX0, CL_TRUE, 0, sizeof(T) * n, x1, 0, nullptr, nullptr);
		} while (counter++ < counterN && norma_sum > epsilon);

		ret = clEnqueueReadBuffer(cpu.queue, cpu.memObjX0, CL_TRUE, 0, sizeof(T) * n, buffer, 0, nullptr, nullptr);
		memcpy(x0, buffer, sizeof(T) * cpu_count);
		ret = clEnqueueReadBuffer(cpu.queue, cpu.memObjX1, CL_TRUE, 0, sizeof(T) * n, buffer, 0, nullptr, nullptr);
		memcpy(x1, buffer, sizeof(T) * cpu_count);
		
		clFinish(cpu.queue);

		return time;
	}

	//GPU
	if (cpu_count == 0){
		openclJacobi gpu = createJacobi(gpu_platform_index, gpu_device_index, gpu_count, n, cpu_count, &a[cpu_count * n], b, x0, x1, norma, file_name, kernel_name);

		do {
			ret = clSetKernelArg(gpu.kernel, 2, sizeof(cl_mem), &gpu.memObjX0);
			//check_ret(ret, "clSetKernelArg 2");

			ret = clEnqueueNDRangeKernel(gpu.queue, gpu.kernel, 1, nullptr, gpu_global_work_size, &group_size, 0, nullptr, &gpu_event);
			clWaitForEvents(1, &gpu_event);

			clGetEventProfilingInfo(gpu_event, CL_PROFILING_COMMAND_START, sizeof(start), &start, nullptr);
			clGetEventProfilingInfo(gpu_event, CL_PROFILING_COMMAND_END, sizeof(end), &end, nullptr);
			time += (end - start) / 1e9;

			ret = clEnqueueReadBuffer(gpu.queue, gpu.memObjNorma, CL_TRUE, 0, sizeof(T) * n, norma, 0, nullptr, nullptr);
			//check_ret(ret, "clEnqueueReadBuffer");

			norma_sum = 0;
			for (int i = cpu_count; i < n; ++i) {
				norma_sum += fabs(norma[i]);
			}

			ret = clEnqueueReadBuffer(gpu.queue, gpu.memObjX1, CL_TRUE, 0, sizeof(T) * n, buffer, 0, nullptr, nullptr);
			//check_ret(ret, "clEnqueueReadBuffer");
			memcpy(x1, buffer, sizeof(T) * gpu_count);

			ret = clEnqueueWriteBuffer(gpu.queue, gpu.memObjX0, CL_TRUE, 0, sizeof(T) * n, x1, 0, nullptr, nullptr);
			check_ret(ret, "EnqueueWriteBuffer X0");
		} while (counter++ < counterN && norma_sum > epsilon);

		ret = clEnqueueReadBuffer(gpu.queue, gpu.memObjX0, CL_TRUE, 0, sizeof(T) * n, buffer, 0, nullptr, nullptr);
		memcpy(x0 + cpu_count, buffer + cpu_count, sizeof(T) * gpu_count);

		ret = clEnqueueReadBuffer(gpu.queue, gpu.memObjX1, CL_TRUE, 0, sizeof(T) * n, buffer, 0, nullptr, nullptr);
		memcpy(x1 + cpu_count, buffer + cpu_count, sizeof(T) * gpu_count);

		clFinish(gpu.queue);

		return time;
	}

	//CPU + GPU

	openclJacobi cpu = createJacobi(cpu_platform_index, cpu_device_index, cpu_count, n, 0, a, b, x0, x1, norma, file_name, kernel_name);
	openclJacobi gpu = createJacobi(gpu_platform_index, gpu_device_index, gpu_count, n, cpu_count, &a[cpu_count * n], b, x0, x1, norma, file_name, kernel_name);
	
	do{
		ret = clSetKernelArg(cpu.kernel, 2, sizeof(cl_mem), &cpu.memObjX0);
		//check_ret(ret, "clSetKernelArg 2");
		ret = clSetKernelArg(gpu.kernel, 2, sizeof(cl_mem), &gpu.memObjX0);
		//check_ret(ret, "clSetKernelArg 2");

		ret = clEnqueueNDRangeKernel(cpu.queue, cpu.kernel, 1, nullptr, cpu_global_work_size, &group_size, 0, nullptr, &cpu_event);
		ret = clEnqueueNDRangeKernel(gpu.queue, gpu.kernel, 1, nullptr, gpu_global_work_size, &group_size, 0, nullptr, &gpu_event);
		clWaitForEvents(1, &cpu_event);
		clWaitForEvents(1, &gpu_event);

		clGetEventProfilingInfo(cpu_event, CL_PROFILING_COMMAND_START, sizeof(start), &start, nullptr);
		clGetEventProfilingInfo(cpu_event, CL_PROFILING_COMMAND_END, sizeof(end), &end, nullptr);
		double t1 = (end - start) / 1e9;
		clGetEventProfilingInfo(gpu_event, CL_PROFILING_COMMAND_START, sizeof(start), &start, nullptr);
		clGetEventProfilingInfo(gpu_event, CL_PROFILING_COMMAND_END, sizeof(end), &end, nullptr);
		double t2 = (end - start) / 1e9;

		time += std::max(t1, t2);
		ret = clEnqueueReadBuffer(cpu.queue, cpu.memObjNorma, CL_TRUE, 0, sizeof(T) * n, norma, 0, nullptr, nullptr);
		//check_ret(ret, "clEnqueueReadBuffer");

		norma_sum = 0;
		for(int i = 0; i < cpu_count; ++i){
			norma_sum += fabs(norma[i]);
		}

		ret = clEnqueueReadBuffer(gpu.queue, gpu.memObjNorma, CL_TRUE, 0, sizeof(T) * n, norma, 0, nullptr, nullptr);
		//check_ret(ret, "clEnqueueReadBuffer");

		for (int i = cpu_count; i < n; ++i) {
			norma_sum += fabs(norma[i]);
		}
		
		ret = clEnqueueReadBuffer(cpu.queue, cpu.memObjX1, CL_TRUE, 0, sizeof(T) * n, buffer, 0, nullptr, nullptr);
		memcpy(x1, buffer, sizeof(T) * cpu_count);
		ret = clEnqueueReadBuffer(gpu.queue, gpu.memObjX1, CL_TRUE, 0, sizeof(T) * n, buffer, 0, nullptr, nullptr);
		memcpy(x1 + cpu_count, buffer + cpu_count, sizeof(T) * gpu_count);
		ret = clEnqueueWriteBuffer(cpu.queue, cpu.memObjX0, CL_TRUE, 0, sizeof(T) * n, x1, 0, nullptr, nullptr);
		ret = clEnqueueWriteBuffer(gpu.queue, gpu.memObjX0, CL_TRUE, 0, sizeof(T) * n, x1, 0, nullptr, nullptr);
	} while (counter++ < counterN && norma_sum > epsilon);

	ret = clEnqueueReadBuffer(cpu.queue, cpu.memObjX0, CL_TRUE, 0, sizeof(T) * n, buffer, 0, nullptr, nullptr);
	memcpy(x0, buffer, sizeof(T) * cpu_count);
	ret = clEnqueueReadBuffer(gpu.queue, gpu.memObjX0, CL_TRUE, 0, sizeof(T) * n, buffer, 0, nullptr, nullptr);
	memcpy(x0 + cpu_count, buffer + cpu_count, sizeof(T) * gpu_count);
	ret = clEnqueueReadBuffer(cpu.queue, cpu.memObjX1, CL_TRUE, 0, sizeof(T) * n, buffer, 0, nullptr, nullptr);
	memcpy(x1, buffer, sizeof(T) * cpu_count);
	ret = clEnqueueReadBuffer(gpu.queue, gpu.memObjX1, CL_TRUE, 0, sizeof(T) * n, buffer, 0, nullptr, nullptr);
	memcpy(x1 + cpu_count, buffer + cpu_count, sizeof(T) * gpu_count);
	
	clFinish(cpu.queue);
	clFinish(gpu.queue);

	return time;
}