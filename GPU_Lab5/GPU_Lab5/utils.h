int n = 32 * 100, m = 32 * 100, k = 32 * 100;


template<typename T>
void GenerateMatrix(T* a, T* b, int n, int m, int k) {
	std::mt19937 gen(123);
	// std::mt19937 gen(std::chrono::high_resolution_clock::now().time_since_epoch().count());
	std::uniform_real_distribution<> dis(0, 1);
	for (int i = 0; i < n; i++) for (int j = 0; j < m; j++) {
		a[i * m + j] = dis(gen);
	}
	for (int i = 0; i < m; i++) for (int j = 0; j < k; j++) {
		b[i * k + j] = dis(gen);
	}
}

template<typename T>
double SequentialGemm(int n, int m, int k, const T const* a, const T const* b, T* c) {
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
bool CheckGemm(int n, int k, const T const* _1, const T const* _2) {
	T ma = 0;
	for (int i = 0; i < n * k; i++) {
		ma += std::abs(_1[i] - _2[i]);
	}
	ma /= n * k;

	if (ma > std::numeric_limits<T>::epsilon() * 1000) {
		std::cout << "threshold = \t\t" << ma << '\n';
		exit(1);
	}
	return 1;
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



std::mt19937 gen(std::chrono::high_resolution_clock::now().time_since_epoch().count());

template<typename T>
void generateA(T* a) {
	for (int i = 0; i < n; i++) {
		for (int j = 0; j < n; j++) {
			T tmp = gen() % 50000;
			tmp /= n;
			a[i * n + j] = (i == j ? tmp + 100000 : tmp);
		}
	}
}

template<typename T>
void generateB(T* b) {
	for (int i = 0; i < n; i++) {
		T tmp = gen();
		tmp /= n;
		b[i] = tmp;
	}
}

template<typename T>
bool check(T* a) {
	for (int i = 0; i < n; i++) {
		T sum = 0;
		for (int j = 0; j < n; j++) sum += a[i * n + j];
		sum -= a[i * n + i];
		if (sum > a[i * n + i]) return false;
	}
	return true;
}

template<typename T>
bool CheckSolution(T* a, T* b, T* x) {
	T* res = new T[n];
	for (int i = 0; i < n; ++i) {
		res[i] = 0;
		for (int j = 0; j < n; ++j) {
			res[i] += a[i * n + j] * x[j];
		}
	}
	T num = 0;
	for (int i = 0; i < n; ++i) {
		num += std::abs((res[i] - b[i]) / res[i]);
	}
	return num < (sizeof(a[0]) == 4 ? T(1) : T(1e-10));
}
