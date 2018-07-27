#include <iostream>
#include <fstream>
#include <cmath>
#include <cuda.h>
const int n = 1025;	
const double h = 1.0 / (double)(n);
const double K2 = 100.0;
const double cft1 = 1.0 / (4.0 + h * h * K2);
const double cft3 = cft1 * h * h;
const double PI = 3.1415926535897932385;
const int MaxIter = 30000;

using namespace std;

__host__  double fRight(double x, double y) 
{
    return 2.0 * sin(PI * y) + K2 * (1.0 - x) * x * sin(PI * y) + PI * PI * (1.0 - x) * x * sin(PI * y);
}

__global__ void kernelJacobi(double* ym, double* um, double* fm)
{
	/*int alpha, beta; // положение блока
	int i, j;		// положение треда в блоке
	alpha = blockIdx.x;
	beta  = blockIdx.y;
	i = threadIdx.x;
	j = threadIdx.y;	
	
	int bnx, bny; // размеры блоков
	bnx = blockDim.x;
	bny = blockDim.y;
	
	int row,col;// положение элемента в матрице
	col = alpha * bnx + i;
	row = beta * bny + j;
	
	int id  = row * (n + 1) + col;
	
	//int k;
	//um[id] = 0.0;
	
	//if ((id <= n * (n + 1) - 1)&&(id >= n + 1)&&((id % (n + 1)) != 0)&&((id % (n + 1)) != n))
	um[id] = (cft1 * (ym[id - n - 1] + ym[id + n + 1] + ym[id - 1] + ym[id + 1]) + fm[id]) * cft3;*/
    int i = blockDim.y * blockIdx.y + threadIdx.y;
	int j = blockDim.x * blockIdx.x + threadIdx.x;
	um[(i + 1) * (n + 1) + (j + 1)] = 0.0;
	um[(i + 1) * (n + 1) + (j + 1)] = cft1 * (ym[(i + 2) * (n + 1) + (j + 1)] + ym[(i + 1) * (n + 1) + j + 2] + ym[i * (n + 1) + (j + 1)] + ym[(i + 1) * (n + 1) + j]) + cft3 * fm[(i + 1) * (n + 1) + (j + 1)];
	
	return;
}

__global__ void kernelSwap(double* ym, double* um)
{
	int alpha, beta; // положение блока
	int i, j;		// положение треда в блоке
	alpha = blockIdx.x;
	beta  = blockIdx.y;
	i = threadIdx.x;
	j = threadIdx.y;	
	
	int bnx, bny; // размеры блоков
	bnx = blockDim.x;
	bny = blockDim.y;
	
	int row,col;// положение элемента в матрице
	col = alpha * bnx + i;
	row = beta * bny + j;
	
	int id  = row * (n + 1) + col;
	
	ym[id] = um[id];
	
	return;
}

double exsol(double x, double y) 
{
    return x * (1.0 - x) * sin(PI * y);
}


int main(int argc, char* argv[])
{
	/*int n = 128;
	
	double h = 1.0 / n;
	
	const double cft1 = 1 / (h * h);
    const double cft2 = 4.0 * cft1 + K2;
    const double cft3 = 1 / cft2;
	const double PI = 3.141592653589793238462643;
	const double PI2 = PI*PI;
	const double K2 = 100.0;
	const int MaxIter = 10000;*/
	
	cout << "cft1 = " << cft1 << endl;
	//cout << "cft2 = " << cft2 << endl;
	cout << "cft3 = " << cft3 << endl;
	
	double * y;
	double * u;
	double * f;

	y = new double [(n + 1) * (n + 1)];
	u = new double [(n + 1) * (n + 1)];
	f = new double [(n + 1) * (n + 1)];

	int nbytes = (n + 1) * (n + 1) * sizeof(double);
	
	int i, j;

	//инициализация
	for (i = 1; i < n; i++)
    {
		y[i * (n + 1)] = 0.0;
		y[i * (n + 1) + n] = 0.0;
		for (j = 1; j < n; j++)
			y[i * (n + 1) + j] = 0.0; //0.5
    }
    for (i = 0; i <= n; i++)
    {
        y[i] = 0.0;
        y[n * (n + 1) + i] = 0.0;
		for (j = 0; j <= n; j++)
			u[i * (n + 1) + j] = 0.0;
    }
	
	for (i = 0; i <= n; i++)
	for (j = 0; j <= n; j++) 
		f[i * (n + 1) + j] = fRight(i * h, j * h);
	//
	
	cudaError_t SD;
	
	SD = cudaSetDevice(0);
	if (SD != cudaSuccess)
	{
		cout << "CUDA set device error" << endl;
		return 1;
	}
	
	double * uDev = NULL;
	double * yDev = NULL;
	double * fDev = NULL;
	
	cudaMalloc ((void **)&uDev, nbytes);
	cudaMalloc ((void **)&yDev, nbytes);
	cudaMalloc ((void **)&fDev, nbytes);
	
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
		
	cudaEventRecord(start,0);	
	cudaEventSynchronize(start);
	
	const int blockDimx = 1; //n; //32; //8; //16; //64; //128;
	const int blockDimy = 1024;
	dim3 threads(blockDimx, blockDimy, 1);
	dim3 blocks((n - 1) / blockDimx, (n - 1)  / blockDimy, 1);
	cout << blocks.x << "\n";
	
	cudaMemcpy(fDev, f, nbytes, cudaMemcpyHostToDevice);
	cudaMemcpy(uDev, u, nbytes, cudaMemcpyHostToDevice);
	cudaMemcpy(yDev, y, nbytes, cudaMemcpyHostToDevice);
	
	//int iter = 1;
	for (int k = 0; k < MaxIter/2; k++)
	{
		kernelJacobi<<<blocks, threads>>>(yDev, uDev, fDev);
		kernelJacobi<<<blocks, threads>>>(uDev, yDev, fDev);
	}
	/*do
	{
		kernelJacobi<<<blocks, threads>>>(yDev, uDev, fDev);
		kernelJacobi<<<blocks, threads>>>(uDev, yDev, fDev);
		//cudaThreadSynchronize();
		
		//kernelSwap<<<blocks, threads>>>(yDev, uDev);
		//cudaThreadSynchronize();
		
		iter++;
	} while (iter <= MaxIter);*/
	//cudaThreadSynchronize();
	
	//test
	/*kernelJacobi<<<blocks, threads>>>(yDev, uDev, fDev);
	cudaMemcpy(u, uDev, nbytes, cudaMemcpyDeviceToHost);*/
	//
	
	cudaMemcpy(y, yDev, nbytes, cudaMemcpyDeviceToHost);
	
	cudaEventRecord(stop,0);	
	cudaEventSynchronize(stop);
	
	float dt;
	cudaEventElapsedTime(&dt,start,stop);
	
	//абсолютная погрешность
	double max = 0.0;
    for (int i = 0; i <= n; i++) 
	{
         //double x = i*h;
        for (int j = 0; j <= n; j++) 
		{
             //double y = j*h;
            double val = fabs(y[i * (n + 1) + j] - exsol(i * h, j * h));
            if (max < val) 
			{
                max = val;
            }
        }
    }
	cout << "Mistake: " << max << endl;
	//
	
	//запись в файл
	/*ofstream outFile;
	outFile.open("res.dat");
	for (int i = 0; i <= n; i++)
	for (int j = 0; j <= n; j++)
		outFile << i * h << " " << j * h << " " << y[i * (n + 1) + j] << endl;
	outFile.close();*/
	//
	
	cout << "Time = " << dt << " ms"<< endl;
	
	delete[] u;
	delete[] y;
	delete[] f;
	
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
	cudaFree(uDev);
	cudaFree(yDev);
	cudaFree(fDev);
	
	return 0;
}

