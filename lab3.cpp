// lab3.cpp: определяет точку входа для консольного приложения.
//

#include "stdafx.h"
#include "mpi.h"
#include <iostream>
#include <fstream>
#include <cmath>

using namespace std;

double pi = 3.1415;

double rhsFunc(double x)                                          //система управления версиями
{
	return pi*pi*sin(pi*x);
}

int _tmain(int argc, char* argv[])
{
	MPI_Init(&argc, &argv);

	double h, tau;
	int n;
	double *u;
	double *uhat;
	double *rhs;

	int globN;
	double Tend;
	int rank, nproc;

	MPI_Comm_size(MPI_COMM_WORLD, &nproc);
	MPI_Comm_rank(MPI_COMM_WORLD, &rank);

	int *locN;
	double *u0;
	double *rhsGlob;
	double *uRes;
	if (rank == 0)
	{
		globN = 1000; //kolvo yacheek 20
		Tend = 10.;

		locN = new int[nproc];
		for (int i = 0; i < nproc; i++)
			locN[i] = (globN-1) / nproc;
		for (int i = 0; i < (globN-1)%nproc; i++)
			locN[i] += 1;

		h = 1. / double(globN);
		tau = h*h*0.25;

		u0 = new double[globN + 1];
		for (int i = 0; i <= globN; i++)
			u0[i] = 0.;

		rhsGlob = new double[globN + 1];
		for (int i = 0; i <= globN; i++)
			rhsGlob[i] = rhsFunc(h*i);

		uRes = new double[globN + 1];
		uRes[0] = uRes[globN] = 0.;
	}
	MPI_Bcast(&tau, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&h, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Bcast(&Tend, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	MPI_Scatter(locN, 1, MPI_INT, &n, 1, MPI_INT, 0, MPI_COMM_WORLD);
	n += 2;
	
	u = new double[n];
	uhat = new double[n];
	if (rank == 0) u[0] = uhat[0] = 0.;
	if (rank == nproc-1) u[n-1] = uhat[n-1] = 0.;

	int *uDispl;
	if (rank == 0)
	{
		uDispl = new int[nproc];
		uDispl[0] = 1;
		for (int i = 1; i < nproc; i++)
			uDispl[i] = uDispl[i - 1] + locN[i - 1];
	}

	MPI_Scatterv(u0, locN, uDispl, MPI_DOUBLE, u + 1, n - 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	MPI_Scatterv(rhsGlob, locN, uDispl, MPI_DOUBLE, rhs + 1, n - 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	int *exchCounts = new int[nproc];
	int *sendDispl = new int[nproc];
	int *recvDispl = new int[nproc];

	for (int i = 0; i < nproc; i++)
		exchCounts[i] = sendDispl[i] = recvDispl[i] = 0;

	if (rank < nproc-1)
	{
		exchCounts[rank + 1] = 1;
		sendDispl[rank + 1] = n - 2;
		recvDispl[rank + 1] = n - 1;
	}

	if (rank > 0)
	{
		exchCounts[rank - 1] = 1;
		sendDispl[rank - 1] = 1;
		recvDispl[rank - 1] = 0;
	}

	double tCurrent = 0.;
	/*if (rank == 0)
	{
		cout << "etalon: ";
		for (int i = 0; i <= globN; i++)
			cout << rhsGlob[i] << " ";
		cout << endl;
	}*/
	do
	{
		MPI_Alltoallv(u, exchCounts, sendDispl, MPI_DOUBLE, u, exchCounts, recvDispl, MPI_DOUBLE, MPI_COMM_WORLD);      //mojno li schityvat' v u?
		//MPI_Alltoallv(rhs, exchCounts, sendDispl, MPI_DOUBLE, rhs, exchCounts, recvDispl, MPI_DOUBLE, MPI_COMM_WORLD);      //mojno li schityvat' v u?

		/*cout << rank << ": ";
		for (int i = 0; i < n; i++)
			cout << rhs[i] << " ";
		cout << endl;*/

		for (int i = 1; i < n - 1; i++)
			uhat[i] = u[i] + tau*(rhs[i] + (u[i - 1] - 2 * u[i] + u[i + 1]) / (h*h));

		//sohranenie v file MPI_Gatherv(uhat + 1, n - 2, MPI_DOUBLE, uRes, locN, uDispl, MPI_DOUBLE, 0, MPI_COMM_WORLD);

		swap(uhat, u);
		tCurrent += tau;
	} while (tCurrent < Tend);

	MPI_Gatherv(u + 1, n - 2, MPI_DOUBLE, uRes, locN, uDispl, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	if (rank == 0)
	{
		ofstream outFile;
		outFile.open("res.dat"); 

		for (int i = 0; i <= globN; i++)
			outFile << i*h << " " << uRes[i] << endl;
		outFile.close();
	}

	MPI_Finalize();
	//iz fila dannye zadachi - po processam - local massivy - smeschenija i kolva - rasparallelivanie
	return 0;
}

