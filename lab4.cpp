#include "mpi.h"
#include <iostream>
#include <cmath>
#include <fstream>

using namespace std;

typedef double tdata;

const tdata PI = 3.141592653589793238462643;
const tdata PI2 = PI*PI;

const tdata K2 = 100.0;//K квадрат
//const tdata L = 1.0;
//const int N2 = (n+1)*(n+1);
//const tdata h = L/n;
//
//const tdata cft1 = 1/(h*h);
//const tdata cft2 = 4.0*cft1 + K2;
//const tdata cft3 = cft2/cft2;
//tdata eps = 1e-5;
//
//const int errtype = 2;
const bool vyvod = false;
//void Jacoby(int n, tdata **U, tdata **F);
void zeroMtx(int n, tdata **A);
//void CountNev(int n, tdata **U, tdata **f);
//tdata MNormDlt(int n, tdata **A, tdata **B);
tdata fRight(tdata x, tdata y);
void init(int n, tdata value, tdata *A);

void PrintVct(int rank, int size, double *Vector, string name);
void PrintLongMtx(int rank, int LineSize, int NofLines, double *M, string name);
void PrintMtx(int LineSize, int NofLines, double *M, string name);

int main(int argc, char* argv[])
{
    MPI_Init(&argc, &argv);
    
    int rank, nproc;
    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int MaxIter = 10000 / nproc;

    int n = 1000; 
	int size = (n + 1) * (n + 1);
    tdata h = 1.0 / n;
    const tdata cft1 = 1 / (h * h);
    const tdata cft2 = 4.0 * cft1 + K2;
    const tdata cft3 = 1 / cft2;
    
	double t0 = MPI_Wtime();

    int nloc; 
	int nldispl; 
	int* locN; 
	int* locNsent;
	int* Displ;
    tdata *F; 
	tdata *U0;
	tdata *U1;
    if (rank == 0)
    {
        locN = new int[nproc];
        locNsent = new int[nproc];
        Displ = new int[nproc];
        for (int i = 0; i < nproc; i++) 
			locN[i] = (n - 1) / nproc;
        for (int s = 0; s < (n - 1) % nproc; s++)
            locN[s] += 1; //(1 - s % 2) * (s / 2) + (s % 2) * (n - s / 2)
        /* чтобы сначала раздавали больше первому и последнему */
        for (int i = 0; i < nproc; i++) 
			locNsent[i] = (n + 1) * (locN[i] + 2);
        /* сколько раздаем локально */
        
        /* с какого номера раздаем */
        Displ[0] = 0;
        for (int i = 1; i < nproc; i++) 
			Displ[i] = Displ[i - 1] + locN[i - 1] * (n + 1);
        
        F = new tdata[size];
        tdata *xGrid = new tdata[n + 1];
        for (int i = 0; i < n+1; i++) 
			xGrid[i] = i * h;
        for (int i = 0; i < n + 1; i++)
        for (int j = 0; j < n + 1; j++) 
			F[i * (n + 1) + j] = fRight(xGrid[i], xGrid[j]); //1.0 * (i * (n + 1) + j);

        U0 = new tdata[size];
		U1 = new tdata[size];

        init(n, 0., U0);
	 init(n, 0., U1);
        
        //PrintLongMtx(rank, n + 1, n + 1, F, "F");
        
        if (vyvod)
        {
            cout << "locN" << endl;
            for (int i = 0; i < nproc; i++) 
				cout << locN[i] << endl;
            PrintLongMtx(rank, n + 1, n + 1, U0, "U0");
        }
    }
    
    MPI_Scatter(locN, 1, MPI_INT, &nloc, 1, MPI_INT, 0, MPI_COMM_WORLD);
    nldispl = nloc + 2;
    
    tdata *flocal = new tdata[(nloc + 2) * (n + 1)];
    MPI_Scatterv(F, locNsent, Displ, MPI_DOUBLE, flocal, (nloc + 2) * (n + 1), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
    tdata *ulocal = new tdata[(nloc + 2) * (n + 1)];
	MPI_Scatterv(U0, locNsent, Displ, MPI_DOUBLE, ulocal, (nloc + 2) * (n + 1), MPI_DOUBLE, 0, MPI_COMM_WORLD);
    
   /* if (rank == 1)
    {
        PrintLongMtx(rank, n + 1, nloc + 2, flocal, "flocal");
        PrintLongMtx(rank, n + 1, nloc + 2, ulocal, "uLocal");
    }*/
    
    tdata *ylocal = new tdata[(nloc + 2) * (n + 1)]; //старое приближение
   
	int * exchCnts = new int[nproc];
	int * sendDisp = new int[nproc];
	int * recvDisp = new int[nproc];
	for (int i = 0; i < nproc; i++)
	    exchCnts[i] = sendDisp[i] = recvDisp[i] = 0;
	
	/*if (rank < nproc - 1)
	{
		exchCnts[rank] = n + 1;
		sendDisp[rank] = (nloc + 1) * (n + 1);
		recvDisp[rank] = ;
	}*/

	if (rank < nproc-1) 
	{
	    exchCnts[rank + 1] = n + 1;
		sendDisp[rank + 1] = nloc * (n + 1);
		recvDisp[rank + 1] = (nloc + 1) * (n + 1);
	}
	
	if (rank > 0) 
	{
	    exchCnts[rank - 1] = n + 1;
	    sendDisp[rank - 1] = n + 1;
	    recvDisp[rank - 1] = 0;
	}
	

    //нужно загнать это в цикл do while (k<=MaxIter) и ввести пересылки alltoall
    //или парные isend, как у Вики
    //for (int i = 1; i < n; i++)
    //    for (int j = 1; j < n; j++)
    //        Y[i][j] = U[i][j];

    //for (int i = 1; i < n; i++)
    //    for (int j = 1; j < n; j++)
    //        U[i][j] = (cft1*(Y[i-1][j] + Y[i+1][j] + Y[i][j-1] + Y[i][j+1]) + F[i][j])*cft3;

	for (int i = 1; i < nloc + 1; i++)
	for (int j = 1; j < n; j++)
		ylocal[i * (n + 1) + j] = ulocal[i * (n + 1) + j];
    
	int iter = 0;

	double tc0 = 0.;
	double tc1 = 0.;
	double tc = 0.;

    do
    { 
		iter++;
		MPI_Alltoallv(ulocal, exchCnts, sendDisp, MPI_DOUBLE, ulocal, exchCnts, recvDisp, MPI_DOUBLE, MPI_COMM_WORLD);
        //MPI_Alltoallv(flocal, exchCnts, sendDisp, MPI_DOUBLE, flocal, exchCnts, recvDisp, MPI_DOUBLE, MPI_COMM_WORLD);
        //cout << rank << " : ";
        //for (int i = 0; i < n; i++)
        //    cout << rhs[i] << " ";
        //cout << endl;

		tc0 = MPI_Wtime();
    
		for (int i = 1; i < nloc + 1; i++)
		for (int j = 1; j < n; j++)
			ulocal[i * (n + 1) + j] = (cft1 * (ylocal[(i - 1) * (n + 1) + j] + ylocal[(i + 1) * (n + 1) + j] + ylocal[i * (n + 1) + j - 1] + ylocal[i * (n + 1) + j + 1]) + flocal[i * (n + 1) + j]) * cft3;
    
		tc1 = MPI_Wtime();

		tc += (tc1 - tc0);
		tc0 = tc1 = 0.;

        // сохранение промежуточных данных ...
    
        swap(ylocal, ulocal);
    } while (iter <= MaxIter);
    
	if (vyvod) 
		cout << "Процесс " << rank << ": " << iter << endl;

	int * recvCnts;
	if (rank == 0)
	{
		recvCnts = new int[nproc];
		for (int i = 0; i < nproc; i++)
			recvCnts[i] = locN[i] * (n + 1);
		Displ[0] = n + 1;
		for (int i = 1; i < nproc; i++)
			Displ[i] = Displ[i - 1] + locN[i - 1] * (n + 1);
	}

	MPI_Gatherv(ulocal + n + 1, nloc * (n + 1), MPI_DOUBLE, U1, recvCnts, Displ, MPI_DOUBLE, 0, MPI_COMM_WORLD);
	
	if (vyvod)
		cout << "Процесс " << rank << ": Still goin' strong" << endl;

	if (vyvod)
		PrintLongMtx(rank, n + 1, nloc + 2, ulocal, "uLocal");

	/*if (rank == 0)
	{
	    ofstream outFile;
	    outFile.open("res.dat");
	    for (int i = 0; i <= n; i++)
		for (int j = 0; j <= n; j++)
			outFile << i * h << " " << j * h << " " << U1[i * (n + 1) + j] << endl;
	    outFile.close();
	}*/

	if ((rank == 0) && (vyvod))
		PrintMtx(n + 1, n + 1, U1, "U1");


	//cout << "nloc" << endl;
	/*if (rank == nproc - 1)
	{
		cout << "nloc" << endl;
		for (int i = 0; i <= n; i++) 
			cout << ulocal[(nloc)*(n + 1) + i] << endl;
		cout << endl;
		cout << "nloc + 1" << endl;
		for (int i = 0; i <= n; i++)
			cout << ulocal[(nloc + 1)*(n + 1) + i] << endl;
		cout << endl;
	}*/
	/*if (rank == 0)
	{
		cout << "U1" << endl;
		for (int i = 0; i <= n; i++) 
			cout << U1[(n - 1)*(n + 1) + i] << endl;
	}*/
	//if (rank == 0) for (int i = 0; i <= n; i++) cout << U1[(n - 1)*(n + 1) + i] << endl;

	//
	/*for (int i = 0; i < (nloc + 2); i++)
	for (int j = 0; j < n + 1; j++)
		ulocal[i*(n + 1) + j] = (rank + 1)*1.0;

	if (rank == 0)
	for (int i = 0; i < (n + 1); i++)
	for (int j = 0; j < n + 1; j++)
		U1[i*(n + 1) + j] = 0.0;

	MPI_Gatherv(ulocal + n + 1, nloc * (n + 1), MPI_DOUBLE, U1, recvCnts, Displ, MPI_DOUBLE, 0, MPI_COMM_WORLD);

	if ((rank == 0) && (vyvod))
	{
		cout << "Proverka" << endl;
		PrintMtx(n + 1, n + 1, U1, "U1");
	}*/
	//

	double t1 = MPI_Wtime();
	double t = t1 - t0;
	cout << "ts = " << t - tc << endl;
	cout << "t = " << t << endl;


    MPI_Finalize();
    
    return 0;
}

void init(int N, tdata value, tdata *A)
{
    //всего узлов N+1, те i от 0 до N включительно
    for (int i = 1; i < N; i++)
    {
		A[i * (N + 1)] = 0.0;
		A[i * (N + 1) + N] = 0.0;
		for (int j = 1; j < N; j++)
			A[i * (N + 1) + j] = value;
    }
    for (int i = 0; i <= N; i++)
    {
        A[i] = 0.0;
        A[N * (N + 1) + i] = 0.0;
    }
}

void PrintLongMtx(int rank, int LineSize, int NofLines, double *M, string name)
{
    cout << endl << "Процесс " << rank << ": Матрица " << name << endl;

    for (int start = 0; start < NofLines; start++)
    {
        for (int i = LineSize * start; i < LineSize * (start + 1); i++) 
			cout << M[i] << " \t";
        cout << endl;
    }
}

void PrintMtx(int LineSize, int NofLines, double *M, string name)
{
	cout << endl << "Матрица " << name << endl;

	for (int start = 0; start < NofLines; start++)
	{
		for (int i = LineSize * start; i < LineSize * (start + 1); i++)
			cout << M[i] << " \t";
		cout << endl;
	}
}

void zeroMtx(int n, tdata **A)
{
    for (int i = 0; i < n; i++)
    {
        for (int j = 0; j < n; j++)
            A[i][j] = 0.0;
    }
}

tdata fRight(tdata x, tdata y)
{
    return sin(PI*y)*((x-x*x)*(K2 + PI*PI) + 2.0);
}

void PrintVct(int rank, int size, double *Vector, string name)
{
    cout << "Процесс " << rank << ": Вектор " << name << "   size = " << size << endl;
    for (int i = 0; i<size; i++) 
		cout << Vector[i] << endl;
    cout << endl;
}

//void Jacoby(int n, tdata **U, tdata **F)
//{
//    tdata error = 1.0;
//
//    tdata **Y = new tdata*[n+1];
//    tdata **f = new tdata*[n+1];
//    for (int i = 0; i <= n; i++)
//    {
//        Y[i] = new tdata[n+1];
//        f[i] = new tdata[n+1];
//    }
//
//    //    init(n+1, Y);
//    zeroMtx(n+1, f);
//    cout.precision(16);
//
//    int k = 0;
//
//    tdata start_time = clock();
//
//    do
//    {
//        k++;
//        for (int i = 1; i < n; i++)
//            for (int j = 1; j < n; j++)
//                Y[i][j] = U[i][j];
//
//        for (int i = 1; i < n; i++)
//            for (int j = 1; j < n; j++)
//                U[i][j] = (cft1*(Y[i-1][j] + Y[i+1][j] + Y[i][j-1] + Y[i][j+1]) + F[i][j])/cft2;
//
//        CountNev(n+1, U, f);
//        error = MNormDlt(n+1, f, F);
//        if (vyvod) cout<<"k = "<<k<<";  Норма невязки = "<<MNormDlt(n+1, f, F)<<endl;
//
//        //        error = MNormDlt(n+1, Y, U);
//        //        if (vyvod) cout<<"err = "<<error<<endl;
//    }
//    while((error >= eps)&&(k<=15000));
//
//    tdata end_time = clock();
//
//    if (vyvod) cout<<"Метод Якоби сошелся за "<<k<<" итераций."<<endl;
//    cout << "Время расчета " << (end_time - start_time) / CLOCKS_PER_SEC <<" секунд."<< endl;
//
//    for (int i = 0; i <= n; i++)
//    {
//        delete [] Y[i];
//        delete [] f[i];
//    }
//    delete [] Y;
//    delete [] f;
//}

//void CountNev(int n, tdata **U, tdata **f)
//{
//    for (int i = 1; i < n-1; i++)
//        for (int j = 1; j < n-1; j++)
//            f[i][j] = -cft1*(U[i-1][j] + U[i+1][j] + U[i][j+1] + U[i][j-1] - 4.0*U[i][j]) + K2*U[i][j];
//}
//
//tdata MNormDlt(int n, tdata **A, tdata **B)
//{
//    tdata err = 0.0;
//
//    if (errtype == 2)
//    {
//        for (int i = 0; i < n; i++)
//            for (int j = 0; j < n; j++)
//                err += pow((A[i][j] - B[i][j]),2);
//
//        err /= cft1;
//        err = sqrt(err);
//    }
//    else
//    {
//        for (int i = 0; i < n; i++)
//            for (int j = 0; j < n; j++)
//                if (fabs(A[i][j] - B[i][j]) > err) err = fabs(A[i][j] - B[i][j]);
//    }
//
//    return err;
//}
//

//double rhsFunc(double x)
//{
//    return pi*pi*sin(pi*x);
//}
//
//int main(int argc, char* argv[])
//{
//    MPI_Init(&argc, &argv);
//
//    double h, tau;
//    int n;
//    double * u;
//    double * uhat;
//    double * rhs;
//
//    int globN;
//    double Tend;
//    int rank, nproc;
//    MPI_Comm_size(MPI_COMM_WORLD, &nproc);
//    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
//
//    int * locN;
//    double * u0;
//    double * uRes;
//    double * rhsGlob;
//    if (rank == 0)
//    {
//        Tend = 5;
//        globN = 200;
//        locN = new int[nproc];
//        for (int i = 0; i < nproc; i++)
//            locN[i] = (globN-1) / nproc;
//        for (int i = 0; i < (globN-1)%nproc; i++)
//            locN[i] += 1;
//        h = 1.0 / double(globN);
//        tau = h*h*0.25;
//        u0 = new double[globN + 1];
//        for (int i = 0; i <= globN; i++)
//            u0[i] = 0;
//
//        uRes = new double[globN + 1];
//        uRes[0] = uRes[globN] = 0;
//        rhsGlob = new double[globN + 1];
//        for (int i = 0; i <= globN; i++)
//            rhsGlob[i] = rhsFunc(h*i);
//    }
//
//    MPI_Bcast(&tau, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
//    MPI_Bcast(&h, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
//    MPI_Bcast(&Tend, 1, MPI_DOUBLE, 0, MPI_COMM_WORLD);
//    MPI_Scatter(locN, 1, MPI_INT, &n, 1, MPI_INT, 0, MPI_COMM_WORLD);
//
//    n += 2;
//    u = new double[n];
//    uhat = new double[n];
//    rhs = new double[n];
//    if (rank == 0) u[0] = uhat[0] = 0;
//    if (rank == nproc-1) u[n-1] = uhat[n-1] = 0;
//
//    int * uDispl;
//    if (rank == 0)
//    {
//        uDispl = new int[nproc];
//        uDispl[0] = 1;
//        for (int i = 1; i < nproc; i++)
//            uDispl[i] = uDispl[i - 1] + locN[i-1];
//    }
//
//    MPI_Scatterv(u0, locN, uDispl, MPI_DOUBLE, u+1, n - 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
//    MPI_Scatterv(rhsGlob, locN, uDispl, MPI_DOUBLE, rhs+1, n - 2, MPI_DOUBLE, 0, MPI_COMM_WORLD);
//
//    int * exchCounts = new int [nproc];
//    int * sendDisp = new int[nproc];
//    int * recvDisp = new int[nproc];
//    for (int i = 0; i < nproc; i++)
//        exchCounts[i] = sendDisp[i] = recvDisp[i] = 0;
//
//    if (rank < nproc-1)
//    {
//        exchCounts[rank+1] = 1;
//        sendDisp[rank + 1] = n - 2;
//        recvDisp[rank + 1] = n - 1;
//    }
//
//    if (rank > 0)
//    {
//        exchCounts[rank - 1] = 1;
//        sendDisp[rank - 1] = 1;
//        recvDisp[rank - 1] = 0;
//    }
//
//    double tCurrent = 0;
//
//    //if (rank == 0)
//    //{
//    //
//    //    cout << "etalon: ";
//    //    for (int i = 0; i <= globN; i++)
//    //        cout << rhsGlob[i] << " ";
//    //    cout << endl;
//    //}
//
//    do
//    {
//        MPI_Alltoallv(u, exchCounts, sendDisp, MPI_DOUBLE, u, exchCounts, recvDisp, MPI_DOUBLE, MPI_COMM_WORLD);
//        //MPI_Alltoallv(rhs, exchCounts, sendDisp, MPI_DOUBLE, rhs, exchCounts, recvDisp, MPI_DOUBLE, MPI_COMM_WORLD);
//        //cout << rank << " : ";
//        //for (int i = 0; i < n; i++)
//        //    cout << rhs[i] << " ";
//        //cout << endl;
//
//        for (int i = 1; i < n - 1; i++)
//            uhat[i] = u[i] + tau*(rhs[i]+(u[i-1]-2*u[i]+u[i+1])/(h*h));
//
//        // сохранение промежуточных данных ...
//
//        swap(uhat, u);
//        tCurrent += tau;
//    } while (tCurrent < Tend);
//
//    MPI_Gatherv(u + 1, n - 2, MPI_DOUBLE, uRes, locN, uDispl, MPI_DOUBLE, 0, MPI_COMM_WORLD);
//
//    if (rank == 0)
//    {
//        ofstream outFile;
//        outFile.open("res.dat");
//        for (int i = 0; i <= globN; i++)
//            outFile << i*h << " " << uRes[i] << endl;
//        outFile.close();
//    }
//
//    MPI_Finalize();
//    return 0;
//}
