#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <stdbool.h>
#include <mpi.h>

int numProcs;
int myrank;
double *** network;
double **weights1;
double **weights2;
double **weights3;
double** outputs;
double * output;
double * sigDevOut;
double ** outputDotProd2;
int numLayers=3;
double lr=1;
int epochs=10;
struct rowsCol* networkWeights;
struct rowsCol{
    int r;
    int c;
};

void init(){
    
    int r=4;
    int c=3;
    
    struct rowsCol rc1;
    rc1.r=r;
    rc1.c=c;

    weights1 = (double **)malloc(c * sizeof(double *));
    for (int i=0; i<c; i++)
         weights1[i] = (double *)malloc(r * sizeof(double));
    
    for(int i=0;i<c;i++)
        for(int j=0;j<r;j++)
            weights1[i][j]=rand()%3+1;
    
    

    r=3;
    c=3;
    
    struct rowsCol rc2;
    rc2.r=r;
    rc2.c=c;


    weights2 = (double **)malloc(c * sizeof(double *));
    for (int i=0; i<c; i++)
         weights2[i] = (double *)malloc(r * sizeof(double));
    
    for(int i=0;i<c;i++)
        for(int j=0;j<r;j++)
            weights2[i][j]=rand()%4+1;
    
    

    r=3;
    c=2;
    
    
    struct rowsCol rc3;
    rc3.r=r;
    rc3.c=c;


    weights3 = (double **)malloc(c * sizeof(double *));
    for (int i=0; i<c; i++)
         weights3[i] = (double *)malloc(r * sizeof(double));
    
    for(int i=0;i<c;i++)
        for(int j=0;j<r;j++)
            weights3[i][j]=rand()%3+1;
    

    int numLayers=3;
    network = (double ***)malloc(numLayers*sizeof(double**));

    network[0]=weights1;
    network[1]=weights2;
    network[2]=weights3;

    networkWeights= (struct rowsCol*)malloc(3 *sizeof(struct rowsCol));
    networkWeights[0]=rc1;
    networkWeights[1]=rc2;
    networkWeights[2]=rc3;
    

    if(myrank==numProcs-1)
		for(int i=0;i<numLayers;i++)
		{
			printf("layer number %d initial weights \n", i);
			int K,L; //r,c
			K=networkWeights[i].r;
			L=networkWeights[i].c;
			for(int k =0; k<L;k++){
			for(int l=0;l<K;l++)
				printf("%lf ",network[i][k][l]);
			printf("\n");
			}
			printf("rows: %d Collumns: %d",networkWeights[i].r,networkWeights[i].c );
			printf(" \n");
			
		}


}


double* sigmoid(double* x,int n){

    for(int i =0; i<n;i++){
        x[i]= 1/(1 + exp(-x[i]));
    }
    return x;
}

double* sigmoidDerivative(double* x,int n)
{
    sigDevOut=(double *)malloc(n*sizeof(double));
    for(int i =0; i<n;i++){
        sigDevOut[i]=x[i] * (1 - x[i]);
    }
    return sigDevOut;

}

double* dotProd(double**w,int r,int c, double *x,int n)
{
    if(r!=n){
        printf("error %d with %d", r,n);
        exit(0);
    }
	output= (double*)malloc(c * sizeof(double));
	if(myrank!=numProcs-1)
	{
			if(myrank<c)
			{
				output[myrank]=0;
				for(int j = 0;j<r;j++)
				{
					output[myrank]+=x[j]*w[myrank][j];

				}
				MPI_Send(&output[myrank], 1, MPI_DOUBLE, numProcs-1, 0, MPI_COMM_WORLD);
			}
	}
	else
	{
		double recieved;

		for(int i =0 ; i < numProcs-1; i++)
		{
			if(i<c)
			{
				MPI_Recv(&recieved, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);      
				output[i]=recieved;
			}
		}	
	}
	
    return output;
}

double** dotProd2(double*x1,int rx1, double *x2,int cx2)
{
    outputDotProd2= (double**)malloc(rx1 * sizeof(double*));
    for(int i =0 ;i < rx1;i++)
        outputDotProd2[i]=(double*)malloc(cx2 * sizeof(double));
    
    if(myrank!=numProcs-1)
	{
		if(myrank<rx1)
		{
			for(int i = 0;i<cx2;i++)
			{
				outputDotProd2[myrank][i]=x1[myrank]*x2[i];
			}
			MPI_Send(outputDotProd2[myrank], cx2, MPI_DOUBLE, numProcs-1, myrank, MPI_COMM_WORLD);
		}
	}
	else
	{
		
		for(int i =0 ; i < numProcs-1; i++)
		{
			if(i<rx1)
				MPI_Recv(outputDotProd2[i], cx2, MPI_DOUBLE, i, i, MPI_COMM_WORLD, MPI_STATUS_IGNORE);   

		}
			
	}
	


    return outputDotProd2;

}


void MatSub(int idx,double ** layerAdjustment,int r,int c)
{
    for(int i =0;i<r;i++)
    {
        for(int j=0;j<c;j++)
        {
            network[idx][i][j]-=layerAdjustment[i][j]*lr; 
        }
    }
}


double** predict(double * x)
{
    outputs = (double **)malloc( numLayers * sizeof(double*)); 
    double *newInputs= x;
    for(int i =0 ; i<numLayers;i++)
    {
        int n;
        if(i==0)
            n=4;
        else if (n==numLayers-1)
            n=2;
        else 
            n=3;    
        outputs[i]=sigmoid(dotProd(network[i],networkWeights[i].r,networkWeights[i].c,newInputs,n),networkWeights[i].c);
        newInputs=outputs[i];
 
    }

    //free(newInputs);

    return outputs;
}

void printWeights()
{

    printf("printing weights of network \n");
    for(int i=0;i<numLayers;i++)
    {
        printf("layer number %d  weights \n", i);
        int K,L; //r,c
        K=networkWeights[i].r;
        L=networkWeights[i].c;
        for(int k =0; k<L;k++){
         for(int l=0;l<K;l++)
            printf("%lf ",network[i][k][l]);
        printf("\n");
        }
        printf("\n");
    }
}

void train(double **x,double **y,int epochs,int numInputs)
{

	for(int e=0;e<=epochs;e++)
	{

	for(int i =0;i<numInputs;i++)
		{
		
			outputs=predict(x[i]);

			//step 1
			double * layerError = (double *)malloc(2 * sizeof(double));
			layerError[0]= y[i][0]-outputs[numLayers-1][0];
			layerError[1]= y[i][1]-outputs[numLayers-1][1];
			
			double * layerDelta = (double *)malloc(2 * sizeof(double));
			layerDelta[0]=layerError[0]*sigmoidDerivative(outputs[numLayers-1],2)[0];
			layerDelta[1]=layerError[1]*sigmoidDerivative(outputs[numLayers-1],2)[1];
			
			double ** layerAdjustment= dotProd2(outputs[numLayers-2],3,layerDelta,2);
			
			MatSub(numLayers-1,layerAdjustment,2,3);


			//step 2 
			layerError = dotProd(network[numLayers-1],3,2,layerDelta,3);

			layerDelta = (double *)malloc(3 * sizeof(double));
			layerDelta[0]=layerError[0]*sigmoidDerivative(outputs[numLayers-2],3)[0];
			layerDelta[1]=layerError[1]*sigmoidDerivative(outputs[numLayers-2],3)[1];
			layerDelta[2]=layerError[2]*sigmoidDerivative(outputs[numLayers-2],3)[2];


			layerAdjustment= dotProd2(outputs[numLayers-2],3,layerDelta,3);
			
			MatSub(numLayers-2,layerAdjustment,3,3);


			//step 3
			layerError = dotProd(network[numLayers-2],3,2,layerDelta,3);

			layerDelta = (double *)malloc(3 * sizeof(double));
			layerDelta[0]=layerError[0]*sigmoidDerivative(outputs[numLayers-3],3)[0];
			layerDelta[1]=layerError[1]*sigmoidDerivative(outputs[numLayers-3],3)[1];
			layerDelta[2]=layerError[2]*sigmoidDerivative(outputs[numLayers-3],3)[2];

			
			layerAdjustment= dotProd2(x[i],4,layerDelta,3);
			
			MatSub(numLayers-3,layerAdjustment,3,3);
			
			
		}
		if(myrank==numProcs-1)
		{
			printf("\n \n");
			printWeights();
			printf("\n done with epoch %d \n ",e);
		}
	}
}



int main(int argc, char **argv) 
{
	
	MPI_Init(NULL, NULL);
	
	MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
	MPI_Comm_size(MPI_COMM_WORLD, &numProcs);

	if(numProcs!=5)
	{
		printf("number of processes should be exactly 5 for the project to run without problems !");
		MPI_Finalize();	
		exit(0);	
	}
	int numInputs;
	double **x;
	double **y;
	if(myrank==0)
    {

        scanf("%d",&numInputs);
		x=malloc(numInputs*sizeof(double*));
		y=malloc(numInputs*sizeof(double*));
		for(int i=0;i<numInputs;i++)
		{
			x[i]=malloc(4*sizeof(double));
			y[i]=malloc(2*sizeof(double));
		}

		for(int i=0;i<numInputs;i++)
        {
            scanf("%lf %lf %lf %lf",&x[i][0],&x[i][1],&x[i][2],&x[i][3]);
        }


		MPI_Bcast(&numInputs,1,MPI_INT,0,MPI_COMM_WORLD);
		
		for(int j=0;j<numInputs;j++)
			MPI_Bcast(x[j],4,MPI_DOUBLE,0,MPI_COMM_WORLD);



		 for(int i=0;i<numInputs;i++)
        {
            scanf("%lf %lf",&y[i][0],&y[i][1]);
        }

    	for(int j=0;j<numInputs;j++)
			MPI_Bcast(y[j],2,MPI_DOUBLE,0,MPI_COMM_WORLD);
	    
    }
    else
    {
		MPI_Bcast(&numInputs,1,MPI_INT,0,MPI_COMM_WORLD);
		
		x=malloc(numInputs*sizeof(double*));
		y=malloc(numInputs*sizeof(double*));
		for(int i=0;i<numInputs;i++)
		{
			x[i]=malloc(4*sizeof(double));
			y[i]=malloc(2*sizeof(double));
		}

        for(int i =0;i<numInputs;i++)
			MPI_Bcast(x[i],4,MPI_DOUBLE,0,MPI_COMM_WORLD);
   

	    for(int i =0;i<numInputs;i++)
			MPI_Bcast(y[i],2,MPI_DOUBLE,0,MPI_COMM_WORLD);

    }

	init();
	   

	train(x,y,epochs,numInputs);

	//testing the training
	double testX[4]={1,0,0,1};
	outputs= predict(testX);
	double y2=outputs[numLayers-1][1];
	double y1=outputs[numLayers-1][0];
	
	if(myrank==numProcs-1)
	{

		printf("for 0 1 1 1 the out put is %lf  %lf \n",y1,y2);
		if(y1>y2)
		{
			printf(" 0 \n");
		}
		else
		{
			printf(" 1 \n");        
		}
	}



	// free(outputs);
	// free(weights1);
	// free(weights2);
	// free(weights3);
	// free(network);
	// free(networkWeights);

	
	MPI_Finalize();	
    
	return 0;
}
