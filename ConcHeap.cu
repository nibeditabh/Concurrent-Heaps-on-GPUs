#include<stdio.h>
#include<stdlib.h>
#include<malloc.h>
#include<cuda.h>
#include<stdbool.h>
#include<math.h>




struct node
{
  int* batch;
  int target;

};
class Heap
{
     public:
     int* partialBuffer;
     int partialBufSize;
	 struct node* nodes;
	 int maxSize;
	 int currentSize;
	 int* mutex;
	 int fullBatchSize;
     __device__ void initHeap(int max,int batchSize);
     __device__ int getMaxSize();
     __device__ void insert(int* insertItems);
     __device__ void deleteItems();
};

__device__ void Heap::initHeap(int max,int batchSize)
{
  nodes=(struct node*)malloc(max*sizeof(struct node));
  partialBuffer=(int*)malloc((batchSize-1)*sizeof(int));
  memset(partialBuffer, 0 , (batchSize - 1) * sizeof(int));
  for(int i=0;i<max;i++)
  {
	  nodes[i].batch=(int*)malloc(batchSize*sizeof(int));
	  nodes[i].target = 0;
     memset(nodes[i].batch,0,sizeof(nodes[i].batch));

  }
  mutex=(int*)malloc(max*sizeof(int));
  memset(mutex,0,sizeof(mutex));

  maxSize=max;
  currentSize=0;
  partialBufSize=0;
  fullBatchSize = batchSize;
 }

 __device__ int Heap::getMaxSize()
{
   return currentSize;
}


__global__ void init(Heap* d_heap,int max,int maxBatchSize)
{
    d_heap->initHeap(max, maxBatchSize);
}

__global__ void getSize(Heap* d_heap,int batchSize)
{
	for (int i = 0; i < d_heap->getMaxSize(); i++)
	{
		for (int j = 0; j < batchSize; j++)
		{
			printf(" j %d val %d \n", j, d_heap->nodes[i].batch[j]);
		}
		printf("end of i %d\n", i);
	}
	printf("Size of heap %d \n", d_heap->getMaxSize());
}

Heap* createHeap()
{
  Heap*   d_heap;
  Heap*	  h_heap;
  h_heap=(Heap*)malloc(sizeof(Heap));
  cudaMalloc(&d_heap, sizeof(Heap));
  cudaMemcpy(d_heap, h_heap, sizeof(Heap), cudaMemcpyHostToDevice);
  return d_heap;
}


void initHeap(Heap* d_heap,int max,int maxbatchSize)
{
  init<<<1,1>>>(d_heap,max,maxbatchSize);
  cudaDeviceSynchronize();
}

void getCurrentSize(Heap* d_heap,int batchSize)
{  
	getSize << <1, 1 >> > (d_heap,batchSize);
	cudaDeviceSynchronize();
	
}




