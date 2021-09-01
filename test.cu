#include "ParallelInstDel.cu"
#include<iostream>


int N = 1200;
int B = 1024;



int main(int argc, char **argv)
{

	FILE *fptr;

	//File Opening for read
	char *fileName = argv[1];
	fptr = fopen(fileName, "r");
	cudaEvent_t start, stop;
	

	//checking if file ptr is NULL
	if (fptr == NULL)
	{
		printf("input.txt file failed to open.");
		return 0;
	}
	Heap* heap;

	int operationCount;
	int operation;
	int batchSize;
	
	fscanf(fptr, "%d ", &operationCount);
	

	for (int i = 0; i < operationCount; i++)
	{
		fscanf(fptr, "%d ", &operation);

		if (operation == 1)
		{
			int itemCount;
			heap = createHeap();
			initHeap(heap, N, B);

			fscanf(fptr, "%d ", &itemCount);
			fscanf(fptr, "%d ", &batchSize);
			int* items = (int*)malloc(itemCount * sizeof(int));
			int* g_items;
			time_t t;
			srand((unsigned)time(&t));
			for (int i = 0; i < itemCount; i++)
			{
				
				fscanf(fptr, "%d ", &items[i]);
				

			}
			int blockDim = batchSize / 2;

			int gridDim = ceil(itemCount / batchSize);
			cudaMalloc(&g_items, itemCount * sizeof(int));
			cudaMemcpy(g_items, items, itemCount * sizeof(int), cudaMemcpyHostToDevice);
			cudaEventCreate(&start);
			cudaEventCreate(&stop);
			float milliseconds = 0;
			cudaEventRecord(start, 0);
			insert << <gridDim, blockDim >> > (heap, g_items, itemCount, batchSize);
			cudaDeviceSynchronize();
			cudaEventRecord(stop, 0);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&milliseconds, start, stop);
			cudaError_t err = cudaGetLastError();
			printf("Addition Time : %f\n", milliseconds);
			printf("error=%d,%s,%s\n", err, cudaGetErrorName(err), cudaGetErrorString(err));
			//getCurrentSize(heap);
		}
		else if (operation == 2)
		{
			int deleteCount;
			fscanf(fptr, "%d ", &deleteCount);
			printf("DeleteCount %d\n", deleteCount);
			int gridDim = deleteCount;
			int blockDim = batchSize / 2;
			int* deletedItems = (int*)malloc(deleteCount * batchSize * sizeof(int));
			int* g_delItem;
			cudaMalloc(&g_delItem, deleteCount * batchSize * sizeof(int));
			cudaEventCreate(&start);
			cudaEventCreate(&stop);
			float milliseconds = 0;
			cudaEventRecord(start, 0);
			deleteKey << <gridDim, blockDim >> > (heap, g_delItem);
			cudaDeviceSynchronize();
			cudaEventRecord(stop, 0);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&milliseconds, start, stop);
			cudaError_t err = cudaGetLastError();
			printf("error=%d,%s,%s\n", err, cudaGetErrorName(err), cudaGetErrorString(err));
			printf("DELETION TIME : %f\n", milliseconds);
			/* this function can be used to view the contents of heap.*/
			getCurrentSize(heap,batchSize);

         }
		else
		{
			int opCode;
			int batch;
			int itemCount;
			int taskCount;
			int deleteCount = 0;
		    heap = createHeap();
			initHeap(heap, N, B);
			//fscanf(fptr, "%d ", &opCode);
			fscanf(fptr, "%d ", &batch);
			fscanf(fptr, "%d ", &itemCount);
			int* items = (int*)malloc(itemCount * sizeof(int));
			int* g_items;
			for (int i = 0; i < itemCount; i++)
			{
				fscanf(fptr, "%d ", &items[i]);

			}
			cudaMalloc(&g_items, itemCount * sizeof(int));
			cudaMemcpy(g_items, items, itemCount * sizeof(int), cudaMemcpyHostToDevice);
			fscanf(fptr, "%d ", &taskCount);
			bool* task;
			bool*  g_task;
			int gridDim = taskCount;
			int blockDim = batch / 2;
			task = (bool*)malloc(gridDim * sizeof(bool));
			printf("TaskCount %d\n", taskCount);
			for (int i = 0; i < taskCount; i++)
			{
				int t;
				fscanf(fptr, "%d ", &t);
				task[i] = (t == 1) ? true : false;
				if (t == 0)
					deleteCount++;
			}
			printf("Delete count %d\n", deleteCount);
			cudaMalloc(&g_task, gridDim * sizeof(bool));
			cudaMemcpy(g_task, task, gridDim * sizeof(bool), cudaMemcpyHostToDevice);
			int* deletedItems = (int*)malloc(deleteCount * batch * sizeof(int));
			int* g_delItem;
			cudaMalloc(&g_delItem, deleteCount * batch * sizeof(int));
			cudaEventCreate(&start);
			cudaEventCreate(&stop);
			float milliseconds = 0;
			cudaEventRecord(start, 0);
			insert_Delete << <gridDim, blockDim >> > (heap, g_items, itemCount, batch, g_delItem, g_task);
			cudaDeviceSynchronize();
			cudaEventRecord(stop, 0);
			cudaEventSynchronize(stop);
			cudaEventElapsedTime(&milliseconds, start, stop);
			cudaError_t err = cudaGetLastError();
			//getCurrentSize(heap,batch);
			cudaMemcpy(deletedItems, g_delItem, deleteCount * batch * sizeof(int), cudaMemcpyDeviceToHost);
			printf("Concurrent Insertion Deletion Time : %f\n", milliseconds);
			printf("Deleted Items \n");
			for (int j = 0; j < deleteCount; j++)
			{
				if (deletedItems[j*batch] == -1)
					continue;
				for (int k = 0; k < batch; k++)
				{ 

					printf("k %d val %d\n", k, deletedItems[j*batch + k]);
				}
			}
			
		}

	}

	

}