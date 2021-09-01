#include<stdio.h>
#include<stdlib.h>
#include<malloc.h>
#include<cuda.h>
#include<stdbool.h>
#include "ConcHeap.cu"


/* Concurrent Operations in a Generalized Heap curated for GPU.*/



__device__ const int k = 32;

struct points
{
	int x_start;
	int y_start;
};


__device__ struct points diagonalIntersection(int* a, int* b, int tid, int p, int n, int m, int diag)
{


	struct points pt;
	if (tid == 0)
	{
		pt.x_start = 0;
		pt.y_start = 0;

		return pt;
	}
	int halfa = diag > n ? 1 : 0;
	int halfb = diag > m ? 1 : 0;
	int b_offset = ((halfa == 0) ? 0 : (diag - n));
	int a_offset = ((halfb == 0) ? 0 : (diag - m));
	if (halfa&&halfb)
	{
		diag = n + m - diag;
	}
	else if (halfa && !halfb)
	{
		diag = n;
	}
	else if (halfb && !halfa)
	{
		diag = m;

	}
	int left = 0;
	int right = diag;
	int atid = -1, btid = -1;
	while (left <= right)
	{
		int mid = (left + right) / 2;
		atid = diag - mid + a_offset;
		btid = mid + b_offset;
		pt.x_start = atid;
		pt.y_start = btid;

		//printf("atid %d btid %d mid %d tid %d\n", atid, btid, mid, tid);
		if ((btid >= m && (btid - 1) >= 0 && a[atid] > b[btid - 1]) ||
			(atid >= n && (atid - 1) >= 0 && a[atid - 1] < b[btid]))
		{
			//printf("in 1st atid %d btid %d tid %d diag %d\n", atid,btid,tid,diag);

			break;
		}
		if ((atid - 1) >= 0 && (btid - 1) >= 0 && a[atid] >= b[btid - 1] && a[atid - 1] < b[btid])
		{
			//printf("in 2nd atid %d btid %d tid %d diag %d\n", atid,btid,tid,diag);

			break;
		}
		else if (atid == n || (btid - 1 >= 0 && a[atid] >= b[btid - 1]))
		{
			//printf("in 3rd atid %d btid %d tid %d diag %d\n", atid, btid,tid,diag);
			left = mid + 1;
		}
		else if (btid == 0)
		{
			if (a[atid] <= b[btid])
			{
				break;
			}
			else
			{
				left = mid + 1;
			}
		}
		else
		{
			//printf("in 4th atid %d btid %d tid %d diag %d\n", atid, btid,tid,diag);
			right = mid - 1;
		}
	}
	//printf("in pt.x %d pt.y %d\n", pt.x_start,pt.y_start);

	return pt;

}

__device__ void mergeArrays(int* a, int* b, int k, int l, int* mergedArray, int tempStart, int length, int n, int m)
{
	// printf("mergedArray length );


	int tempi = 0;
	while (tempi < length&&k < n&&l < m)
	{

		if (a[k] < b[l])
		{

			mergedArray[tempi + tempStart] = a[k];
			//printf("a[k] %d \n", a[k]);
			k++;
		}
		else
		{
			mergedArray[tempi + tempStart] = b[l];
			//printf("b[l] %d \n", b[l]);
			l++;
		}

		tempi++;
	}
	while (tempi < length&&k < n)
	{

		mergedArray[tempi + tempStart] = a[k];
		k++;
		tempi++;
	}
	while (tempi < length&&l < m)
	{

		mergedArray[tempi + tempStart] = b[l];
		l++;
		tempi++;
	}







}

__device__ void placeItemsCorrectly(int* buffer, int* temp,int* mergedArray,int size)
{
	
	
	if (size < k)
	{
		for (int i = 0; i < size; i++)
		{
			buffer[i] = mergedArray[i];
		}
	}
	else
	{
		for (int i = 0; i < k; i++)
		{
			temp[i] = mergedArray[i];
		}
		int index = 0;
		for (int i = k; i < size; i++)
		{
			buffer[index++] = mergedArray[i];
		}
	}
}

__device__ void copyKeys(int* source, int* destination) 
{
	for (int i = 0; i < k; i++)
	{
		destination[i] = source[i];
	}
}

/*The below function performs parallel Insertion to the Heap.

The insertions are implemented as block-level operations. */

__global__ void insert(Heap* d_heap, int* totalItems, int itemCount, int batchSize)
{
	__shared__ int temp[10000];
	__shared__ bool process;
	__shared__ int tar;
	__shared__ int cur;
	__shared__ int level;
	__shared__ int mergedArray[3 * k];

	int blockId = blockIdx.y*gridDim.x + blockIdx.x;
	int blockLocalThreadId = blockDim.x*threadIdx.y + threadIdx.x;
	int globalThreadId = blockId * blockDim.x * blockDim.y + blockLocalThreadId;
	int partPerBlock = batchSize;
	int part = partPerBlock / blockDim.x;
	int partPerThread = (blockLocalThreadId == blockDim.x - 1) ? partPerBlock - (blockDim.x - 1)*part : part;



	for (int i = 0; i < partPerThread; i++)
	{

		temp[blockLocalThreadId*partPerThread + i] = totalItems[blockId*batchSize + blockLocalThreadId * partPerThread + i];

	}



	__syncthreads();

	/* parallel bitonic sort to sort the insert elements.*/

	for (int l = 2; l <= batchSize; l <<= 1) {
		for (int j = l >> 1; j > 0; j = j >> 1)
		{
			int part1, part2;
			int index = blockLocalThreadId * (part);
			int start = index;
			while (start != (index + partPerThread))
			{
				part2 = start ^ j;

				if ((part2) > start) {
					if ((start&l) == 0) {
						if (temp[start] > temp[part2]) {
							int temporary = temp[start];
							temp[start] = temp[part2];
							temp[part2] = temporary;
						}
					}
					if ((start&l) != 0) {
						if (temp[start] < temp[part2]) {
							int temporary = temp[start];
							temp[start] = temp[part2];
							temp[part2] = temporary;
						}
					}
				}
				start++;
			}



		}
		__syncthreads();



	}





	if (blockLocalThreadId == 0)
	{

		process = true;
		//	
		while (atomicCAS(&d_heap->mutex[0], 0, 1) != 0);



	}


	__syncthreads();



	/* GPU-merge path procedure to merge two sorted arrays.*/

	int size = d_heap->partialBufSize + batchSize;
	int length = (size) / blockDim.x;
	int diagonalNum = (blockLocalThreadId)*(size) / blockDim.x;
	struct points p;
	p = diagonalIntersection(d_heap->partialBuffer, temp, blockLocalThreadId, blockDim.x, d_heap->partialBufSize, batchSize, diagonalNum);
	int tempStart = (blockLocalThreadId)*(size / blockDim.x);
	mergeArrays(d_heap->partialBuffer, temp, p.x_start, p.y_start, mergedArray, tempStart, length, d_heap->partialBufSize, batchSize);

	__syncthreads();



	if (blockLocalThreadId == 0)
	{
		/*If size greater than k, select k smallest elements for insertion.*/

		if (size >= k)
		{

			placeItemsCorrectly(d_heap->partialBuffer, temp, mergedArray, size);

			d_heap->partialBufSize = size - k;


		}
		else /* else,1. Store elements in partialBuffer
			         2. Perform sorting of partialBuf and root node's keys and return.
		           3. Store k smallest keys in root nodes and rest in partialBuf and return. */
		{
			placeItemsCorrectly(d_heap->partialBuffer, temp, mergedArray, size);
			if (d_heap->currentSize != 0)
			{

				mergeArrays(d_heap->nodes[0].batch, d_heap->partialBuffer, 0, 0, mergedArray, 0, k + size, k, size);
				placeItemsCorrectly(d_heap->nodes[0].batch, d_heap->partialBuffer, mergedArray, k + size);
			}
			atomicCAS(&d_heap->mutex[0], 1, 0);
			process = false;

		}


	}

	__syncthreads();

	if (process)
	{
		
		if (blockLocalThreadId == 0)
		{
			tar = d_heap->currentSize++;
			cur = 0;
			level = __log2f(d_heap->currentSize);

			if (tar != 0)
			{
				while (atomicCAS(&d_heap->mutex[tar], 0, 2) != 0); //2 here implies targetnode.
			}


		}
		__syncthreads();

		while (cur != tar)  /* performs a top-down insertion of elements.*/
		{

			if (d_heap->mutex[tar] == 3)
			{

				break;
			}
			int size = k + batchSize;
			int length = (size) / blockDim.x;
			int diagonalNum = (blockLocalThreadId)*(size) / blockDim.x;
			struct points p = diagonalIntersection(d_heap->nodes[cur].batch, temp, blockLocalThreadId, blockDim.x, k, batchSize, diagonalNum);
			int tempStart = (blockLocalThreadId)*(size / blockDim.x);
			mergeArrays(d_heap->nodes[cur].batch, temp, p.x_start, p.y_start, mergedArray, tempStart, length, k, batchSize);
			__syncthreads();



			if (blockLocalThreadId == 0)
			{


				placeItemsCorrectly(temp, d_heap->nodes[cur].batch, mergedArray, size);
				//	printf("BlockId enterssssss %d level %d tar %d\n", blockId, level, tar);
				cur = ((tar + 1) >> --level) - 1;

				if (cur != tar)
				{
					while (atomicCAS(&d_heap->mutex[cur], 0, 1) != 0);
				}

				int par = ceilf((float)cur / 2) - 1;
				//printf("blockId %d par %d tar %d curr %d\n", blockId, par, tar,cur);
				atomicExch(&d_heap->mutex[par], 0);
			}
			__syncthreads();
		}

		if (blockLocalThreadId == 0)
		{
			int tstate = atomicCAS(&d_heap->mutex[tar], 2, 1);
			tar = tstate == 2 ? tar : 0;
			for (int i = 0; i < k; i++)
			{
				d_heap->nodes[tar].batch[i] = temp[i];
				//printf("values by blockId %d i %d val %d\n",blockId,i,d_heap->nodes[cur].batch[i]);
			}


			if (tar != cur)
				atomicExch(&d_heap->mutex[tar], 0);
			atomicExch(&d_heap->mutex[cur], 0);



		}

	}

}



/*The below function performs parallel top-down deletion of the Heap.

The deletions are implemented as block-level operations. */


__global__ void deleteKey(Heap* d_heap, int* deleteItems)
{
	__shared__ int l;
	__shared__ int r;
	__shared__ bool process;
	__shared__ bool process1;
	__shared__ bool process2;
	__shared__ bool isLeft;
	__shared__ int tar;
	__shared__ int cur;
	__shared__ int lstate;
	__shared__ int rstate;
	__shared__ int mergedArray[3 * k];
	bool flag = false;


	int blockId = blockIdx.y*gridDim.x + blockIdx.x;
	int blockLocalThreadId = blockDim.x*threadIdx.y + threadIdx.x;
	int globalThreadId = blockId * blockDim.x * blockDim.y + blockLocalThreadId;


	if (blockLocalThreadId == 0)
	{
		process = true;
		process1 = true;
		process2 = true;
		isLeft = true;
		while (atomicCAS(&d_heap->mutex[0], 0, 1) != 0);
	}

	__syncthreads();

	/* If heap is empty, get keys from partial buffer and return. */

	if (blockLocalThreadId == 0 )
	{    
		
		if (d_heap->currentSize == 0)
		{
			if (d_heap->partialBufSize != 0)
			{
				for (int i = 0; i < d_heap->partialBufSize; i++)
				{
					deleteItems[blockId*k + i] = d_heap->partialBuffer[i];
				}
			}

			atomicCAS(&d_heap->mutex[0], 1, 0);
			process = false;
		}
		
	}
	__syncthreads();

	if (process)
	{
		  __syncthreads();

		if (blockLocalThreadId == 0)
		{

			for (int i = 0; i < k; i++)
			{
				deleteItems[blockId*k + i] = d_heap->nodes[0].batch[i];
			}
			tar = --d_heap->currentSize;
			cur = 0;

			int tstate = atomicCAS(&d_heap->mutex[tar], 2, 3);
			if (tstate == 2)
				while (d_heap->mutex[tar] != 0);
			else
			{
				while (atomicCAS(&d_heap->mutex[tar], 0, 1) != 0);
				copyKeys(d_heap->nodes[tar].batch, d_heap->nodes[0].batch);
				atomicCAS(&d_heap->mutex[tar], 1, 0);
			}
		}

		__syncthreads();


		int size = d_heap->partialBufSize + k;
		int length = (size) / blockDim.x;
		int diagonalNum = (blockLocalThreadId)*(size) / blockDim.x;
		struct points p = diagonalIntersection(d_heap->partialBuffer, d_heap->nodes[0].batch, blockLocalThreadId, blockDim.x, d_heap->partialBufSize, k, diagonalNum);
		int tempStart = (blockLocalThreadId)*(size / blockDim.x);
		mergeArrays(d_heap->partialBuffer, d_heap->nodes[0].batch, p.x_start, p.y_start, mergedArray, tempStart, length, d_heap->partialBufSize, k);

		__syncthreads();

		if (blockLocalThreadId == 0)
		{
			placeItemsCorrectly(d_heap->partialBuffer, d_heap->nodes[0].batch, mergedArray, size);
		}

		__syncthreads();


		do
		{ 

			__syncthreads();


			if (blockLocalThreadId == 0)
			{
				l = 2 * cur + 1;
				r = 2 * cur + 2;
				lstate = 1;
				rstate = 1;
				/*printf("blockIdl before %d tar %d l %d curr %d\n", blockId, tar, l, cur);
				while (lstate == 1)
					lstate = atomicCAS(&d_heap->mutex[l], 0, 1);
				printf("blockIdl before %d tar %d l %d curr %d\n", blockId, tar, l, cur);
				if (lstate != 0||l>= d_heap->currentSize)
				{
					atomicCAS(&d_heap->mutex[cur], 1, 0);
					process1 = false;
				}*/

				//printf("blockIdl before %d tar %d l %d mutex %d cur %d\n", blockId, tar, l, d_heap->mutex[l], cur);
				if (l >= tar)
				{
					atomicCAS(&d_heap->mutex[cur], 1, 0);
					process1 = false;
				}
				else /* Acquire lock on the left child. */
				{
					while (lstate == 1)
						lstate = atomicCAS(&d_heap->mutex[l], 0, 1);
				//	printf("blockIdl after %d tar %d l %d\n", blockId, tar, l);
					if (lstate != 0)
					{

						atomicCAS(&d_heap->mutex[cur], 1, 0);
						process1 = false;
					}
				}
			}
			__syncthreads();

			if (process1)
			{
				__syncthreads();

				if (blockLocalThreadId == 0)
				{
					/*printf("blockIdr before %d tar %d r %d curr %d\n", blockId, tar, r, cur);
					while (rstate == 1)
						rstate = atomicCAS(&d_heap->mutex[r], 0, 1);
					printf("blockIdr after %d tar %d r %d curr %d\n", blockId, tar, r, cur);*/

					//printf("blockIdr before  %d tar %d mutextar %d r %d cur %d \n", blockId, tar, d_heap->mutex[r], r, cur);
					if (r >= tar)
					{
						process2 = false;
						rstate = 0;

					}
					else  /* Acquire lock on the right child. */
					{
						while (rstate == 1)
							rstate = atomicCAS(&d_heap->mutex[r], 0, 1);

					}
				//	printf("blockIdr after  %d tar %d r %d \n", blockId, tar, r);

				}
				__syncthreads();


				if (rstate != 0)
				{
					size = 2 * k;
					length = (size) / blockDim.x;
					diagonalNum = (blockLocalThreadId)*(size) / blockDim.x;
					p = diagonalIntersection(d_heap->nodes[cur].batch, d_heap->nodes[l].batch, blockLocalThreadId, blockDim.x, k, k, diagonalNum);
					tempStart = (blockLocalThreadId)*(size / blockDim.x);
					mergeArrays(d_heap->nodes[cur].batch, d_heap->nodes[l].batch, p.x_start, p.y_start, mergedArray, tempStart, length, k, k);
					flag = true;

				}
				__syncthreads();

				if (blockLocalThreadId == 0&&flag)
				{
					placeItemsCorrectly(d_heap->nodes[l].batch, d_heap->nodes[cur].batch, mergedArray, size);
					process2 = false;

				}
				__syncthreads();


				if (process2)
				{
					__syncthreads();

					/* GPU MergePath algo to sort two sorted batch of left and right child node.*/

					size = 2 * k;
					length = (size) / blockDim.x;
					diagonalNum = (blockLocalThreadId)*(size) / blockDim.x;
					p = diagonalIntersection(d_heap->nodes[l].batch, d_heap->nodes[r].batch, blockLocalThreadId, blockDim.x, k, k, diagonalNum);
					tempStart = (blockLocalThreadId)*(size / blockDim.x);
					mergeArrays(d_heap->nodes[l].batch, d_heap->nodes[r].batch, p.x_start, p.y_start, mergedArray, tempStart, length, k, k);

					__syncthreads();

					if (blockLocalThreadId == 0)
					{
						/* if right child has the largest element, hold the left child and unlock right child.*/
						if (d_heap->nodes[l].batch[k - 1] < d_heap->nodes[r].batch[k - 1])
						{
							placeItemsCorrectly(d_heap->nodes[r].batch, d_heap->nodes[l].batch, mergedArray, size);
							atomicCAS(&d_heap->mutex[r], 1, 0);
							isLeft = true;
							if (d_heap->nodes[cur].batch[k - 1] <= d_heap->nodes[l].batch[0])
								process = false;
						}
						else /* if left child has the largest element, hold the right child and unlock the left child.*/
						{
							placeItemsCorrectly(d_heap->nodes[l].batch, d_heap->nodes[r].batch, mergedArray, size);
							atomicCAS(&d_heap->mutex[l], 1, 0);
							isLeft = false;
							if (d_heap->nodes[cur].batch[k - 1] <= d_heap->nodes[r].batch[0])
								process = false;
						}
					}
						__syncthreads();


						if (process)
						{
							__syncthreads();

							size = 2 * k;
							length = (size) / blockDim.x;
							diagonalNum = (blockLocalThreadId)*(size) / blockDim.x;
							if (isLeft)
							{
								p = diagonalIntersection(d_heap->nodes[cur].batch, d_heap->nodes[l].batch, blockLocalThreadId, blockDim.x, k, k, diagonalNum);
								tempStart = (blockLocalThreadId)*(size / blockDim.x);
								mergeArrays(d_heap->nodes[cur].batch, d_heap->nodes[l].batch, p.x_start, p.y_start, mergedArray, tempStart, length, k, k);
							}
							else
							{
								p = diagonalIntersection(d_heap->nodes[cur].batch, d_heap->nodes[r].batch, blockLocalThreadId, blockDim.x, k, k, diagonalNum);
								tempStart = (blockLocalThreadId)*(size / blockDim.x);
								mergeArrays(d_heap->nodes[cur].batch, d_heap->nodes[r].batch, p.x_start, p.y_start, mergedArray, tempStart, length, k, k);
							}


						}

						__syncthreads();

						if (blockLocalThreadId == 0&&process)
						{
							if (isLeft)
							{
								placeItemsCorrectly(d_heap->nodes[l].batch, d_heap->nodes[cur].batch, mergedArray, size);
								atomicCAS(&d_heap->mutex[cur], 1, 0);
								cur = l;
							}
							else
							{
								placeItemsCorrectly(d_heap->nodes[r].batch, d_heap->nodes[cur].batch, mergedArray, size);
								atomicCAS(&d_heap->mutex[cur], 1, 0);
								cur = r;
							}


						}




					}


				}

				__syncthreads();

			}while (process&&process1&&process2);


			__syncthreads();

			if (!process2 || !process)
			{
				if (blockLocalThreadId == 0)
				{
					if (isLeft || !process2)
					{
						
						//atomicExch(&d_heap->mutex[cur], 0);
						//atomicCAS(&d_heap->mutex[cur], 1, 0);
						//atomicExch(&d_heap->mutex[l], 0);
						atomicCAS(&d_heap->mutex[l], 1, 0);
					}
					else
					{
					
						atomicCAS(&d_heap->mutex[r], 1, 0);
						
					}
				}

			}


		if (blockLocalThreadId == 0)
		{
		
			if (!process || !process2)
				atomicCAS(&d_heap->mutex[cur], 1, 0);

		}

		__syncthreads();
	}
	  
		
	}

/*The below function performs parallel bottom-up insertion of the Heap.

The deletions are implemented as block-level operations. */

	
__global__ void bottom_upI(Heap* d_heap, int* totalItems, int itemCount, int batchSize)
	{

		__shared__ int temp[10000];
		__shared__ bool process;
		__shared__ int par;
		__shared__ int cur;
		__shared__ int pState;
		__shared__ int cState;
		__shared__ bool process1;
		__shared__ bool process2;
		__shared__ int mergedArray[3 * k];


		int blockId = blockIdx.y*gridDim.x + blockIdx.x;
		int blockLocalThreadId = blockDim.x*threadIdx.y + threadIdx.x;
		int globalThreadId = blockId * blockDim.x * blockDim.y + blockLocalThreadId;
		int partPerBlock = batchSize;
		int part = partPerBlock / blockDim.x;
		int partPerThread = (blockLocalThreadId == blockDim.x - 1) ? partPerBlock - (blockDim.x - 1)*part : part;






		for (int i = 0; i < partPerThread; i++)
		{

			temp[blockLocalThreadId*partPerThread + i] = totalItems[blockId*batchSize + blockLocalThreadId * partPerThread + i];
			//printf("i %d val %d threadId %d\n", i, temp[blockLocalThreadId*partPerThread + i], blockLocalThreadId);
		}

		__syncthreads();


		if (blockLocalThreadId == 0)
		{
			//process = true;
			process1 = true;
			process2 = true;
			//isLeft = true;
		}

		__syncthreads();

		for (int l = 2; l <= batchSize; l <<= 1) {
			for (int j = l >> 1; j > 0; j = j >> 1)
			{
				int part1, part2;
				//part1 = blockLocalThreadId;
				int index = blockLocalThreadId * (part);
				int start = index;
				while (start != (index + partPerThread))
				{
					part2 = start ^ j;

					if ((part2) > start) {
						if ((start&l) == 0) {
							if (temp[start] > temp[part2]) {
								int temporary = temp[start];
								temp[start] = temp[part2];
								temp[part2] = temporary;
							}
						}
						if ((start&l) != 0) {
							if (temp[start] < temp[part2]) {
								int temporary = temp[start];
								temp[start] = temp[part2];
								temp[part2] = temporary;
							}
						}
					}
					start++;
				}



			}
			__syncthreads();



		}



		if (blockLocalThreadId == 0)
		{
			//printf("blockId starts %d\n", blockId);
			process = true;
			while (atomicCAS(&d_heap->mutex[0], 0, 1) != 0);
			//printf("blockId goes ahead %d\n", blockId);
		}

		__syncthreads();

		int size = d_heap->partialBufSize + batchSize;
		int length = (size) / blockDim.x;
		int diagonalNum = (blockLocalThreadId)*(size) / blockDim.x;
	//	int* mergedArray = (int*)malloc(size * sizeof(int));
		struct points p = diagonalIntersection(d_heap->partialBuffer, temp, blockLocalThreadId, blockDim.x, d_heap->partialBufSize, batchSize, diagonalNum);
		int tempStart = (blockLocalThreadId)*(size / blockDim.x);
		mergeArrays(d_heap->partialBuffer, temp, p.x_start, p.y_start, mergedArray, tempStart, length, d_heap->partialBufSize, batchSize);

		__syncthreads();

		if (blockLocalThreadId == 0)
		{
			if (size >= k)
			{
				
				placeItemsCorrectly(d_heap->partialBuffer, temp, mergedArray, size);
				d_heap->partialBufSize = size - k;
			}
			else
			{
				placeItemsCorrectly(d_heap->partialBuffer, temp, mergedArray, size);
				if (d_heap->currentSize != 0)
				{
				//	mergedArray = (int*)malloc(size * sizeof(int));
					mergeArrays(d_heap->nodes[0].batch, d_heap->partialBuffer, 0, 0, mergedArray, 0, k + size, k, size);
					placeItemsCorrectly(d_heap->nodes[0].batch, d_heap->partialBuffer, mergedArray, k + size);
				}
				atomicCAS(&d_heap->mutex[0], 1, 0);
				process = false;

			}
		}
		__syncthreads();

		if (process)
		{
			if (blockLocalThreadId == 0)
			{
				cur = d_heap->currentSize++;
				par = ceilf((float)cur / 2) - 1;

				
				if (cur != 0)
				{
					
					while (atomicCAS(&d_heap->mutex[cur], 0, 1) != 0);
					atomicCAS(&d_heap->mutex[0], 1, 0);

					
				}

			
				/*for (int i = 0; i < batchSize; i++)
					printf("i %d temp %d\n ", i, temp[i]);*/

				
				copyKeys(temp,d_heap->nodes[cur].batch);
			}
			__syncthreads();

			while (cur != 0 && process1&&process2)
			{
				__syncthreads();

				if (blockLocalThreadId == 0)
				{
					
					atomicCAS(&d_heap->mutex[cur], 1, 2);//inuse->inshold

					pState = 1;
					cState = 1;

				

					while (pState == 1 || pState == 2)
					{
						//printf("blockId %d curr %d par %d pState %d\n", blockId, cur, par,d_heap->mutex[par]);
						pState = atomicCAS(&d_heap->mutex[par], 0, 1);
					}
					if (pState != 0)
						process1 = false;

					

				}
				__syncthreads();


				if (process1)
				{
					if (blockLocalThreadId == 0)
					{
					

						while (cState == 1)
						{
							cState=atomicCAS(&d_heap->mutex[cur],2,1);
						}
						
					//	printf("blockId %d car %d atomicCAS(&d_heap->mutex[cur] %d cState %d\n", blockId, cur, d_heap->mutex[cur], cState);

						if (cState == 3)
							atomicCAS(&d_heap->mutex[par], 3, 0);
						else
							if (cState == 0)
							{
								
								atomicCAS(&d_heap->mutex[par], 1, 0);
								process2 = false;
							}

						
					}

				}

				__syncthreads();

				if (process2)
				{
					if (blockLocalThreadId == 0)
					{
						if (cState != 3)
						{
							if (d_heap->nodes[cur].batch[0] >= d_heap->nodes[par].batch[k - 1])
							{
								atomicCAS(&d_heap->mutex[par], 1, 0);
								process = false;
							}
						}
					}

					__syncthreads();
					if (process)
					{


						int size = 2 * k;
						int length = (size) / blockDim.x;
						int diagonalNum = (blockLocalThreadId)*(size) / blockDim.x;
					//	int* mergedArray = (int*)malloc(size * sizeof(int));
						struct points p = diagonalIntersection(d_heap->nodes[par].batch, d_heap->nodes[cur].batch, blockLocalThreadId, blockDim.x, k, k, diagonalNum);
						int tempStart = (blockLocalThreadId)*(size / blockDim.x);
						mergeArrays(d_heap->nodes[par].batch, d_heap->nodes[cur].batch, p.x_start, p.y_start, mergedArray, tempStart, length, k, k);
						__syncthreads();

						if (blockLocalThreadId == 0)
						{
							placeItemsCorrectly(d_heap->nodes[cur].batch, d_heap->nodes[par].batch, mergedArray, size);
						}
						atomicCAS(&d_heap->mutex[cur], 1, 0);
					}
					if (blockLocalThreadId == 0)
					{
						cur = par;
						par = ceilf((float)cur / 2) - 1;
					}


				}

				__syncthreads();
			}


			if (blockLocalThreadId == 0)
			{
				//printf("blockId %d ends\n", blockId);
				atomicCAS(&d_heap->mutex[cur], 1, 0);
			}


		}
	}


	__global__ void deleteKeyV2(Heap* d_heap, int* deleteItems)
		{
			__shared__ int l;
			__shared__ int r;
			__shared__ bool process;
			__shared__ bool process1;
			__shared__ bool process2;
			__shared__ bool isLeft;
			__shared__ int tar;
			__shared__ int cur;
			__shared__ int lstate;
			__shared__ int rstate;
			__shared__ int mergedArray[3 * k];
			bool flag = false;


			int blockId = blockIdx.y*gridDim.x + blockIdx.x;
			int blockLocalThreadId = blockDim.x*threadIdx.y + threadIdx.x;
			int globalThreadId = blockId * blockDim.x * blockDim.y + blockLocalThreadId;


			if (blockLocalThreadId == 0)
			{
				process = true;
				process1 = true;
				process2 = true;
				isLeft = true;
				while (atomicCAS(&d_heap->mutex[0], 0, 1) != 0);
			}

			__syncthreads();

			if (blockLocalThreadId == 0)
			{

				if (d_heap->currentSize == 0)
				{
					if (d_heap->partialBufSize != 0)
					{
						for (int i = 0; i < d_heap->partialBufSize; i++)
						{
							deleteItems[blockId*k + i] = d_heap->partialBuffer[i];
						}
					}

					atomicCAS(&d_heap->mutex[0], 1, 0);
					process = false;
				}

			}
			__syncthreads();

			if (process)
			{
				__syncthreads();

				if (blockLocalThreadId == 0)
				{

					for (int i = 0; i < k; i++)
					{
						deleteItems[blockId*k + i] = d_heap->nodes[0].batch[i];
					}
					tar = --d_heap->currentSize;
					cur = 0;

					int tstate = atomicCAS(&d_heap->mutex[tar], 2, 3);
					if (tstate == 2)
						while (d_heap->mutex[tar] != 0);
					else
					{
						while (atomicCAS(&d_heap->mutex[tar], 0, 1) != 0);
						copyKeys(d_heap->nodes[tar].batch, d_heap->nodes[0].batch);
						atomicCAS(&d_heap->mutex[tar], 1, 0);
					}
				}

				__syncthreads();


				int size = d_heap->partialBufSize + k;
				int length = (size) / blockDim.x;
				int diagonalNum = (blockLocalThreadId)*(size) / blockDim.x;
				struct points p = diagonalIntersection(d_heap->partialBuffer, d_heap->nodes[0].batch, blockLocalThreadId, blockDim.x, d_heap->partialBufSize, k, diagonalNum);
				int tempStart = (blockLocalThreadId)*(size / blockDim.x);
				mergeArrays(d_heap->partialBuffer, d_heap->nodes[0].batch, p.x_start, p.y_start, mergedArray, tempStart, length, d_heap->partialBufSize, k);

				__syncthreads();

				if (blockLocalThreadId == 0)
				{
					placeItemsCorrectly(d_heap->partialBuffer, d_heap->nodes[0].batch, mergedArray, size);
				}

				__syncthreads();


				do
				{

					__syncthreads();


					if (blockLocalThreadId == 0)
					{
						l = 2 * cur + 1;
						r = 2 * cur + 2;
						lstate = 1;
						rstate = 1;
						
						while (lstate == 1)
							lstate = atomicCAS(&d_heap->mutex[l], 0, 1);
						if (lstate != 0 || l >= d_heap->currentSize)
						{
							atomicCAS(&d_heap->mutex[l], 1, 0);
							process1 = false;
						}

						//printf("blockIdl after %d tar %d l %d curr %d\n", blockId, tar, l, cur);
					}
					__syncthreads();

					if (process1)
					{
						__syncthreads();

						if (blockLocalThreadId == 0)
						{
							//printf("blockIdr before  %d tar %d mutextar %d r %d cur %d \n", blockId, tar, d_heap->mutex[r], r, cur);
							while (rstate == 1)
								rstate = atomicCAS(&d_heap->mutex[r], 0, 1);
							//printf("blockIdr after  %d tar %d mutextar %d r %d cur %d \n", blockId, tar, d_heap->mutex[r], r, cur);
						}
						__syncthreads();


						if (rstate != 0 || r >= d_heap->currentSize)
						{
							size = 2 * k;
							length = (size) / blockDim.x;
							diagonalNum = (blockLocalThreadId)*(size) / blockDim.x;
							p = diagonalIntersection(d_heap->nodes[cur].batch, d_heap->nodes[l].batch, blockLocalThreadId, blockDim.x, k, k, diagonalNum);
							tempStart = (blockLocalThreadId)*(size / blockDim.x);
							mergeArrays(d_heap->nodes[cur].batch, d_heap->nodes[l].batch, p.x_start, p.y_start, mergedArray, tempStart, length, k, k);
							flag = true;

						}
						__syncthreads();

						if (blockLocalThreadId == 0 && flag)
						{
							placeItemsCorrectly(d_heap->nodes[l].batch, d_heap->nodes[cur].batch, mergedArray, size);
							process2 = false;

						}
						__syncthreads();


						if (process2)
						{
							__syncthreads();


							size = 2 * k;
							length = (size) / blockDim.x;
							diagonalNum = (blockLocalThreadId)*(size) / blockDim.x;
							p = diagonalIntersection(d_heap->nodes[l].batch, d_heap->nodes[r].batch, blockLocalThreadId, blockDim.x, k, k, diagonalNum);
							tempStart = (blockLocalThreadId)*(size / blockDim.x);
							mergeArrays(d_heap->nodes[l].batch, d_heap->nodes[r].batch, p.x_start, p.y_start, mergedArray, tempStart, length, k, k);

							__syncthreads();

							if (blockLocalThreadId == 0)
							{

								if (d_heap->nodes[l].batch[k - 1] < d_heap->nodes[r].batch[k - 1])
								{
									placeItemsCorrectly(d_heap->nodes[r].batch, d_heap->nodes[l].batch, mergedArray, size);
									atomicCAS(&d_heap->mutex[r], 1, 0);
									isLeft = true;
									if (d_heap->nodes[cur].batch[k - 1] <= d_heap->nodes[l].batch[0])
										process = false;
								}
								else
								{
									placeItemsCorrectly(d_heap->nodes[l].batch, d_heap->nodes[r].batch, mergedArray, size);
									atomicCAS(&d_heap->mutex[l], 1, 0);
									isLeft = false;
									if (d_heap->nodes[cur].batch[k - 1] <= d_heap->nodes[r].batch[0])
										process = false;
								}
							}
							__syncthreads();


							if (process)
							{
								__syncthreads();

								size = 2 * k;
								length = (size) / blockDim.x;
								diagonalNum = (blockLocalThreadId)*(size) / blockDim.x;
								if (isLeft)
								{
									p = diagonalIntersection(d_heap->nodes[cur].batch, d_heap->nodes[l].batch, blockLocalThreadId, blockDim.x, k, k, diagonalNum);
									tempStart = (blockLocalThreadId)*(size / blockDim.x);
									mergeArrays(d_heap->nodes[cur].batch, d_heap->nodes[l].batch, p.x_start, p.y_start, mergedArray, tempStart, length, k, k);
								}
								else
								{
									p = diagonalIntersection(d_heap->nodes[cur].batch, d_heap->nodes[r].batch, blockLocalThreadId, blockDim.x, k, k, diagonalNum);
									tempStart = (blockLocalThreadId)*(size / blockDim.x);
									mergeArrays(d_heap->nodes[cur].batch, d_heap->nodes[r].batch, p.x_start, p.y_start, mergedArray, tempStart, length, k, k);
								}


							}

							__syncthreads();

							if (blockLocalThreadId == 0 && process)
							{
								if (isLeft)
								{
									placeItemsCorrectly(d_heap->nodes[l].batch, d_heap->nodes[cur].batch, mergedArray, size);
									atomicCAS(&d_heap->mutex[cur], 1, 0);
									cur = l;
								}
								else
								{
									placeItemsCorrectly(d_heap->nodes[r].batch, d_heap->nodes[cur].batch, mergedArray, size);
									atomicCAS(&d_heap->mutex[cur], 1, 0);
									cur = r;
								}


							}




						}


					}

					__syncthreads();

				} while (process&&process1&&process2);


				__syncthreads();

				if (!process2 || !process)
				{
					if (blockLocalThreadId == 0)
					{
						if (isLeft || !process2)
						{
                           atomicCAS(&d_heap->mutex[l], 1, 0);
						}
						else
						{

							atomicCAS(&d_heap->mutex[r], 1, 0);

						}
					}

				}


				if (blockLocalThreadId == 0)
				{
					//atomicExch(&d_heap->mutex[cur], 0);
					atomicCAS(&d_heap->mutex[cur], 1, 0);
				}

				__syncthreads();
			}


		}


