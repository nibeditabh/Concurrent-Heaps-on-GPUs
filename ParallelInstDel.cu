#include "ParallelOp.cu"




__device__  int counter = 0;
__device__ int dCounter = 0;


/* Concurrent insertion/deletion into the heap. */

__global__ void insert_Delete(Heap* d_heap, int* totalItems, int itemCount, int batchSize, int* deleteItems, bool* Task)
{
	__shared__ bool isInsert;

	int blockId = blockIdx.y*gridDim.x + blockIdx.x;
	int blockLocalThreadId = blockDim.x*threadIdx.y + threadIdx.x;
	int globalThreadId = blockId * blockDim.x * blockDim.y + blockLocalThreadId;


	if (blockLocalThreadId == 0)
	{
		isInsert = Task[blockId];

	}


	__syncthreads();

	/* Block-Level insert operation performed by the block allocated with an insertion task. */

	if (isInsert)
	{
		
		__shared__ int temp[10000];
		__shared__ bool process;
		__shared__ int tar;
		__shared__ int cur;
		__shared__ int level;
		__shared__ int mergedArray[3 * k];
		__shared__ int itemIndex;

		int blockId = blockIdx.y*gridDim.x + blockIdx.x;
		int blockLocalThreadId = blockDim.x*threadIdx.y + threadIdx.x;
		int globalThreadId = blockId * blockDim.x * blockDim.y + blockLocalThreadId;
		int partPerBlock = batchSize;
		int part = partPerBlock / blockDim.x;
		int partPerThread = (blockLocalThreadId == blockDim.x - 1) ? partPerBlock - (blockDim.x - 1)*part : part;


		if (blockLocalThreadId == 0)
		{
			itemIndex = atomicAdd(&counter, batchSize);
			
		}


		__syncthreads();

		for (int i = 0; i < partPerThread; i++)
		{

			temp[blockLocalThreadId*partPerThread + i] = totalItems[itemIndex + blockLocalThreadId * partPerThread + i];

		}



		__syncthreads();



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



		__syncthreads();

		if (blockLocalThreadId == 0)
		{

			process = true;
			while (atomicCAS(&d_heap->mutex[0], 0, 1) != 0);
			



		}


		__syncthreads();




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
				tar = atomicAdd(&d_heap->currentSize, 1);
				cur = 0;
				level = __log2f(d_heap->currentSize);

				/* Set the target node's field to TARGET(2) to identified by deletion
		          operation that it is Target node of some insertion operation.*/

				atomicExch(&d_heap->nodes[tar].target, 2); 
				if (tar != 0)
				{
					
					while (atomicCAS(&d_heap->mutex[tar], 0, 1) != 0);
			
				}


			}
			__syncthreads();

			while (cur != tar)
			{
				__syncthreads();


				/* Break from loop, if the target node's field is set as MARKED(3) by a deletion operation.*/
				if (d_heap->nodes[tar].target == 3)
				{
					//printf("blockId-MARK %d tar %d\n", blockId,tar);
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
					
					cur = ((tar + 1) >> --level) - 1;
					//printf("BlockId enterssssss %d level %d tar %d cur %d\n", blockId, level, tar,cur);
					if (cur != tar)
					{
						while (atomicCAS(&d_heap->mutex[cur], 0, 1) != 0);
					}

					int par = ceilf((float)cur / 2) - 1;
				//	printf("blockId %d par %d tar %d curr %d\n", blockId, par, tar,cur);
					atomicCAS(&d_heap->mutex[par], 1, 0);
				}
				__syncthreads();
			}

			if (blockLocalThreadId == 0)
			{
				int tstate = atomicCAS(&d_heap->nodes[tar].target, 2, 1);
				int oldTar = tar;

				/* If the target node is not MARKED(3), 
				insert elements to target node or else, insert elements to root node. */

				tar = tstate == 2 ? tar : 0; 
				for (int i = 0; i < k; i++)
				{
					d_heap->nodes[tar].batch[i] = temp[i];
					
				}


				if (tar != cur)
				{
					atomicCAS(&d_heap->mutex[oldTar], 1, 0);
				}
			//printf("end for blockId %d curr %d tar %d oldTar %d mutexold %d\n", blockId,cur,tar,oldTar, d_heap->mutex[oldTar]);
				atomicExch(&d_heap->nodes[oldTar].target, 0);
				atomicCAS(&d_heap->mutex[cur],1,0);



			}

		}



	}
	else /* Block-Level deletion operation performed by the block allocated with an deletion task. */
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
		__shared__ int dIndex;
		bool flag = false;


		

		if (blockLocalThreadId == 0)
		{
			process = true;
			process1 = true;
			process2 = true;
			isLeft = true;
			dIndex = atomicAdd(&dCounter, 1);


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
						deleteItems[dIndex*k + i] = d_heap->partialBuffer[i];
					}
				}
				else
				{
					deleteItems[dIndex*k] = -1;  /* Indicator that the deletion operation 
					                                 was performed on an empty heap */
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
					deleteItems[dIndex*k + i] = d_heap->nodes[0].batch[i];
				}
				int oldV = atomicAdd(&d_heap->currentSize, -1);
				tar = oldV - 1;
				cur = 0;

				int tstate = atomicCAS(&d_heap->nodes[tar].target, 2, 3);

				/* If the target node is the target node for a insertion operation, wait till target node is unlocked. */

				if (tstate == 2)
				{
					
					while (d_heap->mutex[tar] != 0);
				}
				else
				{
				
					if (tar != 0)
					{

						while (atomicCAS(&d_heap->mutex[tar], 0, 1) != 0);
						copyKeys(d_heap->nodes[tar].batch, d_heap->nodes[0].batch);
						atomicCAS(&d_heap->mutex[tar], 1, 0);
					}

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

					/* Check for leftchild node presence, and acquire a lock if present. */

					if ((l >= d_heap->currentSize)||(l>=tar)||(d_heap->nodes[l].target==2)/*||(tar>= d_heap->currentSize)*/)
					{
						atomicCAS(&d_heap->mutex[cur], 1, 0);
						process1 = false;
					}
					else
					{
						while (lstate == 1)
							lstate = atomicCAS(&d_heap->mutex[l], 0, 1);
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
				

						/* Check for leftchild node presence, and acquire a lock if present. */

						if ((r >= d_heap->currentSize)||(r>=tar)|| (d_heap->nodes[r].target == 2)/*||(tar>= d_heap->currentSize)*/)
						{
							process2 = false;
							rstate = 0;

						}
						else
						{
							while (rstate == 1)
								rstate = atomicCAS(&d_heap->mutex[r], 0, 1);
							
						}
					

					}
					__syncthreads();


					if (rstate != 0 )
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
								/* If current node in heapified state already, return. */
								if (d_heap->nodes[cur].batch[k - 1] <= d_heap->nodes[l].batch[0])
									process = false;
							}
							else
							{
								placeItemsCorrectly(d_heap->nodes[l].batch, d_heap->nodes[r].batch, mergedArray, size);
								atomicCAS(&d_heap->mutex[l], 1, 0);
								isLeft = false;
								/* If current node in heapified state already, return. */
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

			if (!process || !process2)
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
				if (!process || !process2)
					atomicCAS(&d_heap->mutex[cur], 1, 0);
			}

			__syncthreads();
		}



	}

}








