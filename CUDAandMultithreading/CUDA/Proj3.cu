#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <iostream>
#include <ctype.h>
#include <cuda.h>
#include "math.h"

#define	CEIL(a,b)		((a+b-1)/b)
#define SWAP(a,b,t)		t=b; b=a; a=t;
#define DATAMB(bytes)			(bytes/1024/1024)
#define DATABW(bytes,timems)	((float)bytes/(timems * 1.024*1024.0*1024.0))

typedef unsigned char uch;
typedef unsigned long ul;
typedef unsigned int  ui;

uch *TheImg, *CopyImg, *TheImg2, *CopyImg2;					// Where images are stored in CPU
uch *GPUImg, *GPUCopyImg, *GPUResult, *GPUImg2, *GPUCopyImg2, *GPUResult2;	// Where images are stored in GPU

struct ImgProp{
	int Hpixels;
	int Vpixels;
	uch HeaderInfo[54];
	ul Hbytes;
}ip;

struct birdProp{
	int Hpixels;
	int Vpixels;
	uch HeaderInfo[54];
	ul Hbytes;
}bird;

#define	IPHB		ip.Hbytes
#define	IPH			ip.Hpixels
#define	IPV			ip.Vpixels
#define	IPHB2		bird.Hbytes
#define	IPH2		bird.Hpixels
#define	IPV2		bird.Vpixels
#define	IMAGESIZE	(IPHB*IPV)
#define IMAGESIZE2  (IPHB2*IPV2)
#define	IMAGEPIX	(IPH*IPV)
#define IMAGEPIX2   (IPH2*IPV2)

//Puts our Bird on to the location marked by the X and Y position from the Command Line
__global__
void Operation1(uch *ImgDst, uch *MainImage, uch *ImageToBeMixed, ui Hpixels, ui RowBytes,ui xPosition, ui yPosition, ui ImageToBeMixedHpixels, ui ImageToBeMixedVpixels, ui RowBytes2){
	ui ThrPerBlk = blockDim.x;
	ui MYbid = blockIdx.x;
	ui MYtid = threadIdx.x;
	ui MYrow = blockIdx.y;
	ui MYcol = MYbid*ThrPerBlk + MYtid;
	if (MYcol >= Hpixels) return;			// col out of range
	ui MYoffset = MYrow * RowBytes;
	ui MYsrcIndex = MYoffset + 3 * MYcol;
    if((MYcol > xPosition && MYcol < xPosition + ImageToBeMixedHpixels) && (MYrow > yPosition && MYrow < yPosition + ImageToBeMixedVpixels)){ //You are withing the range of the Box Placement Destination
        ui baseX = (MYrow - yPosition) * RowBytes2 ;
        ui baseY = (MYcol - xPosition) * 3; //These two variables normalize the x and y terms (You can't just access the bird image using MYsrcindex, you have to translate your current 
        //position to something small enough to translation to the bird image
        ui translatedBirdCoord = baseX + baseY;
        uch B = ImageToBeMixed[translatedBirdCoord];
        uch G = ImageToBeMixed[translatedBirdCoord+1];
        uch R = ImageToBeMixed[translatedBirdCoord+2];
        if((B > 0xC8) && (G > 0xC8) && (R > 0xC8)){ //I know white is 255 all BGR but lets make it 200 just in case the white background isn't exactly white.
            ImgDst[MYsrcIndex] = MainImage[MYsrcIndex];
            ImgDst[MYsrcIndex+1] = MainImage[MYsrcIndex+1]; 
            ImgDst[MYsrcIndex+2] = MainImage[MYsrcIndex+2]; 
        }
        else{
            ImgDst[MYsrcIndex] = ImageToBeMixed[translatedBirdCoord];
            ImgDst[MYsrcIndex+1] = ImageToBeMixed[translatedBirdCoord+1]; 
            ImgDst[MYsrcIndex+2] = ImageToBeMixed[translatedBirdCoord+2]; 
        }
    }
    else{
        ImgDst[MYsrcIndex] = MainImage[MYsrcIndex];
        ImgDst[MYsrcIndex+1] = MainImage[MYsrcIndex+1]; 
        ImgDst[MYsrcIndex+2] = MainImage[MYsrcIndex+2];         
    }
}
//Creates Regions where the bird is suspected to be at
__global__
void Operation2(uch *ImgDst, uch *MainImage, ui Hpixels, ui RowBytes, ui Vpixels){
	ui ThrPerBlk = blockDim.x;
	ui MYbid = blockIdx.x;
	ui MYtid = threadIdx.x;
	ui MYrow = blockIdx.y;
	ui MYcol = MYbid*ThrPerBlk + MYtid;
	int i, remainderX,remainderY, j;
	remainderX = (MYcol*16 + 15) - Hpixels;
	remainderY = (MYrow*16 +15) - Vpixels;
	if (((MYcol)*16) >= Hpixels) return;			// col out of range
	if(((MYrow)*16 >= Vpixels)) return;
	if((remainderX > 0 ) || ( remainderY > 0)){ //Basically if an uneven block, just fill the extra white space with with white pixels. 
		if(remainderX <= 0){ //This isn't what triggered the edge case, would be negative and we don't want that
			remainderX = 16;
		}
		else{
			remainderX = 16-remainderX;
		}
		if(remainderY <= 0){ //This isn't what triggered the edge case, would be negative and we don't want that
			remainderY = 16;
		}
		else{
			remainderY = 16-remainderY;
		}
		MYrow = MYrow*16;
		for(i = 0; i < remainderY; i++){ //fill out the x edges
			ui MYoffset = (MYrow+i) * RowBytes;
			for(j = 0; j < remainderX; j++){ //fill out the y edges
				ui MYsrcIndex = MYoffset + 3 * (16*MYcol +j);
				ImgDst[MYsrcIndex] = 0xFF;
        		ImgDst[MYsrcIndex+1] = 0xFF; 
        		ImgDst[MYsrcIndex+2] = 0xFF;
			}
		}
		return;
	}
	//ui MYoffset = MYrow * RowBytes;
	//ui MYsrcIndex = MYoffset + 3 * MYcol;
	ui NumberOfRedPixels = 0;
	ui oldMYrow = MYrow;
	MYrow = MYrow*16;
    for(i = 0; i < 16; i++){ //fill out the x edges
		ui MYoffset = (MYrow+i) * RowBytes;
		for(j = 0; j < 16; j++){ //fill out the y edges
			ui MYsrcIndex = MYoffset + 3 * (16*MYcol +j);
			if ((MainImage[MYsrcIndex] <= 0x40) && (MainImage[MYsrcIndex+1] <= 0x40) && (MainImage[MYsrcIndex+2] >= 0xC8)){ //BRG- if B and G are 0 and R > = 240, then this is a red pixel
				NumberOfRedPixels++;
				ImgDst[MYsrcIndex] = 0x00;
        		ImgDst[MYsrcIndex+1] = 0x00; 
        		ImgDst[MYsrcIndex+2] = 0xFF;
			}
		}
	}
	oldMYrow = oldMYrow *16;
	if(NumberOfRedPixels >= 160){ //If the number of red pixels are over 62.5%
		for(i = 0; i < 16; i++){ //fill out the x edges
		ui MYoffset = (oldMYrow+i) * RowBytes;
			for(j = 0; j < 16; j++){ //fill out the y edges
				ui MYsrcIndex = MYoffset + 3 * (16*MYcol +j);
				ImgDst[MYsrcIndex] = 0x00;
        		ImgDst[MYsrcIndex+1] = 0x00; 
        		ImgDst[MYsrcIndex+2] = 0xFF;
			}
		}
	}
	else{
		for(i = 0; i < 16; i++){ //fill out the x edges
		ui MYoffset = (oldMYrow+i) * RowBytes;
			for(j = 0; j < 16; j++){ //fill out the y edges
				ui MYsrcIndex = MYoffset + 3 * (16*MYcol +j);
				ImgDst[MYsrcIndex] = 0xFF;
        		ImgDst[MYsrcIndex+1] = 0xFF; 
        		ImgDst[MYsrcIndex+2] = 0xFF;
			}
		}
	} 
}

// Read a 24-bit/pixel BMP file into a 1D linear array.
// Allocate memory to store the 1D image and return its pointer.
uch *ReadBMPlin(char* fn)
{
	static uch *Img;
	FILE* f = fopen(fn, "rb");
	if (f == NULL){	printf("\n\n%s NOT FOUND\n\n", fn);	exit(EXIT_FAILURE); }
	uch HeaderInfo[54];
	fread(HeaderInfo, sizeof(uch), 54, f); // read the 54-byte header
	// extract image height and width from header
	int width = *(int*)&HeaderInfo[18];			ip.Hpixels = width;
	int height = *(int*)&HeaderInfo[22];		ip.Vpixels = height;
	int RowBytes = (width * 3 + 3) & (~3);		ip.Hbytes = RowBytes;
	//save header for re-use
	memcpy(ip.HeaderInfo, HeaderInfo,54);
	printf("\n Input File name: %17s  (%u x %u)   File Size=%u\n", fn, 
			ip.Hpixels, ip.Vpixels, IMAGESIZE);
	// allocate memory to store the main image (1 Dimensional array)
	Img  = (uch *)malloc(IMAGESIZE);
	if (Img == NULL) return Img;      // Cannot allocate memory
	// read the image from disk
	fread(Img, sizeof(uch), IMAGESIZE, f);
	fclose(f);
	return Img;
}

// Read a 24-bit/pixel BMP file into a 1D linear array.
// Allocate memory to store the 1D image and return its pointer.
uch *ReadBMPlin2(char* fn)
{
	static uch *Img2;
	FILE* f = fopen(fn, "rb");
	if (f == NULL){	printf("\n\n%s NOT FOUND\n\n", fn);	exit(EXIT_FAILURE); }

	uch HeaderInfo[54];
	fread(HeaderInfo, sizeof(uch), 54, f); // read the 54-byte header
	// extract image height and width from header
	int width = *(int*)&HeaderInfo[18];			bird.Hpixels = width;
	int height = *(int*)&HeaderInfo[22];		bird.Vpixels = height;
	int RowBytes = (width * 3 + 3) & (~3);		bird.Hbytes = RowBytes;
	//save header for re-use
	memcpy(bird.HeaderInfo, HeaderInfo,54);
	printf("\n Input File name: %17s  (%u x %u)   File Size=%u", fn, 
		    bird.Hpixels, bird.Vpixels, IMAGESIZE2);
	// allocate memory to store the main image (1 Dimensional array)
	Img2  = (uch *)malloc(IMAGESIZE2);
	if (Img2 == NULL) return Img2;      // Cannot allocate memory
	// read the image from disk
	fread(Img2, sizeof(uch), IMAGESIZE2, f);
	fclose(f);
	return Img2;
}

// Write the 1D linear-memory stored image into file.
void WriteBMPlin(uch *Img, char* fn)
{
	FILE* f = fopen(fn, "wb");
	if (f == NULL){ printf("\n\nFILE CREATION ERROR: %s\n\n", fn); exit(1); }
	//write header
	fwrite(ip.HeaderInfo, sizeof(uch), 54, f);
	//write data
	fwrite(Img, sizeof(uch), IMAGESIZE, f);
	printf("\nOutput File name: %17s  (%u x %u)   File Size=%u", fn, ip.Hpixels, ip.Vpixels, IMAGESIZE);
	fclose(f);
}


int main(int argc, char **argv)
{
	float			totalTime, tfrCPUtoGPU, tfrGPUtoCPU, kernelExecutionTime; // GPU code run times
	cudaError_t		cudaStatus, cudaStatus2,cudaStatus3;
	cudaEvent_t		time1, time2, time3, time4;
	char			InputFileName[255], OutputFileName[255], ProgName[255], InputFile2Name[255];
	ui				BlkPerRow; //BlkPerRowInt, BlkPerRowInt2;
	ui				ThrPerBlk = 256, NumBlocks, xPosition, yPosition;
	//ui			    NB2, NB4, NB8;
	ui 				GPUDataTransfer;
	ui				RowBytes, RowBytes2;
	cudaDeviceProp	GPUprop;
	ul				SupportedKBlocks, SupportedMBlocks, MaxThrPerBlk;
	//ui				*GPUCopyImg32, *GPUImg32, *GPUImg322;
	char			SupportedBlocks[100];
	//int				KernelNum=1;
	char			KernelName[255];
	int 			Operation; //Start it as the base operation
	strcpy(ProgName, "Proj3");
	if(argc > 4 && atoi(argv[3]) != 1){
		printf("Invalid arguments! Must be in order of:\n");
		printf("\n\nUsage:   %s InputFilename OutputFilename 1 redbird.bmp BaseXCoordinate BaseYCoordinate\n", ProgName);
		printf("\n\nUsage:   %s InputFilename OutputFilename 2\n", ProgName);
		exit(EXIT_FAILURE);
	}
	switch (argc){
    case 7:  xPosition = atoi(argv[5]);
             yPosition = atoi(argv[6]);
	case 5:  strcpy(InputFile2Name,argv[4]);
	case 4:  Operation = atoi(argv[3]);
	case 3:  strcpy(InputFileName, argv[1]);
			 strcpy(OutputFileName, argv[2]);
			 break;
	default: printf("\n\nUsage:   %s InputFilename OutputFilename 1 redbird<L,96>.bmp BaseXCoordinate BaseYCoordinate\n", ProgName);
			 printf("\n\nUsage:   %s InputFilename OutputFilename 2\n", ProgName);
			 exit(EXIT_FAILURE);
	}
	//if(Operation != 1){ 
	if ((Operation < 1) || (Operation > 2)) {
		printf("Invalid Operation Option '%d' . Must be 1 or 2\n",Operation); //printf("Invalid Operation Option '%d'. Must be 1,2 or 3 ... \n", Operation);
		exit(EXIT_FAILURE);
	}
	// Create CPU memory to store the input and output images
	TheImg = ReadBMPlin(InputFileName); // Read the input image if memory can be allocated
	if(Operation == 1){
    	TheImg2 = ReadBMPlin2(InputFile2Name);
	}
	if (TheImg == NULL){
		printf("Cannot allocate memory for the input image...\n");
		exit(EXIT_FAILURE);
	}
	if(Operation == 1){
   		if(TheImg2 == NULL){
        	printf("Cannot allocate memory for the second input image..\n");
    	}
	}
	CopyImg = (uch *)malloc(IMAGESIZE);
	if(Operation == 1){
    	CopyImg2 = (uch *)malloc(IMAGESIZE2);
	}
	if (CopyImg == NULL){
		free(TheImg);
		printf("Cannot allocate memory for the input image...\n");
		exit(EXIT_FAILURE);
	}
	if(Operation == 1){
		if (CopyImg2 == NULL){
			free(TheImg2);
			printf("Cannot allocate memory for the second input image...\n");
			exit(EXIT_FAILURE);
		}
		if((xPosition+bird.Hpixels+16 > ip.Hpixels) || (yPosition+bird.Vpixels+16 > ip.Vpixels)){
			printf("\nBird and Box Around Bird won't fit in the image! Please keep it inside the image enough to make the box fit!\n");
			exit(EXIT_FAILURE);
		}
	}
	// Choose which GPU to run on, change this on a multi-GPU system.
	int NumGPUs = 0;
	cudaGetDeviceCount(&NumGPUs);
	if (NumGPUs == 0){
		printf("\nNo CUDA Device is available\n");
		exit(EXIT_FAILURE);
	}
	cudaStatus = cudaSetDevice(0);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?");
		exit(EXIT_FAILURE);
	}
	cudaGetDeviceProperties(&GPUprop, 0);
	SupportedKBlocks = (ui)GPUprop.maxGridSize[0] * (ui)GPUprop.maxGridSize[1] * (ui)GPUprop.maxGridSize[2] / 1024;
	SupportedMBlocks = SupportedKBlocks / 1024;
	sprintf(SupportedBlocks, "%u %c", (SupportedMBlocks >= 5) ? SupportedMBlocks : SupportedKBlocks, (SupportedMBlocks >= 5) ? 'M' : 'K');
	MaxThrPerBlk = (ui)GPUprop.maxThreadsPerBlock;

	cudaEventCreate(&time1);
	cudaEventCreate(&time2);
	cudaEventCreate(&time3);
	cudaEventCreate(&time4);

	cudaEventRecord(time1, 0);		// Time stamp at the start of the GPU transfer
	// Allocate GPU buffer for the input and output images
	cudaStatus = cudaMalloc((void**)&GPUImg, IMAGESIZE);
	cudaStatus2 = cudaMalloc((void**)&GPUCopyImg, IMAGESIZE);
	if(Operation == 1){
    	cudaStatus3 = cudaMalloc((void**)&GPUImg2, IMAGESIZE2);
	}
	if ((cudaStatus != cudaSuccess) || (cudaStatus2 != cudaSuccess)){
		fprintf(stderr, "cudaMalloc failed! Can't allocate GPU memory");
		exit(EXIT_FAILURE);
	}
	if(Operation == 1){
		if((cudaStatus3 != cudaSuccess)){
			fprintf(stderr, "cudaMalloc failed! Can't allocate GPU memory");
			exit(EXIT_FAILURE);
		}
	}
	// These are the same pointers as GPUCopyImg and GPUImg, however, casted to an integer pointer
	/*
	GPUCopyImg32 = (ui *)GPUCopyImg;
	GPUImg32 = (ui *)GPUImg; //Our Astronaut.bmp
	if(Operation == 1){
	   	GPUImg322 = (ui *)GPUImg2; //Our redBird.bmp
	}
	*/
	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(GPUImg, TheImg, IMAGESIZE, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy  CPU to GPU  failed!");
		exit(EXIT_FAILURE);
	}
	if(Operation == 1){
		cudaStatus3 = cudaMemcpy(GPUImg2, TheImg2, IMAGESIZE2, cudaMemcpyHostToDevice);
		if (cudaStatus3 != cudaSuccess) {
			fprintf(stderr, "cudaMemcpy  CPU to GPU  failed for the Second Image!");
			exit(EXIT_FAILURE);
		}
	}
	cudaEventRecord(time2, 0);		// Time stamp after the CPU --> GPU tfr is done
	
	RowBytes = (IPH * 3 + 3) & (~3);
    RowBytes2 = (IPH2 * 3 + 3) & (~3);
	//RowInts = RowBytes / 4;
	BlkPerRow = CEIL(IPH,ThrPerBlk);
	NumBlocks = IPV*BlkPerRow;
	dim3 dimGrid2D(BlkPerRow,		   ip.Vpixels);
	dim3 dimGrid2D8(CEIL(BlkPerRow,16), CEIL(ip.Vpixels,16));
	switch (Operation){
		case 1: 
			Operation1 <<< dimGrid2D, ThrPerBlk >>> (GPUCopyImg, GPUImg, GPUImg2, IPH, RowBytes,xPosition,yPosition,IPH2,IPV2,RowBytes2);
			strcpy(KernelName,"Operation 1 : Just Place the Bird on to an Image");
			GPUResult = GPUCopyImg;
			GPUDataTransfer = 2*IMAGESIZE + IMAGESIZE2;
			break;
        
		case 2:
			Operation2 <<< dimGrid2D8, ThrPerBlk >>> (GPUCopyImg, GPUImg, IPH,RowBytes,IPV);
			strcpy(KernelName,"Operation 2 : Determine Supsected Object Regions");
			GPUResult = GPUCopyImg;
			GPUDataTransfer = 2*IMAGESIZE;
			break;
		/*
		case 3:
			Operation3A <<< dimGrid2DOp3NotDivBy4A, ThrPerBlk >>> (GPUCopyImg32, GPUImg32, IPH, RowBytes,CenterPixelX,CenterPixelY,longestLineMultiplier,RowInts);
			strcpy(KernelName,"Operation 3 : Each thread will brighten or darken 1 pixel (using a 2D grid and more improved from Operation 1!)");
			GPUResult = GPUCopyImg;
			GPUDataTransfer = 2*IMAGESIZE;
			break;
        */
		default:
			printf("...... Operation Number ... NOT IMPLEMENTED .... \n");
			strcpy(KernelName,"*** NOT IMPLEMENTED ***");
			GPUResult = GPUCopyImg;
			GPUDataTransfer = 2 * IMAGESIZE;
			break;
		}
	// cudaDeviceSynchronize waits for the kernel to finish, and returns
	// any errors encountered during the launch.
	cudaStatus = cudaDeviceSynchronize();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "\n\ncudaDeviceSynchronize returned error code %d after launching the kernel!\n", cudaStatus);
		exit(EXIT_FAILURE);
	}
	cudaEventRecord(time3, 0);
	// Copy output (results) from GPU buffer to host (CPU) memory.
	cudaStatus = cudaMemcpy(CopyImg, GPUResult, IMAGESIZE, cudaMemcpyDeviceToHost);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy GPU to CPU  failed!");
		exit(EXIT_FAILURE);
	}
	cudaEventRecord(time4, 0);

	cudaEventSynchronize(time1);
	cudaEventSynchronize(time2);
	cudaEventSynchronize(time3);
	cudaEventSynchronize(time4);

	cudaEventElapsedTime(&totalTime, time1, time4);
	cudaEventElapsedTime(&tfrCPUtoGPU, time1, time2);
	cudaEventElapsedTime(&kernelExecutionTime, time2, time3);
	cudaEventElapsedTime(&tfrGPUtoCPU, time3, time4);

	cudaStatus = cudaDeviceSynchronize();
	//checkError(cudaGetLastError());	// screen for errors in kernel launches
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "\n Program failed after cudaDeviceSynchronize()!");
		free(TheImg);
		free(CopyImg);
		exit(EXIT_FAILURE);
	}
	WriteBMPlin(CopyImg, OutputFileName);		// Write the flipped image back to disk
	printf("\n--------------------------------------------------------------------------\n");
	printf("%s    ComputeCapab=%d.%d  [max %s blocks; %d thr/blk] \n",
		GPUprop.name, GPUprop.major, GPUprop.minor, SupportedBlocks, MaxThrPerBlk);
	printf("--------------------------------------------------------------------------\n");
	printf("%s %s %s Operation: %d %u  [%u BLOCKS, %u BLOCKS/ROW]\n", ProgName, InputFileName, OutputFileName, Operation, ThrPerBlk, NumBlocks, BlkPerRow);
	printf("--------------------------------------------------------------------------\n");
	printf("%s\n",KernelName);
	printf("--------------------------------------------------------------------------\n");
	printf("CPU->GPU Transfer   =%7.2f ms  ...  %4d MB  ...  %6.2f GB/s\n", tfrCPUtoGPU, DATAMB(IMAGESIZE), DATABW(IMAGESIZE, tfrCPUtoGPU));
	printf("Kernel Execution    =%7.2f ms  ...  %4d MB  ...  %6.2f GB/s\n", kernelExecutionTime, DATAMB(GPUDataTransfer), DATABW(GPUDataTransfer, kernelExecutionTime));
	printf("GPU->CPU Transfer   =%7.2f ms  ...  %4d MB  ...  %6.2f GB/s\n", tfrGPUtoCPU, DATAMB(IMAGESIZE), DATABW(IMAGESIZE, tfrGPUtoCPU)); 
	printf("--------------------------------------------------------------------------\n");
	printf("Total time elapsed  =%7.2f ms       %4d MB  ...  %6.2f GB/s\n", totalTime, DATAMB((2*IMAGESIZE+GPUDataTransfer)), DATABW((2 * IMAGESIZE + GPUDataTransfer), totalTime));
	printf("--------------------------------------------------------------------------\n\n");

	// Deallocate CPU, GPU memory and destroy events.
	cudaFree(GPUImg);
	cudaFree(GPUCopyImg);
	cudaEventDestroy(time1);
	cudaEventDestroy(time2);
	cudaEventDestroy(time3);
	cudaEventDestroy(time4);
	// cudaDeviceReset must be called before exiting in order for profiling and
	// tracing tools such as Parallel Nsight and Visual Profiler to show complete traces.
	cudaStatus = cudaDeviceReset();
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaDeviceReset failed!");
		free(TheImg);
		free(CopyImg);
		exit(EXIT_FAILURE);
	}
	free(TheImg);
	free(CopyImg);
	return(EXIT_SUCCESS);
}




