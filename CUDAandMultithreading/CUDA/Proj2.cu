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

uch *TheImg, *CopyImg;					// Where images are stored in CPU
uch *GPUImg, *GPUCopyImg, *GPUResult;	// Where images are stored in GPU

struct ImgProp{
	int Hpixels;
	int Vpixels;
	uch HeaderInfo[54];
	ul Hbytes;
} ip;

#define	IPHB		ip.Hbytes
#define	IPH			ip.Hpixels
#define	IPV			ip.Vpixels
#define	IMAGESIZE	(IPHB*IPV)
#define	IMAGEPIX	(IPH*IPV)

// Naive Operation1 Kernel, Performs a basic calculation of a the length of a line 
// from the center of the image to the pixel we are currently working on using libraries and functions. 
// From there, compares this line to the length of the biggest line to see how big it is compared to it in a %.
// Based on how big it is (Say line is length 12 and our line is length 9 so our line is 75% of of the biggest line)
// it will multiply this percentage or fraction to 2.0-0.5 = 1.5 to determine how much of this multiplier should
// we take so in this case 0.75 * 1.5 = 1.125 -> 1.125+0.5 = 1.625 is our multiplier. From there, we mulitply the RGB Pixel by 
// this multiplier to get the final pixel brightness 
__global__
void Operation1(uch *ImgDst, uch *ImgSrc, ui Hpixels, ui RowBytes,ui CenterPixelX,ui CenterPixelY,float longestLine){
	ui ThrPerBlk = blockDim.x;
	ui MYbid = blockIdx.x;
	ui MYtid = threadIdx.x;
	ui MYrow = blockIdx.y;
	ui MYcol = MYbid*ThrPerBlk + MYtid;
	if (MYcol >= Hpixels) return;			// col out of range
	ui MYoffset = MYrow * RowBytes;
	ui MYsrcIndex = MYoffset + 3 * MYcol;
	float lengthOfLine;
	if(MYcol <= CenterPixelX){ //To the left of or at the middle Vertical Bar (or at the middle)
		if(MYrow <= CenterPixelY){ //Below or at the middle Horizontal Bar (Quadrant 1)
			lengthOfLine = sqrtf(powf((CenterPixelX-MYcol),2)+powf((CenterPixelY-MYrow),2));
		}
		else{ //Above the middle horizontal bar
			lengthOfLine = sqrtf(powf((CenterPixelX-MYcol),2)+powf((MYrow-CenterPixelY),2));
		}
	}
	else{ //To the right of the middle Vertical bar
		if(MYrow <= CenterPixelY){ //Below of the middle Horizontal Bar
			lengthOfLine = sqrtf(powf((MYcol-CenterPixelX),2)+powf((CenterPixelY-MYrow),2));
		}
		else{ //Above the middle horizontal bar
			lengthOfLine = sqrtf(powf((MYcol-CenterPixelX),2)+powf((MYrow-CenterPixelY),2));
		}
	}
	float percentageOfLine = lengthOfLine/longestLine;
	float multiplier = 0.5+ 1.5*percentageOfLine;
	float PixelR = multiplier*ImgSrc[MYsrcIndex];
	float PixelG = multiplier*ImgSrc[MYsrcIndex + 1];
	float PixelB = multiplier*ImgSrc[MYsrcIndex + 2];
	if(PixelR > 0xFF){
		ImgDst[MYsrcIndex] = 0xFF;
	}
	else{
		ImgDst[MYsrcIndex] = PixelR;
	}
	if(PixelG > 0xFF){
		ImgDst[MYsrcIndex + 1] = 0xFF;
	}
	else{
		ImgDst[MYsrcIndex + 1] = PixelG;
	}
	if(PixelB > 0xFF){
		ImgDst[MYsrcIndex + 2] = 0xFF;
	}
	else{
		ImgDst[MYsrcIndex + 2] = PixelB;
	}
}

//This operation removes the power and simply mulitplies both values by itself. 
//It also removes the division operation by doing a constant division mutliple in the main and instead passing it as variable 
//"longestLineMultiplier" which can now be treated as a float point multiplication operation
__global__
void Operation2(uch *ImgDst, uch *ImgSrc, ui Hpixels, ui RowBytes,ui CenterPixelX,ui CenterPixelY,float longestLineMultiplier)
{
	ui ThrPerBlk = blockDim.x;
	ui MYbid = blockIdx.x;
	ui MYtid = threadIdx.x;
	ui MYrow = blockIdx.y;
	ui MYcol = MYbid*ThrPerBlk + MYtid;
	if (MYcol >= Hpixels) return;			// col out of range
	ui MYoffset = MYrow * RowBytes;
	ui MYsrcIndex = MYoffset + 3 * MYcol;
	float lengthOfLine;
	if(MYcol <= CenterPixelX){ //To the left of or at the middle Vertical Bar (or at the middle)
		if(MYrow <= CenterPixelY){ //Below or at the middle Horizontal Bar (Quadrant 1)
			lengthOfLine = sqrtf((CenterPixelX-MYcol)*(CenterPixelX-MYcol)+(CenterPixelY-MYrow)*(CenterPixelY-MYrow));
		}
		else{ //Above the middle horizontal bar
			lengthOfLine = sqrtf((CenterPixelX-MYcol)*(CenterPixelX-MYcol)+(MYrow-CenterPixelY)*(MYrow-CenterPixelY));
		}
	}
	else{ //To the right of the middle Vertical bar
		if(MYrow <= CenterPixelY){ //Below of the middle Horizontal Bar
			lengthOfLine = sqrtf((MYcol-CenterPixelX)*(MYcol-CenterPixelX)+(CenterPixelY-MYrow)*(CenterPixelY-MYrow));
		}
		else{ //Above the middle horizontal bar
			lengthOfLine = sqrtf((MYcol-CenterPixelX)*(MYcol-CenterPixelX)+(MYrow-CenterPixelY)*(MYrow-CenterPixelY));
		}
	}
	float percentageOfLine = lengthOfLine*longestLineMultiplier;
	float multiplier = 0.5+ 1.5*percentageOfLine;
	float PixelR = multiplier*ImgSrc[MYsrcIndex];
	float PixelG = multiplier*ImgSrc[MYsrcIndex + 1];
	float PixelB = multiplier*ImgSrc[MYsrcIndex + 2];
	if(PixelR > 0xFF){
		ImgDst[MYsrcIndex] = 0xFF;
	}
	else{
		ImgDst[MYsrcIndex] = PixelR;
	}
	if(PixelG > 0xFF){
		ImgDst[MYsrcIndex + 1] = 0xFF;
	}
	else{
		ImgDst[MYsrcIndex + 1] = PixelG;
	}
	if(PixelB > 0xFF){
		ImgDst[MYsrcIndex + 2] = 0xFF;
	}
	else{
		ImgDst[MYsrcIndex + 2] = PixelB;
	}
}

// This improvement moves 4 pixels at a time instead of 3 using registers and swapping of types from uch to ui
// Although, it had to be split up into two kernels because this method will fail if your image length is not divisible by 4
// If we try to process an image not divisible by 4, we will run the risk of consuming data that falls outside of our 
// data sets limits which will produce a segmentation fault type error.
__global__
void Operation3A(ui *ImgDst32, ui *ImgSrc32, ui Hpixels, ui RowBytes,ui CenterPixelX,ui CenterPixelY,float longestLineMultiplier,ui RowInts){
	ui ThrPerBlk = blockDim.x;
	ui MYbid = blockIdx.x;
	ui MYtid = threadIdx.x;
	ui MYrow = blockIdx.y;
	ui MYcol = (MYbid*ThrPerBlk + MYtid)*3;
	if (MYcol >= RowInts) return;			// col out of range
	ui MYoffset = MYrow * RowInts;
	ui MYsrcIndex = MYoffset + MYcol;
	ui A,B,C;
	A = ImgSrc32[MYsrcIndex]; // [B1 R0 G0 B0] As in this is how it looks bitwise (B1 is Bits 31-24) (R is bits 23-16)(G is Bits 15-8) (B is Bits 0-7)
	B = ImgSrc32[MYsrcIndex + 1]; // [G2 B2 R1 G1]
	C = ImgSrc32[MYsrcIndex + 2]; // [R3 G3 B3 R2]
	float charMyCol = (MYbid*ThrPerBlk + MYtid)*4 + 1; //Translates it to a normal char col (To reduce distance shrinkage effects)
	float lengthOfLine;
	if(charMyCol <= CenterPixelX){ //To the left of or at the middle Vertical Bar (or at the middle)
		if(MYrow <= CenterPixelY){ //Below or at the middle Horizontal Bar (Quadrant 1)
			lengthOfLine = sqrtf((CenterPixelX-charMyCol)*(CenterPixelX-charMyCol)+(CenterPixelY-MYrow)*(CenterPixelY-MYrow));
		}
		else{ //Above the middle horizontal bar
			lengthOfLine = sqrtf((CenterPixelX-charMyCol)*(CenterPixelX-charMyCol)+(MYrow-CenterPixelY)*(MYrow-CenterPixelY));
		}
	}
	else{ //To the right of the middle Vertical bar
		if(MYrow <= CenterPixelY){ //Below of the middle Horizontal Bar
			lengthOfLine = sqrtf((charMyCol-CenterPixelX)*(charMyCol-CenterPixelX)+(CenterPixelY-MYrow)*(CenterPixelY-MYrow));
		}
		else{ //Above the middle horizontal bar
			lengthOfLine = sqrtf((charMyCol-CenterPixelX)*(charMyCol-CenterPixelX)+(MYrow-CenterPixelY)*(MYrow-CenterPixelY));
		}
	}
	float percentageOfLine = lengthOfLine*longestLineMultiplier;
	float multiplier = 0.5+ 1.5*percentageOfLine;
	int B0 = multiplier * (ImgSrc32[MYsrcIndex] & 0x000000FF);
	int G0 = multiplier * (ImgSrc32[MYsrcIndex] >> 8 & 0x000000FF);
	int R0 = multiplier * (ImgSrc32[MYsrcIndex] >> 16 & 0x000000FF);
	int B1 = multiplier * (ImgSrc32[MYsrcIndex] >> 24 & 0x000000FF);
	int G1 = multiplier * (ImgSrc32[MYsrcIndex + 1] & 0x000000FF);
	int R1 = multiplier * (ImgSrc32[MYsrcIndex + 1] >> 8 & 0x000000FF);
	int B2 = multiplier * (ImgSrc32[MYsrcIndex + 1] >> 16 & 0x000000FF);
	int G2 = multiplier * (ImgSrc32[MYsrcIndex + 1] >> 24 & 0x000000FF);
	int R2 = multiplier * (ImgSrc32[MYsrcIndex + 2]  & 0x000000FF);
	int B3 = multiplier * (ImgSrc32[MYsrcIndex + 2] >> 8 & 0x000000FF);
	int G3 = multiplier * (ImgSrc32[MYsrcIndex + 2] >> 16 & 0x000000FF);
	int R3 = multiplier * (ImgSrc32[MYsrcIndex + 2] >> 24 & 0x000000FF);
	if(B0 > 0xFF){
		B0 = 0x000000FF;
	}
	if(G0 > 0xFF){
		G0 = 0x000000FF;
	}
	if(R0 > 0xFF){
		R0 = 0x000000FF;
	}
	if(B1 > 0xFF){
		B1 = 0x000000FF;
	}
	if(G1 > 0xFF){
		G1 = 0x000000FF;
	}
	if(R1 > 0xFF){
		R1 = 0x000000FF;
	}
	if(B2 > 0xFF){
		B2 = 0x000000FF;
	}
	if(G2 > 0xFF){
		G2 = 0x000000FF;
	}
	if(R2 > 0xFF){
		R2 = 0x000000FF;
	}
	if(B3 > 0xFF){
		B3 = 0x000000FF;
	}
	if(G3 > 0xFF){
		G3 = 0x000000FF;
	}
	if(R3 > 0xFF){
		R3 = 0x000000FF;
	}
	A = (B0) | (G0 << 8) | (R0 << 16) | (B1 << 24);
	B = (G1) | (R1 << 8) | (B2 << 16) | (G2 << 24);
	C = (R2) | (B3 << 8) | (G3 << 16) | (R3 << 24);
	ImgDst32[MYsrcIndex] = A;
	ImgDst32[MYsrcIndex + 1] = B;
	ImgDst32[MYsrcIndex + 2] = C;
}

// This is the second part operation 3, essentially, it is operation 2. Finishes off the section of the image that is not divisible
// by far, thus protecting the programs integrity. Considering this should be the last column, I don't anticipate this to ruin the programs
// efficiency depending on Smaller Image Sizes and ThrPerBlk numbers.
__global__
void Operation3B(uch *ImgDst, uch *ImgSrc, ui Hpixels, ui RowBytes,ui CenterPixelX,ui CenterPixelY,float longestLineMultiplier,ui BlockBase){
	ui ThrPerBlk = blockDim.x;
	ui MYbid = BlockBase + blockIdx.x;
	ui MYtid = threadIdx.x;
	ui MYrow = blockIdx.y;
	ui MYcol = MYbid*ThrPerBlk + MYtid;
	if (MYcol >= Hpixels) return;			// col out of range
	ui MYoffset = MYrow * RowBytes;
	ui MYsrcIndex = MYoffset + 3 * MYcol;
	float lengthOfLine;
	if(MYcol <= CenterPixelX){ //To the left of or at the middle Vertical Bar (or at the middle)
		if(MYrow <= CenterPixelY){ //Below or at the middle Horizontal Bar (Quadrant 1)
			lengthOfLine = sqrtf((CenterPixelX-MYcol)*(CenterPixelX-MYcol)+(CenterPixelY-MYrow)*(CenterPixelY-MYrow));
		}
		else{ //Above the middle horizontal bar
			lengthOfLine = sqrtf((CenterPixelX-MYcol)*(CenterPixelX-MYcol)+(MYrow-CenterPixelY)*(MYrow-CenterPixelY));
		}
	}
	else{ //To the right of the middle Vertical bar
		if(MYrow <= CenterPixelY){ //Below of the middle Horizontal Bar
			lengthOfLine = sqrtf((MYcol-CenterPixelX)*(MYcol-CenterPixelX)+(CenterPixelY-MYrow)*(CenterPixelY-MYrow));
		}
		else{ //Above the middle horizontal bar
			lengthOfLine = sqrtf((MYcol-CenterPixelX)*(MYcol-CenterPixelX)+(MYrow-CenterPixelY)*(MYrow-CenterPixelY));
		}
	}
	float percentageOfLine = lengthOfLine*longestLineMultiplier;
	float multiplier = 0.5+ 1.5*percentageOfLine;
	float PixelR = multiplier*ImgSrc[MYsrcIndex];
	float PixelG = multiplier*ImgSrc[MYsrcIndex + 1];
	float PixelB = multiplier*ImgSrc[MYsrcIndex + 2];
	if(PixelR > 0xFF){
		ImgDst[MYsrcIndex] = 0xFF;
	}
	else{
		ImgDst[MYsrcIndex] = PixelR;
	}
	if(PixelG > 0xFF){
		ImgDst[MYsrcIndex + 1] = 0xFF;
	}
	else{
		ImgDst[MYsrcIndex + 1] = PixelG;
	}
	if(PixelB > 0xFF){
		ImgDst[MYsrcIndex + 2] = 0xFF;
	}
	else{
		ImgDst[MYsrcIndex + 2] = PixelB;
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
	printf("\n Input File name: %17s  (%u x %u)   File Size=%u", fn, 
			ip.Hpixels, ip.Vpixels, IMAGESIZE);
	// allocate memory to store the main image (1 Dimensional array)
	Img  = (uch *)malloc(IMAGESIZE);
	if (Img == NULL) return Img;      // Cannot allocate memory
	// read the image from disk
	fread(Img, sizeof(uch), IMAGESIZE, f);
	fclose(f);
	return Img;
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
	cudaError_t		cudaStatus, cudaStatus2;
	cudaEvent_t		time1, time2, time3, time4;
	char			InputFileName[255], OutputFileName[255], ProgName[255];
	ui				BlkPerRow; //BlkPerRowInt, BlkPerRowInt2;
	ui				ThrPerBlk = 256, NumBlocks;
	//ui			    NB2, NB4, NB8;
	ui 				GPUDataTransfer;
	ui				RowBytes, RowInts;
	cudaDeviceProp	GPUprop;
	ul				SupportedKBlocks, SupportedMBlocks, MaxThrPerBlk;
	ui				*GPUCopyImg32, *GPUImg32;
	char			SupportedBlocks[100];
	//int				KernelNum=1;
	char			KernelName[255];
	int 			Operation; //Start it as the base operation

	strcpy(ProgName, "Proj2");
	switch (argc){
	case 5:  Operation = atoi(argv[4]);
	case 4:  ThrPerBlk = atoi(argv[3]);
	case 3:  strcpy(InputFileName, argv[1]);
			 strcpy(OutputFileName, argv[2]);
			 break;
	default: printf("\n\nUsage:   %s InputFilename OutputFilename [ThrPerBlk] [Operations 1-3]", ProgName);
			 printf("\n\nPlease Note: Operation and ThrPerBlk Do NOT have set values, you HAVE to put something valid!");
			 printf("\n\nThrPerBlk Limit: 32-1024 , Operations: 1= Basic Forumala, 2 = Faster Operation 1, 3 = Much Faster Operation 1\n\n");
			 exit(EXIT_FAILURE);
	}
	if ((Operation < 1) || (Operation > 3)) {
		printf("Invalid Operation Option '%d'. Must be 1,2 or 3 ... \n", Operation);
		exit(EXIT_FAILURE);
	}
	if ((ThrPerBlk < 32) || (ThrPerBlk > 1024)) {
		printf("Invalid ThrPerBlk option '%u'. Must be between 32 and 1024. \n", ThrPerBlk);
		exit(EXIT_FAILURE);
	}
	// Create CPU memory to store the input and output images
	TheImg = ReadBMPlin(InputFileName); // Read the input image if memory can be allocated
	if (TheImg == NULL){
		printf("Cannot allocate memory for the input image...\n");
		exit(EXIT_FAILURE);
	}
	CopyImg = (uch *)malloc(IMAGESIZE);
	if (CopyImg == NULL){
		free(TheImg);
		printf("Cannot allocate memory for the input image...\n");
		exit(EXIT_FAILURE);
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
	if ((cudaStatus != cudaSuccess) || (cudaStatus2 != cudaSuccess)){
		fprintf(stderr, "cudaMalloc failed! Can't allocate GPU memory");
		exit(EXIT_FAILURE);
	}
	// These are the same pointers as GPUCopyImg and GPUImg, however, casted to an integer pointer
	GPUCopyImg32 = (ui *)GPUCopyImg;
	GPUImg32 = (ui *)GPUImg;

	// Copy input vectors from host memory to GPU buffers.
	cudaStatus = cudaMemcpy(GPUImg, TheImg, IMAGESIZE, cudaMemcpyHostToDevice);
	if (cudaStatus != cudaSuccess) {
		fprintf(stderr, "cudaMemcpy  CPU to GPU  failed!");
		exit(EXIT_FAILURE);
	}

	cudaEventRecord(time2, 0);		// Time stamp after the CPU --> GPU tfr is done
	
	RowBytes = (IPH * 3 + 3) & (~3);
	RowInts = RowBytes / 4;
	BlkPerRow = CEIL(IPH,ThrPerBlk);
	//BlkPerRowInt = CEIL(RowInts, ThrPerBlk);
	//BlkPerRowInt2 = CEIL(CEIL(RowInts,2), ThrPerBlk);
	NumBlocks = IPV*BlkPerRow;
	//ui lastColumn = CEIL(BlkPerRow,4)-1;
	dim3 dimGrid2D(BlkPerRow,		   ip.Vpixels);
	dim3 dimGrid2DOp3DivBy4(CEIL(BlkPerRow,4), ip.Vpixels);
	dim3 dimGrid2DOp3NotDivBy4A(CEIL(BlkPerRow,4)-1, ip.Vpixels);
	int Helper = IPH%4;
	dim3 dimGrid2DOp3NotDivBy4B(4, ip.Vpixels); //Only do blocks of 1 pixel
	//dim3 dimGrid2D2(CEIL(BlkPerRow,2), ip.Vpixels);
	//dim3 dimGrid2D4(CEIL(BlkPerRow,4), ip.Vpixels);
	//dim3 dimGrid2D4H91A(CEIL(BlkPerRow,4)-1, ip.Vpixels);
	//dim3 dimGrid2D4H91B(1, ip.Vpixels);
	//dim3 dimGrid2Dint(BlkPerRowInt,    ip.Vpixels);
	//dim3 dimGrid2Dint2(BlkPerRowInt2,  ip.Vpixels);
	float CenterPixelX = IPH*0.5; //Figure out the middle x pixel
	float CenterPixelY = IPV*0.5; //Figure out the middle y pixel
	float longestLine = sqrtf(powf((CenterPixelX-0),2)+powf((CenterPixelY-0),2)); //Figure out the length of the longest line
	//float Operation23longestLine = 0.1*(powf((CenterPixelX-0),2)+powf((CenterPixelY-0),2)); //Figure out the length of the longest line
	float longestLineMultiplier = 1/longestLine;

	switch (Operation){
		case 1: 
			Operation1 <<< dimGrid2D, ThrPerBlk >>> (GPUCopyImg, GPUImg, IPH, RowBytes,CenterPixelX,CenterPixelY,longestLine);
			strcpy(KernelName,"Operation 1 : Each thread will brighten or darken 1 pixel (using a 2D grid)");
			GPUResult = GPUCopyImg;
			GPUDataTransfer = 2*IMAGESIZE;
			break;
		case 2:
			Operation2 <<< dimGrid2D, ThrPerBlk >>> (GPUCopyImg, GPUImg, IPH, RowBytes,CenterPixelX,CenterPixelY,longestLineMultiplier);
			strcpy(KernelName,"Operation 2 : Each thread will brighten or darken 1 pixel (using a 2D grid and improved from Operation 1!)");
			GPUResult = GPUCopyImg;
			GPUDataTransfer = 2*IMAGESIZE;
			break;
		case 3:
			if(IPH%4 != 0){ //We need to split it up into two kernels
				Operation3A <<< dimGrid2DOp3NotDivBy4A, ThrPerBlk >>> (GPUCopyImg32, GPUImg32, IPH, RowBytes,CenterPixelX,CenterPixelY,longestLineMultiplier,RowInts);
				Operation3B <<< dimGrid2DOp3NotDivBy4B, ThrPerBlk >>> (GPUCopyImg, GPUImg, IPH, RowBytes,CenterPixelX,CenterPixelY,longestLineMultiplier,BlkPerRow-4);
			}
			else{ //We can process the image as is, no extra stuff
				Operation3A <<< dimGrid2DOp3DivBy4, ThrPerBlk >>> (GPUCopyImg32, GPUImg32, IPH, RowBytes,CenterPixelX,CenterPixelY,longestLineMultiplier,RowInts);
			}
			strcpy(KernelName,"Operation 3 : Each thread will brighten or darken 1 pixel (using a 2D grid and more improved from Operation 1!)");
			GPUResult = GPUCopyImg;
			GPUDataTransfer = 2*IMAGESIZE;
			break;
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



