/*
 * http://github.com/dusty-nv/jetson-inference
 */

#include "detectNet.h"
#include "loadImage.h"

//.. add hkkim start
#include "glDisplay.h"
#include "glTexture.h"
#include "cudaNormalize.h"
#include "cudaFont.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <signal.h>
#include <unistd.h>
//.. add hkkim end

#include "cudaMappedMemory.h"


#include <sys/time.h>


uint64_t current_timestamp() {
    struct timeval te; 
    gettimeofday(&te, NULL); // get current time
    return te.tv_sec*1000LL + te.tv_usec/1000; // caculate milliseconds
}


// main entry point
int main( int argc, char** argv )
{
	// create detectNet
	detectNet* net = detectNet::Create("./lwir-vehicle-model/deploy.prototxt", "./lwir-vehicle-model/snapshot_iter_45090.caffemodel", "./lwir-vehicle-model/mean.binaryproto" );
	
	if( !net )
	{
		printf("detectnet-lwir:   failed to initialize detectNet\n");
		return 0;
	}

	net->EnableProfiler();
	
	// alloc memory for bounding box & confidence value output arrays
	const uint32_t maxBoxes = net->GetMaxBoundingBoxes();		printf("maximum bounding boxes:  %u\n", maxBoxes);
	const uint32_t classes  = net->GetNumClasses();
	
	float* bbCPU    = NULL;
	float* bbCUDA   = NULL;
	float* confCPU  = NULL;
	float* confCUDA = NULL;
	
	if( !cudaAllocMapped((void**)&bbCPU, (void**)&bbCUDA, maxBoxes * sizeof(float4)) ||
	    !cudaAllocMapped((void**)&confCPU, (void**)&confCUDA, maxBoxes * classes * sizeof(float)) )
	{
		printf("detectnet-lwir:  failed to alloc output memory\n");
		return 0;
	}
	
	// load image from file on disk
	float* imgCPU    = NULL;
	float* imgCUDA   = NULL;
	int    imgWidth  = 320;
	int    imgHeight = 240;

	/* hkkim
	 * create openGL window
	 */
	glDisplay* display = glDisplay::Create();
	glTexture* texture = NULL;
	
	if( !display ) {
		printf("\ndetectnet-lwir:  failed to create openGL display\n");
	}
	else
	{
		texture = glTexture::Create(imgWidth, imgHeight, GL_RGBA32F_ARB/*GL_RGBA8*/);

		if( !texture )
			printf("detectnet-lwir:  failed to create openGL texture\n");
	}

	//const char* imgFilename = argv[1];
	//const char* imgFilename = "./test/veh/000001.bmp";
	char imgFilename[256];
	int ii;
	for(int num=0; num<15025; num++)
	{	
		//ii = sprintf(imgFile, "/home/ubuntu/GITC_LWIR/DB_1st/images/%6d.bmp", num);
		sprintf(imgFilename, "/home/ubuntu/GITC_LWIR/DB_1st/images/%06d.bmp", num);
		printf("%s\n", imgFilename);
	//}

		
		if( !loadImageRGBA(imgFilename, (float4**)&imgCPU, (float4**)&imgCUDA, &imgWidth, &imgHeight) )
		{
			printf("failed to load image '%s'\n", imgFilename);
			return 0;
		}

		// classify image
		int numBoundingBoxes = maxBoxes;
	
		printf("detectnet-lwir:  beginning processing network (%zu)\n", current_timestamp());

		const bool result = net->Detect(imgCUDA, imgWidth, imgHeight, bbCPU, &numBoundingBoxes, confCPU);

		printf("detectnet-lwir:  finished processing network  (%zu)\n", current_timestamp());

		if( !result )
			printf("detectnet-lwir:  failed to classify '%s'\n", imgFilename);

		printf("%i bounding boxes detected\n", numBoundingBoxes);
	
		int lastClass = 0;
		int lastStart = 0;
	
		for( int n=0; n < numBoundingBoxes; n++ )
		{
			const int nc = confCPU[n*2+1];
			float* bb = bbCPU + (n * 4);
		
			printf("bounding box %i   (%f, %f)  (%f, %f)  w=%f  h=%f\n", n, bb[0], bb[1], bb[2], bb[3], bb[2] - bb[0], bb[3] - bb[1]); 
		
			if( nc != lastClass || n == (numBoundingBoxes - 1) )
			{
				if( !net->DrawBoxes(imgCUDA, imgCUDA, imgWidth, imgHeight, bbCUDA + (lastStart * 4), (n - lastStart) + 1, lastClass) )
					printf("detectnet-lwir:  failed to draw boxes\n");
				
				lastClass = nc;
				lastStart = n;
			}
		}
	
		CUDA(cudaThreadSynchronize());

		// update display : hkkim
		if( display != NULL )
		{
			display->UserEvents();
			display->BeginRender();

			if( texture != NULL )
			{
				// rescale image pixel intensities for display
				CUDA(cudaNormalizeRGBA((float4*)imgCPU, make_float2(0.0f, 255.0f), 
								   (float4*)imgCPU, make_float2(0.0f, 1.0f), 
		 						   imgWidth, imgHeight));

				// map from CUDA to openGL using GL interop
				void* tex_map = texture->MapCUDA();

				if( tex_map != NULL )
				{
					cudaMemcpy(tex_map, imgCPU, texture->GetSize(), cudaMemcpyDeviceToDevice);
					texture->Unmap();
				}

				// draw the texture
				texture->Render(100,100);		
			}

			display->EndRender();
		}

		//for(int k=0; k<1000000; k++)
		//{}
		CUDA(cudaFreeHost(imgCPU));
	}
	
	
	printf("\nshutting down...\n");
	


	delete net;
	
	// delete : hkkim
	if( display != NULL )
	{
		delete display;
		display = NULL;
	}

	return 0;
}
