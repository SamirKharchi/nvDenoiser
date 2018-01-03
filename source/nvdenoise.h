#ifndef NVDENOISE_H_
#define NVDENOISE_H_

#include <vector/vector3d.h>
#include <vector>

class BaseBitmap;

struct nvDenoiseSettings
{
	float h;
	int patch_size;
	int search_size;
	int search_offset;
};

namespace NAVIE_GLOBAL
{
	/****************************************************************************/
	/* This class implements the non-local mean filter for image denoising		*/
	/* (Buades et al, 2005)														*/
	/*																			*/
	/*	Patch Size																*/
	/*	Too large => oversmoothing of subtle textures and edges					*/
	/*  Larger => better patch similarity recognition.							*/
	/*  Smaller => less smoothing												*/		
	/*  Too small => mottling effect; fake patterns in const intensity regions	*/
	/*																			*/
	/*	Search Zone Size														*/
	/*	Zone around pixel to sample for similar patches							*/
	/*																			*/
	/*	h																		*/
	/*	Exponential Power. Lower values result in more conservative results		*/
	/*																			*/
	/*  search offset															*/		
	/*	higher pixel offset => less patches being checked in the search window	*/
	/****************************************************************************/
	class nvNLMdenoiser
	{
	public:
		void non_local_mean		(const nvDenoiseSettings* settings, const BaseBitmap *noisy_image, std::vector<NAVIE_GLOBAL::Color> &denoised_image );
		void non_local_mean_cl	(const nvDenoiseSettings& settings, const std::vector<float>& noisy_image, std::vector<float> &denoised_image, const int width, const int height );

		/* Returns a 1d index from a 2d pixel coordinate. size is the width of the image */
		static inline int get_index(int x, int y, int size) { return (size * y) + x; }
		static inline int get_index_array(int x, int y, int size) { return get_index(x,y,size) * 3; }

		/* Fills dst with the color (normal range) of the pixel at (x,y) */
		void get_pixel(const BaseBitmap* bmp, int x, int y, NAVIE_GLOBAL::vector3d& dst);
	};
	
}


#endif