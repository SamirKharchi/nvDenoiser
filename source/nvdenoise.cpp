#include "nvdenoise.h"
#include <navie_math.h>
#include "c4d.h"

#define USE_MP
#ifdef USE_MP
#include <thread.h>
#endif
	
void NAVIE_GLOBAL::nvNLMdenoiser::get_pixel(const BaseBitmap* bmp, int x, int y, NAVIE_GLOBAL::vector3d& dst)
{
	UInt16 pr, pg, pb;
	bmp->GetPixel(x, y, &pr, &pg, &pb);
	dst.x = float(pr) / 255;
	dst.y = float(pg) / 255;
	dst.z = float(pb) / 255;
}

/* patchwise serial or mp approach */
void NAVIE_GLOBAL::nvNLMdenoiser::non_local_mean(const nvDenoiseSettings* settings, const BaseBitmap *noisy_image, std::vector<NAVIE_GLOBAL::Color>& denoised_image)
{
	const int beginX = 0;
	const int beginY	= 0;
	const int endX		= noisy_image->GetBw();
	const int endY		= noisy_image->GetBh();
	
	const int patch_size_half = (settings->patch_size - 1) / 2;
	const int seach_size_half = (settings->search_size - 1) / 2;

	const double h2 = 2 * sqr(settings->h); //In the paper they only have (h² but it should be 2h². found that info on the internet somewhere)
	
#ifdef USE_MP
	THREADS::nv_parallel_for<boost::thread>((size_t)0, (size_t)endY, [&](size_t y)
	{
		std::vector<Color> Weight(settings->search_size * settings->search_size);
#else
	std::vector<Color> Weight(settings->search_size * settings->search_size);
	for(int y = beginY; y < endY; y++) 
	{
#endif
		for(int x = beginX; x < endX; x++) 
		{			
			//current pixel's search window start coordinates (centered at p = (x,y))
			const int x_start = x - seach_size_half;
			const int y_start = y - seach_size_half;

			//current pixel's patch start coordinates (centered at p = (x,y))
			const int x_startp = x - patch_size_half;
			const int y_startp = y - patch_size_half;
						
			Color norm_factor = Color(0.0); //normalizing factor
			
			int searchpos_y = y_start;
			//Browse the 'search window' and accumulate pixel weights for each patch
			for(int j = 0; j < settings->search_size; j += settings->search_offset, searchpos_y += settings->search_offset) 
			{
				int searchpos_x	= x_start;
				int idx_w = get_index(0, j, settings->search_size); //Weight index
				for(int i = 0; i < settings->search_size; i += settings->search_offset, idx_w += settings->search_offset, searchpos_x += settings->search_offset) 
				{
					Weight[idx_w] = Color(0.0);

					if((searchpos_x >= endX) || (searchpos_x < beginX) || (searchpos_y >= endY) || (searchpos_y < beginY)) 
						continue;

					//Neighbor patch to evaluate weighting kernel in
					//patch coordinates centered at q
					int nbpatch_qx = searchpos_x - patch_size_half;
					int nbpatch_qy = searchpos_y - patch_size_half;

					//retrieve w(i,j) kernel:
					//
					//w(i,j)	= exp(-(d(i,j)² / 2h²)) / E(j){ exp(-(d(i,j)² / 2h²)) }
					//d(i,j)	= v(i + t) - v(j + t)
					//v(i)		= pixel value
					//i,j		= 2D patch coordinates(x,y) of p,q (p and q are pixels)
					int patchpos_py = y_startp;
					int patchpos_qy = nbpatch_qy;

					// weight value equals sum of all squared variance of neighborhood pixels (see denominator of w(i,j))
					for(int b = 0; b < settings->patch_size; b++, ++patchpos_py, ++patchpos_qy) 
					{
						int patchpos_px = x_startp;
						int patchpos_qx = nbpatch_qx;
						for(int a = 0; a < settings->patch_size; a++, ++patchpos_px, ++patchpos_qx) 
						{
							//out of bounds check
							//idea: instead of skipping this pixel, shift the patch into the search window
							//or: clamp the size of each neighbor patch rectangle; if pixel p is near the search windows boundary
							if((patchpos_qx >= endX) || (patchpos_qx < beginX) || (patchpos_qy >= endY) || (patchpos_qy < beginY) ||
							   (patchpos_px >= endX) || (patchpos_px < beginX) || (patchpos_py >= endY) || (patchpos_py < beginY))
								continue;

							//get color values at p & q
							vector3d p = vector3d(0.0), q = vector3d(0.0);
							get_pixel(noisy_image, patchpos_px, patchpos_py, p);
							get_pixel(noisy_image, patchpos_qx, patchpos_qy, q);

							// accumulate squared intensity distance for each color component
							for(int dim = 0; dim < 3; ++dim)
								Weight[idx_w][dim] += sqr(p[dim] - q[dim]);
						}
					}

					for(int dim = 0; dim < 3; ++dim) 
					{
						//getting the exponent of current pixel in the weight matrix and divide it by squared h (nominator part)
						Weight[idx_w][dim] = exp(-Weight[idx_w][dim] / h2);
						//accumulate normalizing factor z (which is the nominator)
						norm_factor[dim] += Weight[idx_w][dim];
					}
				}
			}
			
			//accumulate the final weighted output pixel value
			Color out;
			searchpos_y = y_start;
			for(int jj = 0; jj < settings->search_size; jj += settings->search_offset, searchpos_y += settings->search_offset) 
			{
				int searchpos_x = x_start;
				int idx_w		= get_index(0,jj,settings->search_size); //weight index
				for(int ii = 0; ii < settings->search_size; ii += settings->search_offset, idx_w += settings->search_offset, searchpos_x += settings->search_offset) 
				{
					if((searchpos_x >= endX) || (searchpos_x < beginX) || (searchpos_y >= endY) || (searchpos_y < beginY)) //out of bounds
						continue;
					
					vector3d noisy_src_pixel;
					get_pixel(noisy_image, searchpos_x, searchpos_y, noisy_src_pixel);

					for(int dim = 0; dim < 3; ++dim)
						out[dim] += Weight[idx_w][dim] * noisy_src_pixel[dim];
				}
			}
			out /= norm_factor;

			denoised_image[get_index(x, y, noisy_image->GetBw())] = out;
		}
	}
#ifdef USE_MP
	);
#endif
		
}