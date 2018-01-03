#include <boost/compute/core.hpp>
#include <boost/compute/type_traits.hpp>
#include <boost/compute/utility/source.hpp>
#include <boost/compute/container/vector.hpp>
#include <boost/compute/types/struct.hpp>
#include "nvdenoise.h"

#include "c4d_general.h"
#include "c4d_string.h"

BOOST_COMPUTE_ADAPT_STRUCT(nvDenoiseSettings, nvDenoiseSettings, (patch_size, search_size, search_offset, h))

const char nlm[] = BOOST_COMPUTE_STRINGIZE_SOURCE
(
	uint create_index(const uint x, const uint y, const uint size) { return (size * y) + x; }

	uint create_index_arr(const uint x, const uint y, const uint size) { return (size * y * 3) + (x * 3); }

	float square(const float x) { return x * x; }
		
	__kernel void non_local_mean( __global const float* noisy_image
								, __global float* denoised_image
								, const int width
								, const int height
								, const int patch_size
								, const int search_size
								, const int search_offset
								, const float h2)
	{
		const uint x = get_global_id(0);
		const uint y = get_global_id(1);

		uint index = create_index(x,y,width) * 3;
		
		const int patch_size_half = (patch_size - 1) / 2;
		const int search_size_half = (search_size - 1) / 2;

		//current pixel's search window start coordinates (centered at p = (x,y))
		const int x_start = x - search_size_half;
		const int y_start = y - search_size_half;

		//current pixel's patch start coordinates (centered at p = (x,y))
		const int x_startp = x - patch_size_half;
		const int y_startp = y - patch_size_half;

		float3 norm_factor = (float3)(0.0f); //normalizing factor
		
		//accumulate the final weighted output pixel value
		float3 out		= (float3)(0.0f);
		float3 cweight	= (float3)(0.0f);

		//Browse the 'search zone' and average pixel weights for each patch
		int searchpos_y = y_start, searchpos_x, nindex, idx_w, pindex, qindex;
		int i, j, a, b, nbpatch_qx, nbpatch_qy, patchpos_py, patchpos_qy, patchpos_px, patchpos_qx;

		for(j = 0; j < search_size; j += search_offset, searchpos_y += search_offset) 
		{
			searchpos_x = x_start;
			nindex		= create_index_arr(searchpos_x, searchpos_y, width); //image index
			idx_w		= create_index_arr(0, j, search_size); //Weight index
			for(i = 0; i < search_size; i += search_offset, idx_w += (search_offset * 3), searchpos_x += search_offset, nindex += (search_offset * 3)) 
			{
				if((searchpos_x >= width) || (searchpos_x < 0) || (searchpos_y >= height) || (searchpos_y < 0))
					continue;

				//Neighbor patch to evaluate weighting kernel in
				//patch coordinates centered at q
				nbpatch_qx = searchpos_x - patch_size_half;
				nbpatch_qy = searchpos_y - patch_size_half;

				//retrieve w(i,j) kernel:
				//
				//w(i,j)	= exp(-(d(i,j)² / 2h²)) / E(j){ exp(-(d(i,j)² / 2h²)) }
				//d(i,j)	= v(i + t) - v(j + t)
				//v(i)		= pixel value
				//i,j		= 2D patch coordinates(x,y) of p,q
				patchpos_py = y_startp;
				patchpos_qy = nbpatch_qy;

				cweight = (float3)(0.0f);
				// weight value equals sum of all squared differences of neighborhood pixels (see denominator of w(i,j))
				for(b = 0; b < patch_size; b++, ++patchpos_py, ++patchpos_qy) 
				{
					if((patchpos_qy >= height) || (patchpos_qy < 0) || (patchpos_py >= height) || (patchpos_py < 0)) 
						continue;

					patchpos_px = x_startp;
					patchpos_qx = nbpatch_qx;
					
					pindex = create_index_arr(patchpos_px, patchpos_py, width);
					qindex = create_index_arr(patchpos_qx, patchpos_qy, width);

					for(a = 0; a < patch_size; a++, ++patchpos_px, ++patchpos_qx, pindex += 3, qindex += 3) 
					{
						//out of bounds check
						if((patchpos_qx >= width) || (patchpos_qx < 0) || (patchpos_px >= width) || (patchpos_px < 0))
						   continue;

						//get intensity values (color) at patch pixels p & q
						//accumulate squared difference for each color component
						cweight.x += square(noisy_image[pindex] - noisy_image[qindex]);
						cweight.y += square(noisy_image[pindex + 1] - noisy_image[qindex + 1]);
						cweight.z += square(noisy_image[pindex + 2] - noisy_image[qindex + 2]);						
					}
				}

				//getting the exponent of current pixel in the weight matrix and divide it by squared h (nominator)
				cweight.x	= exp(-cweight.x / h2);
				cweight.y	= exp(-cweight.y / h2);
				cweight.z	= exp(-cweight.z / h2);
				
				//accumulate the final weighted output pixel value
				out.x += (cweight.x * noisy_image[nindex]);
				out.y += (cweight.y * noisy_image[nindex + 1]);
				out.z += (cweight.z * noisy_image[nindex + 2]);

				//accumulate normalizing factor z (denominator)
				norm_factor.x += cweight.x;
				norm_factor.y += cweight.y;
				norm_factor.z += cweight.z;				
			}
		}
		denoised_image[index]		= out.x / norm_factor.x;
		denoised_image[index + 1]	= out.y / norm_factor.y;
		denoised_image[index + 2]	= out.z / norm_factor.z;		
	}
);

void NAVIE_GLOBAL::nvNLMdenoiser::non_local_mean_cl(const nvDenoiseSettings& settings
													, const std::vector<float>& noisy_image
													, std::vector<float>& denoised_image
													, const int width
													, const int height)
{
	const int beginX = 0;
	const int beginY = 0;
	const int endX = width;
	const int endY = height;

	const int patch_size_half = (settings.patch_size - 1) / 2;
	const int seach_size_half = (settings.search_size - 1) / 2;

	const float h2 = 1.f / (settings.h * settings.h);

	boost::compute::device device = boost::compute::system::default_device();
	boost::compute::context context(device);
	boost::compute::command_queue queue(context, device/*, CL_QUEUE_OUT_OF_ORDER_EXEC_MODE_ENABLE*/);

	/* create gpu memory and fill it with data from the host */
	boost::compute::vector<float> input_image(noisy_image, queue);

	/* create gpu memory for the denoised result */
	boost::compute::vector<float> output_image(denoised_image.size(), 0.0f, queue);

	/* Create the kernel */
	std::string final_code = boost::compute::type_definition<nvDenoiseSettings>() + "\n" + nlm;
	boost::compute::program nlm_program = boost::compute::program::build_with_source(final_code, context);
		
	boost::compute::kernel nlm_kernel(nlm_program, "non_local_mean");
	nlm_kernel.set_arg(0, input_image);
	nlm_kernel.set_arg(1, output_image);
	nlm_kernel.set_arg(2, width);
	nlm_kernel.set_arg(3, height);
	nlm_kernel.set_arg(4, settings.patch_size);
	nlm_kernel.set_arg(5, settings.search_size);
	nlm_kernel.set_arg(6, settings.search_offset);
	nlm_kernel.set_arg(7, h2);

	size_t kernels		= 32;
	size_t clusters		= 1;
	size_t size_fraction = (endY / kernels);

	size_t kernelsm		= kernels - 1;
	size_t clustersm	= clusters - 1;
	
	std::vector<boost::compute::event> events;
	int cluster = 0;
	for(int i = 0; i < kernels; ++i, ++cluster)
	{
		size_t start[2] = { 0, size_fraction * i }; //start offset
		size_t end[2] = { endX, size_fraction }; //length of the sequence
		if(i == kernelsm) //Last one takes the rest
			end[1] = endY - start[1];

		try {
			if(i == 0)
				events.push_back(queue.enqueue_nd_range_kernel(nlm_kernel, 2, start, end, 0));
			else
				events.push_back(queue.enqueue_nd_range_kernel(nlm_kernel, 2, start, end, 0, events.back()));

			if(cluster >= clustersm || i == kernelsm) {
				queue.finish();
				cluster = 0;
			}
		}
		catch(boost::compute::opencl_error& e) { GePrint(e.to_string(e.error_code()).c_str()); }
	}

	//Copy denoised result back to host output vector
	boost::compute::copy(output_image.begin(), output_image.end(), denoised_image.begin(), queue);
}