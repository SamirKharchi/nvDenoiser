#include "nvdenoise.h"
#include "c4d_denoise_vp.h"

#include "nvdenoiser.h"

RENDERRESULT nvDenoise::Execute(BaseVideoPost* node, VideoPostStruct* vps)
{
	if (vps->vp == VIDEOPOSTCALL_RENDER && !vps->open && *vps->error == RENDERRESULT_OK && !vps->thread->TestBreak())
	{
		VPBuffer*			rgba = vps->render->GetBuffer(VPBUFFER_RGBA, NOTOK);
		const RayParameter* ray	 = vps->vd->GetRayParameter();	// only in VP_INNER & VIDEOPOSTCALL_RENDER
		if (!ray || !rgba)
			return RENDERRESULT_OUTOFMEMORY;
		
		Int32 x1, y1, x2, y2, x, y, cnt/*, i*/;

		// example functions
		Int32 cpp = rgba->GetInfo(VPGETINFO_CPP);

		x1	= ray->left;
		y1	= ray->top;
		x2	= ray->right;
		y2	= ray->bottom;
		cnt = x2 - x1 + 1;
				
		MultipassBitmap* bmp = reinterpret_cast<MultipassBitmap*>(rgba);

		/************************************************************************/
		// Settings
		/************************************************************************/
		BaseContainer *data = node->GetDataInstance();
		nvDenoiseSettings settings;
		settings.patch_size		= data->GetInt32(NVDENOISE_PATCHSIZE,7); 
		settings.search_size	= data->GetInt32(NVDENOISE_SEARCHSIZE,20);
		settings.search_offset	= data->GetInt32(NVDENOISE_SEARCHOFFSET,3); 
		settings.h				= data->GetFloat(NVDENOISE_STRENGTH,0.4);
		
		/************************************************************************/
		/* Denoise */
		/************************************************************************/		
		NAVIE_GLOBAL::nvNLMdenoiser dn;		

		if(data->GetBool(NVDENOISE_USEGPU, false)) 
		{
			/************************************************************************/
			Int			 bufferSize = cpp * cnt;
			Float32* b, *buffer = nullptr;

			if(bufferSize > 0)
				buffer = NewMemClear(Float32, bufferSize);
			if(!buffer)
				return RENDERRESULT_OUTOFMEMORY;

			/************************************************************************/
			std::vector<float> in_bmp((bmp->GetBw() * bmp->GetBh()) * 3); //color vectors encoded in a float array
			//Copy the buffer into the float vector
			for(y = y1; y <= y2; y++) 
			{
				rgba->GetLine(x1, y, cnt, buffer, 32, true);

				int index = NAVIE_GLOBAL::nvNLMdenoiser::get_index_array(x1, y, bmp->GetBw()); //Weight index
				for(b = buffer, x = x1; x <= x2; x++, b += cpp, index += 3) {
					for(int i = 0; i < 3; i++) 
					{
						in_bmp[index + i] = b[i]; 
					}
				}
			}
			
			std::vector<float> out_bmp(in_bmp);
			dn.non_local_mean_cl(settings, in_bmp, out_bmp, bmp->GetBw(), bmp->GetBh());
			
			/************************************************************************/
			//Apply denoising result to our buffer
			/************************************************************************/
			NAVIE_GLOBAL::vector3d rgb;
			for(y = y1; y <= y2; y++) {
				rgba->GetLine(x1, y, cnt, buffer, 32, true);

				int index = NAVIE_GLOBAL::nvNLMdenoiser::get_index_array(x1, y, bmp->GetBw()); //Weight index
				for(b = buffer, x = x1; x <= x2; x++, b += cpp, index += 3) {
					for(int i = 0; i < 3; i++) 
					{
						b[i] = out_bmp[index + i]; 
					}
				}

				rgba->SetLine(x1, y, cnt, buffer, 32, true);
			}
			DeleteMem(buffer);
		}
		else 
		{
			std::vector<NAVIE_GLOBAL::Color> out_bmp(bmp->GetBw() * bmp->GetBh());
			dn.non_local_mean(&settings, bmp, out_bmp);

			/************************************************************************/
			//Apply denoising result to our buffer
			/************************************************************************/
			Int			 bufferSize = cpp * cnt;
			Float32* b, *buffer = nullptr;

			if(bufferSize > 0)
				buffer = NewMemClear(Float32, bufferSize);
			if(!buffer)
				return RENDERRESULT_OUTOFMEMORY;

			NAVIE_GLOBAL::vector3d rgb;
			for(y = y1; y <= y2; y++) {
				rgba->GetLine(x1, y, cnt, buffer, 32, true);

				int index = NAVIE_GLOBAL::nvNLMdenoiser::get_index(0, y, bmp->GetBw()); //Weight index
				for(b = buffer, x = x1; x <= x2; x++, b += cpp, ++index) {
					for(int i = 0; i < 3; i++) { b[i] = out_bmp[index][i]; }
				}

				rgba->SetLine(x1, y, cnt, buffer, 32, true);
			}
			DeleteMem(buffer);
		}
	}

	return RENDERRESULT_OK;
}

Bool nvDenoise::RenderEngineCheck(BaseVideoPost* node, Int32 id)
{
	// the following render engines are not supported by this effect
	if (id == RDATA_RENDERENGINE_PREVIEWSOFTWARE || id == RDATA_RENDERENGINE_CINEMAN)
		return false;

	return true;
}

Bool nvDenoise::Init(GeListNode* node)
{	
	BaseVideoPost *pp = (BaseVideoPost*)node;
	BaseContainer *data = pp->GetDataInstance();

	data->SetInt32(NVDENOISE_SEARCHSIZE,20);
	data->SetInt32(NVDENOISE_SEARCHOFFSET,3);
	data->SetInt32(NVDENOISE_PATCHSIZE,7);
	data->SetFloat(NVDENOISE_STRENGTH,0.4);

	data->SetBool(NVDENOISE_USEGPU,true);
	data->SetInt32(NVDENOISE_CPUTHREADS,0);
	return true;
}

#define DENOISE_PLUGINID 1800455

Bool RegisterDenoiser(void)
{
	return RegisterVideoPostPlugin(DENOISE_PLUGINID, "nvDenoiser", PLUGINFLAG_VIDEOPOST_MULTIPLE, nvDenoise::Alloc, "nvdenoiser", 0, 0);
}