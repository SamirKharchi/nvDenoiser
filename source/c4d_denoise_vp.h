#ifndef C4D_DENOISE_VP_H
#define C4D_DENOISE_VP_H

#include "c4d.h"
#include "c4d_symbols.h"

class nvDenoise : public VideoPostData
{
public:
	virtual RENDERRESULT		Execute				(BaseVideoPost* node, VideoPostStruct* vps);
	virtual Bool				RenderEngineCheck	(BaseVideoPost* node, Int32 id);
	virtual VIDEOPOSTINFO		GetRenderInfo		(BaseVideoPost* node) { return VIDEOPOSTINFO_0; }
		
	static NodeData*			Alloc				(void) { return NewObjClear(nvDenoise); }

	virtual Bool Init(GeListNode* node) override;

};

#endif