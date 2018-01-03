#include "c4d.h"

Bool RegisterDenoiser(void);

Bool PluginStart(void)
{
	if(!RegisterDenoiser())
		return false;

	return true;
}

void PluginEnd(void)
{
}

Bool PluginMessage(Int32 id, void *data)
{
	switch (id)
	{
		case C4DPL_INIT_SYS:
			{
				if (!resource.Init()) return FALSE;
				return TRUE;
			}
			break;
	}

	return FALSE;
}