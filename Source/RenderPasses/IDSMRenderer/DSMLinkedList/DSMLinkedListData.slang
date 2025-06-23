#ifndef USE_COLOR_TRANSPARENCY
#define USE_COLOR_TRANSPARENCY 0
#endif

#if USE_COLOR_TRANSPARENCY
#define LL_TRANSPARENCY_FORMAT float3
#else
#define LL_TRANSPARENCY_FORMAT float
#endif

static const bool kUseColorTransparency = USE_COLOR_TRANSPARENCY;

struct DSMLinkedListNode
{
    float depth;
    LL_TRANSPARENCY_FORMAT visibility;
    uint next;
};

struct DSMLinkedListNeighborsNode
{
    uint prev;
    uint right;
    uint bot;
}
