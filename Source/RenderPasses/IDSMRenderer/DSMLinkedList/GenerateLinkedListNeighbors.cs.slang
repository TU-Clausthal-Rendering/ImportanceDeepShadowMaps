#include "DSMLinkedListData.slang"

#ifndef MAX_ELEMENTS
#define MAX_ELEMENTS 256
#endif

cbuffer CB
{
    uint SMSize; // shadow map size
};

StructuredBuffer<DSMLinkedListNode> gLinkedList;
RWStructuredBuffer<DSMLinkedListNeighborsNode> gLinkedListNeighbors;

[numthreads(16, 16, 1)]
void main(uint2 pixel: SV_DispatchThreadID)
{
    if (any(pixel > SMSize))
        return;
    uint index = pixel.y * SMSize + pixel.x;
    DSMLinkedListNeighborsNode neigh;
    neigh.prev = 0;
    neigh.right = 0;
    neigh.bot = 0;
    if (pixel.x == SMSize - 1 || pixel.y == SMSize - 1)
    {
        gLinkedListNeighbors[index] = neigh;
        return; // no neighbors
    }

    DSMLinkedListNode center = gLinkedList[index];
    uint rightIndex = pixel.y * SMSize + pixel.x + 1;
    DSMLinkedListNode right = gLinkedList[rightIndex];
    uint botIndex = (pixel.y + 1) * SMSize + pixel.x;
    DSMLinkedListNode bot = gLinkedList[botIndex];
    for (uint i = 0; i < MAX_ELEMENTS; ++i)
    {
        // find right
        for (uint j = 0; j < MAX_ELEMENTS; ++j)
        {
            if (right.next == 0) break;
            if (gLinkedList[right.next].depth > center.depth)
                break;
            rightIndex = right.next;
            right = gLinkedList[right.next];
        }
        neigh.right = rightIndex;

        // find bot
        for (uint j = 0; j < MAX_ELEMENTS; ++j)
        {
            if (bot.next == 0) break;
            if (gLinkedList[bot.next].depth > center.depth)
                break;
            botIndex = bot.next;
            bot = gLinkedList[bot.next];
        }
        neigh.bot = botIndex;

        // write the node
        gLinkedListNeighbors[index] = neigh;
        neigh.prev = index; // update for next iteration
        index = center.next;
        if (index == 0) break;

        // get the next node
        center = gLinkedList[index];
    }
}
