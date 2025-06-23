#ifndef USE_HALTON_SAMPLE_PATTERN
#define USE_HALTON_SAMPLE_PATTERN 0
#endif

Buffer<float2> gHaltonSamples;

static const bool kUseHaltonSamplePattern = USE_HALTON_SAMPLE_PATTERN;
static const uint kNumHaltonSamples = NUM_HALTON_SAMPLES;

/** Uses the SD to generate a sample for the current ray.
    \param[in] sampleDistribution   SD texture
    \param[in] idx1D                1D dispatch index for SD traversal
    \param[in] smRes                Base resolution of the shadow map (IM/SD)
    \param[in] startMip             Start MipMap index for traversal. Traversal starts at the second highest level (2x2)
    \param[in] frameCount           Current frame count needed if per-sample halton jitter is enabled
    \param[in/out] pixelExtend      Size of the current pixel box. Used for the AABBs
    \param[in] sampleFactor         ISM sample multiplication factor. Needs to a integer that fulfills x^2 (e.g 1, 4, 9, 16, ..)
    \return Exact pixel location in [0,smRes]. Is fractional
*/
float2 generateRayTracingSampleFromSD(Texture2D<uint> sampleDistribution, uint idx1D, uint2 smRes, uint startMip, uint frameCount, inout float4 pixelExtend, uint sampleFactor = 1u) {
    // Variables used for traversal
    uint2 mipIdx = uint2(0);       // Working index for mip chain traversal. Starts top left (0)
    uint2 selectedMipIdx = mipIdx; //Selected index for the traversal
    uint weightIdx = idx1D; // Uint weight index for traversal
    uint sampleCount = 0;
    uint prevMipRes = 1; // MipMap level resolution for previous iteration
    int sdIndex = 0;     // Index of the sample in the SD
    uint selectedMipLevel = startMip; //Current selected MipMap level in case traversal is stopped early

    // Start traversal from the second highest miplevel to level 0
    for (int m = startMip; m >= 0; m--) {
        //Fetch samples in 2x2 grid
        uint4 mipSamples;
        mipSamples[0] = sampleDistribution.Load(uint3(mipIdx.x, mipIdx.y, m));
        mipSamples[1] = sampleDistribution.Load(uint3(mipIdx.x + 1, mipIdx.y, m));
        mipSamples[2] = sampleDistribution.Load(uint3(mipIdx.x, mipIdx.y + 1, m));
        mipSamples[3] = sampleDistribution.Load(uint3(mipIdx.x + 1, mipIdx.y + 1, m));
        mipSamples *= sampleFactor;

        //Select a sample using the current 1D index
        uint totalSamples = mipSamples[0] + mipSamples[1] + mipSamples[2] + mipSamples[3];
        uint currSamples = 0;      // Total for select in loop
        uint selectedSubpixel = 0; // Selected subpixel id
        uint numSamples = 0;       // Number of samples for that subpixel index
        for (uint x = 0; x < 4 && totalSamples > 0; x++) {
            numSamples = mipSamples[x];
            currSamples += numSamples;
            selectedSubpixel = x;
            if (weightIdx < currSamples)
                break;
        }

        // If the total is smaller than the sample count from the iteration above,
        // abort and select the sample and mipmap level from last iteration (Paper case (a))
        if (totalSamples < sampleCount)
            break;

        // Update traversal weight index
        uint prevSampleIdx = currSamples - numSamples;
        weightIdx -= prevSampleIdx;
        sampleCount = numSamples;

        //Update SD index (needed for Linked List version)
        for (uint i = 0; i < selectedSubpixel; i++)
            sdIndex += mipSamples[i];

        // Update the index used to fetch the SD samples
        if (selectedSubpixel == 1 || selectedSubpixel == 3)
            mipIdx.x += 1;
        if (selectedSubpixel > 1) // 2 or 3
            mipIdx.y += 1;

        //Store them in case next iteration has a early abort
        selectedMipLevel = m;
        selectedMipIdx = mipIdx;

        // Update for one MipMap level deeper
        mipIdx *= 2;
        prevMipRes *= 2;
    }

    int mipMulFactor = 1 << selectedMipLevel; // Multiplier if sample is on a higher mipmap level
    // Get pixel on default resolution
    float2 relPixel = float2(selectedMipIdx) / prevMipRes;
    float2 pixel = floor((relPixel * smRes) + 0.5);
    pixel += 0.5 * mipMulFactor; // Move to center of the pixel
    int gridIndex = max(idx1D - sdIndex, 0); //Get index in grid 

    //Get grid infos
    uint samplesSq = uint(sqrt(sampleCount));   //Sample count is capped on grids (e.g. 2x2, 3x3, 4x4)
    const float gridCellSize = 1.0 / samplesSq;
    uint2 gridLocation = float2(gridIndex % samplesSq, gridIndex / samplesSq); //2D position in grid
    float2 gridCellBorderMin = gridLocation * gridCellSize; //Relative x and y min

    // Per sample jitter with halton (optional)
    float2 gridCenter = float2(0.5);
    if (kUseHaltonSamplePattern) {
        uint jitterIdx = (gridLocation.x + (gridLocation.y * samplesSq) + frameCount) % kNumHaltonSamples;
        gridCenter = gHaltonSamples[jitterIdx]; // pixel offset in [0,1]
    }

    //Get the grid offset and extend
    float2 gridOffset = float2(0); //Relative offset of the grid position to the pixel
    // X
    float2 border = float2(gridCellBorderMin.x, gridCellBorderMin.x + gridCellSize) - 0.5; //From [0, 1] to [-0.5,0.5]; min,max of the grid texel from center
    gridOffset.x = border.x + gridCenter.x * gridCellSize; //Add jitter offset (or center if no jitter is applied)
    pixelExtend.xy = float2(border.x - gridOffset.x, border.y - gridOffset.x);  //Extend of the grid texel
    // Y
    border = float2(gridCellBorderMin.y, gridCellBorderMin.y + gridCellSize) - 0.5; // From [0, 1] to [-0.5,0.5]; min,max of the grid texel from center
    gridOffset.y = border.x + gridCenter.y * gridCellSize; // Add jitter offset (or center if no jitter is applied)
    pixelExtend.zw = float2(border.x - gridOffset.y, border.y - gridOffset.y); // Extend of the grid texel

    // Scale with selected mipmap level
    gridOffset *= mipMulFactor;
    pixelExtend *= mipMulFactor;

    return pixel + gridOffset;
}
