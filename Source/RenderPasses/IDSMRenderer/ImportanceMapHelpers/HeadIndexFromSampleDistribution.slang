/** Traveses the SD using a UV coordinate to aquire the head buffer index.
    \param[in] sampleDistribution   SD texture
    \param[in] idx1D                1D dispatch index for SD traversal
    \param[in] smUV                 UV coordinates from shadow map perspective
    \param[in] startMip             Start MipMap index for traversal. Traversal starts at the second highest level (2x2)
    \param[in] sampleFactorISM      ISM sample multiplication factor. Needs to a integer that fulfills x^2 (e.g 1, 4, 9, 16, ..)
    \return uint2 sampleIndex (x: IDSM; y: ISM, multiplied with the sample factor). Negative values indicate that there is no sample present (unshadowed)
*/

int2 getIndexFromSD(Texture2D<uint> sampleDistribution, float2 smUV, uint startMip, uint sampleFactorISM = 1u) {
    // Go through the mip chain to get the head buffer index of the sample
    uint2 mipIdx = uint2(0);       // Working index for mip chain traversal. Starts top left (0)
    uint selectedIndex = 0;        // Selected head buffer index.
    uint choosenSampleCount = 0;   // Ray sample count of the currently choosen sample.

    // Start traversal from the second highest miplevel to level 0
    for (int m = startMip; m >= 0; m--) {
        // Fetch samples in 2x2 grid
        uint4 mipSamples;
        mipSamples[0] = sampleDistribution.Load(uint3(mipIdx.x, mipIdx.y, m));
        mipSamples[1] = sampleDistribution.Load(uint3(mipIdx.x + 1, mipIdx.y, m));
        mipSamples[2] = sampleDistribution.Load(uint3(mipIdx.x, mipIdx.y + 1, m));
        mipSamples[3] = sampleDistribution.Load(uint3(mipIdx.x + 1, mipIdx.y + 1, m));

        //Select sample using the UV coordinates
        uint totalSamples = mipSamples[0] + mipSamples[1] + mipSamples[2] + mipSamples[3];
        uint pixelLoc = 0;
        if (smUV.x > 0.5)
            pixelLoc += 1;
        if (smUV.y > 0.5)
            pixelLoc += 2;

        // If the total is smaller than the sample count from the iteration above,
        // abort and select the sample from last iteration (Paper case (a))
        if (totalSamples < choosenSampleCount)
            break;

        // Update choosen sample count
        choosenSampleCount = mipSamples[pixelLoc]; 

        //Early out if no samples are distributed at the current position
        if (choosenSampleCount == 0)
            break;

        // Update selected index
        for (uint i = 0; i < pixelLoc; i++)
            selectedIndex += mipSamples[i];

        //Update UV coordinates for the next iteration
        smUV *= 2.0;
        if (pixelLoc == 1 || pixelLoc == 3) // x > 0.5
            smUV.x -= 1.0;
        if (pixelLoc > 1) // y > 0.5
            smUV.y -= 1.0;
        smUV = saturate(smUV); // Guarantee [0,1]

        // Update the index used to fetch the SD samples
        if (pixelLoc == 1 || pixelLoc == 3)
            mipIdx.x += 1;
        if (pixelLoc > 1) // 2 or 3
            mipIdx.y += 1;

        mipIdx *= 2; // Update for one MipMap level deeper
    }

    // No head buffer entry found (unshadowed)
    if (choosenSampleCount == 0)
        return -1; // Invalid index (Unshadowed (1.0))

    uint selectedMaskISMIndex = selectedIndex * sampleFactorISM; // Is calculated seperatly if sampleFactorISM > 1

    // Get final index offset (grid) for the selected sample
    if (choosenSampleCount > 1) {
        uint samplesSq = uint(round(sqrt(choosenSampleCount))); // Num samples should be set capped to grids (e.g. 2x2, 3x3, 4x4)
        const float gridSize = 1.0 / samplesSq;
        uint2 gridOffset = uint2(smUV / gridSize);
        gridOffset.y *= samplesSq;

        selectedIndex += gridOffset.x + gridOffset.y;
    }

    //The ISM has a seperate index if the multiplier is > 1. Else the index is identical to IDSM
    if (sampleFactorISM > 1) {
        choosenSampleCount *= sampleFactorISM; //Factor in multiplier
        uint samplesSq = uint(round(sqrt(choosenSampleCount))); // Num samples should be set capped to grids (e.g. 2x2, 3x3, 4x4)
        const float gridSize = 1.0 / samplesSq;
        uint2 gridOffset = uint2(smUV / gridSize);
        gridOffset.y *= samplesSq;

        selectedMaskISMIndex += gridOffset.x + gridOffset.y;
    } else {
        selectedMaskISMIndex = selectedIndex;
    }

    //Return indices. x: ISDM, y: ISM
    return int2(selectedIndex, selectedMaskISMIndex);
}
