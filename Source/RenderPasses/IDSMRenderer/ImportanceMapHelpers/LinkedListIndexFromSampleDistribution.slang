

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
float2 getIndexFromSD(Texture2D<float> sampleDistribution, uint idx1D, uint2 smRes, uint startMip, uint frameCount, inout float4 pixelExtend, uint sampleFactor = 1u) {
   
}
