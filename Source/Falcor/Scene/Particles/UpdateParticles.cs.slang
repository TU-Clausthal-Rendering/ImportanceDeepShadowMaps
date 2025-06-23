import Scene.SceneTypes;

cbuffer CB
{
    uint gIndexOffset;      //Offset particle buffer
    uint gNumParticles;     //Number of particles (dispatch size)
    bool gCopyPrev;         //Should copy prev?
    uint _pad;
    uint4 gVBOffsets;       // Offsets for the vertex buffer
    uint4 gPrevVBOffset;    //Offset in the prev vertex buffer
}

cbuffer ConstantCameraBuffer
{
    float3 gCamU;   //U vec of the camera (right)
    float3 gCamV;   //V vec of the camera (up)
}

StructuredBuffer<ParticlePointDesc> gParticlePoints;
RWStructuredBuffer<PackedStaticVertexData> gVertexBuffer;
RWStructuredBuffer<PrevVertexData> gPrevVertexBuffer;

[numthreads(64, 1, 1)]
void main(uint3 dispatchThreadId : SV_DispatchThreadID)
{
    //Return if thread id is bigger than the number of particles
    if(dispatchThreadId.x >= gNumParticles)
        return;

    const uint particleIdx = dispatchThreadId.x;

    //Get Particle
    ParticlePointDesc prt = gParticlePoints[gIndexOffset + particleIdx];

    if (gCopyPrev){
        [unroll]
        for(uint pIdx = 0; pIdx < 4; pIdx++){
            uint vertexID = gVBOffsets[pIdx] + particleIdx * 4;
            uint prevVertexID = gPrevVBOffset[pIdx] + particleIdx * 4;
            [unroll]
            for(uint i=0; i<4; i++){
                StaticVertexData prevVert = gVertexBuffer[vertexID + i].unpack();
                gPrevVertexBuffer[prevVertexID + i].position = prevVert.position;
            }
        }
    }

    //Verts
    StaticVertexData verts[4];
    float3 normal;
    float4 tangent;
    uint vbIndex;

    //Set TexCoord and curve radius (unchanged for all)
    verts[0].texCrd = float2(0);
    verts[1].texCrd = float2(0,1);
    verts[2].texCrd = float2(1,1);
    verts[3].texCrd = float2(1,0);

    verts[0].curveRadius = 0.0;
    verts[1].curveRadius = 0.0;
    verts[2].curveRadius = 0.0;
    verts[3].curveRadius = 0.0;

    //Camera plane (0)
    float3 offsetRight = (gCamU * prt.radius);
    float3 offetTop = (gCamV * prt.radius);
    verts[0].position = prt.position - offsetRight + offetTop;
    verts[1].position = prt.position + offsetRight + offetTop;
    verts[2].position = prt.position + offsetRight - offetTop;
    verts[3].position = prt.position - offsetRight - offetTop;
    normal = normalize(cross(verts[2].position - verts[0].position, verts[1].position - verts[0].position));
    tangent = float4(normalize(verts[1].position - verts[0].position), 1);
    [unroll]
    for(uint i=0; i<4; i++){
        verts[i].normal = normal;
        verts[i].tangent = tangent;
    }

    vbIndex = gVBOffsets[0] + particleIdx * 4;
    [unroll]
    for(uint i=0; i<4; i++){
        gVertexBuffer[vbIndex + i].pack(verts[i]);
    }
        

    //XY Plane (1)
    verts[0].position = prt.position + float3(-prt.radius, prt.radius,0);
    verts[1].position = prt.position + float3(prt.radius, prt.radius,0);
    verts[2].position = prt.position + float3(prt.radius, -prt.radius,0);
    verts[3].position = prt.position + float3(-prt.radius, -prt.radius,0 );
    normal = normalize(cross(verts[2].position - verts[0].position, verts[1].position - verts[0].position));
    tangent = float4(normalize(verts[1].position - verts[0].position), 1);
    [unroll]
    for(uint i=0; i<4; i++){
        verts[i].normal = normal;
        verts[i].tangent = tangent;
    }

    vbIndex = gVBOffsets[1] + particleIdx * 4;
    [unroll]
    for(uint i=0; i<4; i++){
        gVertexBuffer[vbIndex + i].pack(verts[i]);
    }

    //YZ Plane (2)
    verts[0].position = prt.position + float3(0, prt.radius,-prt.radius);
    verts[1].position = prt.position + float3(0, prt.radius,prt.radius);
    verts[2].position = prt.position + float3(0, -prt.radius,prt.radius);
    verts[3].position = prt.position + float3(0, -prt.radius,-prt.radius);
    normal = normalize(cross(verts[2].position - verts[0].position, verts[1].position - verts[0].position));
    tangent = float4(normalize(verts[1].position - verts[0].position), 1);
    [unroll]
    for(uint i=0; i<4; i++){
        verts[i].normal = normal;
        verts[i].tangent = tangent;
    }

    vbIndex = gVBOffsets[2] + particleIdx * 4;
    [unroll]
    for(uint i=0; i<4; i++){
        gVertexBuffer[vbIndex + i].pack(verts[i]);
    }

    //XZ Plane (3)
    verts[0].position = prt.position + float3(-prt.radius, 0,prt.radius);
    verts[1].position = prt.position + float3(prt.radius, 0,prt.radius);
    verts[2].position = prt.position + float3(prt.radius,0, -prt.radius);
    verts[3].position = prt.position + float3(-prt.radius,0, -prt.radius);
    normal = normalize(cross(verts[2].position - verts[0].position, verts[1].position - verts[0].position));
    tangent = float4(normalize(verts[1].position - verts[0].position), 1);
    [unroll]
    for(uint i=0; i<4; i++){
        verts[i].normal = normal;
        verts[i].tangent = tangent;
    }

    vbIndex = gVBOffsets[3] + particleIdx * 4;
    [unroll]
    for(uint i=0; i<4; i++){
        gVertexBuffer[vbIndex + i].pack(verts[i]);
    }
}
