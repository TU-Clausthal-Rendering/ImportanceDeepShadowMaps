from pathlib import WindowsPath, PosixPath
from falcor import *

def render_graph_IDSM_Demo_DLSS():
    g = RenderGraph('IDSM_Demo')
    g.create_pass('IDSMRenderer', 'IDSMRenderer', {})
    g.create_pass('AccumulatePass', 'AccumulatePass', {'enabled': False, 'outputSize': 'Default', 'autoReset': True, 'precisionMode': 'Single', 'maxFrameCount': 0, 'overflowMode': 'Stop'})
    g.create_pass('DLSSPass', 'DLSSPass', {'enabled': True, 'outputSize': 'Default', 'profile': 'DLAA', 'preset': 'Default(CNN)', 'motionVectorScale': 'Relative', 'isHDR': True, 'useJitteredMV': False, 'sharpness': 0.0, 'exposure': 0.0})
    g.create_pass('VideoRecorder', 'VideoRecorder', {})
    g.create_pass('PathBenchmark', 'PathBenchmark', {})
    g.create_pass('FSRPass', 'FSRPass', {})
    g.create_pass('ParticlePass', 'ParticlePass', {})
    g.create_pass('SwitchPass', 'SwitchPass', {'count': 4, 'selected': 0, 'i0': 'DLSS', 'i1': 'FSR', 'i2': 'TAA', 'i3': 'Accumulate'})
    g.create_pass('ToneMapper', 'ToneMapper', {'outputSize': 'Default', 'useSceneMetadata': True, 'exposureCompensation': 0.0, 'autoExposure': False, 'filmSpeed': 100.0, 'whiteBalance': False, 'whitePoint': 6500.0, 'operator': 'Aces', 'clamp': True, 'whiteMaxLuminance': 1.0, 'whiteScale': 11.199999809265137, 'fNumber': 1.0, 'shutter': 1.0, 'exposureMode': 'AperturePriority'})
    g.create_pass('TAA', 'TAA', {'alpha': 0.10000000149011612, 'colorBoxSigma': 1.0, 'antiFlicker': True})
    g.add_edge('IDSMRenderer.outDepth', 'DLSSPass.depth')
    g.add_edge('IDSMRenderer.outMotion', 'DLSSPass.mvec')
    g.add_edge('IDSMRenderer.outColor', 'AccumulatePass.input')
    g.add_edge('IDSMRenderer.outColor', 'DLSSPass.color')
    g.add_edge('VideoRecorder', 'PathBenchmark')
    g.add_edge('PathBenchmark', 'IDSMRenderer')
    g.add_edge('IDSMRenderer.outColor', 'FSRPass.color')
    g.add_edge('IDSMRenderer.outDepth', 'FSRPass.depth')
    g.add_edge('IDSMRenderer.outMotion', 'FSRPass.mvec')
    g.add_edge('ParticlePass', 'VideoRecorder')
    g.add_edge('DLSSPass.output', 'SwitchPass.i0')
    g.add_edge('FSRPass.output', 'SwitchPass.i1')
    g.add_edge('TAA.colorOut', 'SwitchPass.i2')
    g.add_edge('AccumulatePass.output', 'SwitchPass.i3')
    g.add_edge('SwitchPass.out', 'ToneMapper.src')
    g.add_edge('IDSMRenderer.outMotion', 'TAA.motionVecs')
    g.add_edge('IDSMRenderer.outColor', 'TAA.colorIn')
    g.mark_output('ToneMapper.dst')
    return g

IDSM_Demo= render_graph_IDSM_Demo_DLSS()
try: m.addGraph(IDSM_Demo)
except NameError: None
