from pathlib import WindowsPath, PosixPath
from falcor import *

def render_graph_IDSM_DEMO():
    g = RenderGraph('IDSM_DEMO')
    g.create_pass('Renderer', 'TransparencyRenderer', {})
    g.create_pass('AccumulatePass', 'AccumulatePass', {'enabled': True, 'outputSize': 'Default', 'autoReset': True, 'precisionMode': 'Single', 'maxFrameCount': 0, 'overflowMode': 'Stop'})
    g.create_pass('DLSSPass', 'DLSSPass', {'enabled': True, 'outputSize': 'Default', 'profile': 'DLAA', 'preset': 'Default(CNN)', 'motionVectorScale': 'Relative', 'isHDR': True, 'useJitteredMV': False, 'sharpness': 0.0, 'exposure': 0.0})
    g.create_pass('FSRPass', 'FSRPass', {})
    g.create_pass('ParticlePass', 'ParticlePass', {})
    g.create_pass('SwitchAAMethod', 'SwitchPass', {'count': 4, 'selected': 0, 'i0': 'DLSS', 'i1': 'FSR', 'i2': 'TAA', 'i3': 'Accumulate'})
    g.create_pass('ToneMapper', 'ToneMapper', {'outputSize': 'Default', 'useSceneMetadata': True, 'exposureCompensation': 0.0, 'autoExposure': False, 'filmSpeed': 100.0, 'whiteBalance': False, 'whitePoint': 6500.0, 'operator': 'Aces', 'clamp': True, 'whiteMaxLuminance': 1.0, 'whiteScale': 11.199999809265137, 'fNumber': 1.0, 'shutter': 1.0, 'exposureMode': 'AperturePriority'})
    g.create_pass('TAA', 'TAA', {'alpha': 0.10000000149011612, 'colorBoxSigma': 1.0, 'antiFlicker': True})
    g.add_edge('Renderer.outDepth', 'DLSSPass.depth')
    g.add_edge('Renderer.outMotion', 'DLSSPass.mvec')
    g.add_edge('Renderer.outColor', 'AccumulatePass.input')
    g.add_edge('Renderer.outColor', 'DLSSPass.color')
    g.add_edge('Renderer.outColor', 'FSRPass.color')
    g.add_edge('Renderer.outDepth', 'FSRPass.depth')
    g.add_edge('Renderer.outMotion', 'FSRPass.mvec')
    g.add_edge('ParticlePass', 'Renderer')
    g.add_edge('DLSSPass.output', 'SwitchAAMethod.i0')
    g.add_edge('FSRPass.output', 'SwitchAAMethod.i1')
    g.add_edge('TAA.colorOut', 'SwitchAAMethod.i2')
    g.add_edge('AccumulatePass.output', 'SwitchAAMethod.i3')
    g.add_edge('SwitchAAMethod.out', 'ToneMapper.src')
    g.add_edge('Renderer.outMotion', 'TAA.motionVecs')
    g.add_edge('Renderer.outColor', 'TAA.colorIn')
    g.mark_output('ToneMapper.dst')
    return g

IDSM_DEMO = render_graph_IDSM_DEMO()
try: m.addGraph(IDSM_DEMO)
except NameError: None
