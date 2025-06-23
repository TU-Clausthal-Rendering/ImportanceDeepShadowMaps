from pathlib import WindowsPath, PosixPath
from falcor import *

def render_graph_SceneDebugger():
    g = RenderGraph('SceneDebugger')
    g.create_pass('SceneDebugger', 'SceneDebugger', {'mode': 'FaceNormal', 'showVolumes': 1})
    g.create_pass('ParticlePass', 'ParticlePass', {})
    g.add_edge('ParticlePass', 'SceneDebugger')
    g.mark_output('SceneDebugger.output')
    return g

SceneDebugger = render_graph_SceneDebugger()
try: m.addGraph(SceneDebugger)
except NameError: None
