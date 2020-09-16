from hls4ml.templates.vivado_template import VivadoBackend

class PynqBackend(VivadoBackend):
    def __init__(self):
        super(PynqBackend, self).__init__()
        self.name = 'Pynq'
