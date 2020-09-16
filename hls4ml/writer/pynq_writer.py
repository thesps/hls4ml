import os
import numpy as np
from hls4ml.writer.vivado_writer import VivadoWriter
from hls4ml.model.hls_model import IntegerPrecisionType, FixedPrecisionType

class PynqWriter(VivadoWriter):

    def next_axi_type(self, p):
        # Return a new type with the width rounded to the next factor of 8 up to p's width
        W = p.width
        newW = int(np.ceil(W / 8) * 8)
        if isinstance(p, FixedPrecisionType):
            return FixedPrecisionType(newW, p.integer, p.signed, p.rounding_mode, p.saturation_mode, p.saturation_bits)
        elif isinstance(p, IntegerPrecisionType):
            return IntegerPrecisionType(newW, p.signed)


    def write_axi_wrapper(self, model):
        #######################
        ## myproject_axi.h
        #######################

        filedir = os.path.dirname(os.path.abspath(__file__))
        f = open(os.path.join(filedir,'../templates/pynq/myproject_axi.h'),'r')
        fout = open('{}/firmware/{}_axi.h'.format(model.config.get_output_dir(), model.config.get_project_name()),'w')

        model_inputs = model.get_input_variables()
        model_outputs = model.get_output_variables()
        assert len(model_inputs) == 1, "Only models with one input tensor are currently supported by PynqBackend"
        assert len(model_outputs) == 1, "Only models with one output tensor are currently supported by PynqBackend"
        inp = model_inputs[0]
        out = model_outputs[0]
        inp_axi_t = self.next_axi_type(inp.type.precision)
        out_axi_t = self.next_axi_type(inp.type.precision)

        indent = '    '

        for line in f.readlines():

            if 'MYPROJECT' in line:
                newline = line.replace('MYPROJECT',format(model.config.get_project_name().upper()))
            elif 'void myproject(' in line:
                newline = 'void {}(\n'.format(model.config.get_project_name())
            elif '//hls-fpga-machine-learning insert definitions' in line:
                newline = ''
                newline += indent + 'static const unsigned in_size = {}\n'.format(inp.size())
                newline += indent + 'static const unsigned out_size = {}\n'.format(out.size())
                newline += indent + 'typedef input_axi_t {}\n'.format(inp_axi_t)
                newline += indent + 'typedef output_axi_t {}\n'.format(out_axi_t)
                newline += indent + 'typedef input_t {}\n'.format(inp.type.precision)
                newline += indent + 'typedef output_t {}\n'.format(out.type.precision)
                newline += indent + 'typedef {} input_axi_t\n'.format(inp_axi_t)
                newline += indent + 'typedef {} output_axi_t\n'.format(out_axi_t)
                newline += indent + 'typedef {} input_t\n'.format(inp.type.precision)
                newline += indent + 'typedef {} output_t\n'.format(out.type.precision)
            elif '//hls-fpga-machine-learning insert call' in line:
                newline = indent + '{}(in_local, out_local, in_size, out_size);\n'.format(model.config.get_project_name())
            else:
                newline = line
            fout.write(newline)
        f.close()
        fout.close()
        
    def write_hls(self, model):
        super(PynqWriter, self).write_hls(model)
        self.write_axi_wrapper(model)


