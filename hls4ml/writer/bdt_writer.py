from .writers import Writer
from .vivado_writer import VivadoWriter

class BDTWriterHLS(VivadoWriter):

    def write_hls(self, model):
    #######################################
    ## Print a BDT to C++
    #######################################
    #def bdt_writer_hls(ensemble_dict, yamlConfig):

        filedir = os.path.dirname(os.path.abspath(__file__))

        ###################
        ## myproject.cpp
        ###################

        #f = open(os.path.join(filedir,'../hls-template/firmware/myproject.cpp'),'r')
        fout = open('{}/firmware/{}.cpp'.format(yamlConfig['OutputDir'], yamlConfig['ProjectName']),'w')
        fout.write('#include "BDT.h"\n')
        fout.write('#include "parameters.h"\n')
        fout.write('#include "{}.h"\n'.format(yamlConfig['ProjectName']))

        fout.write('void {}(input_arr_t x, score_arr_t score){{\n'.format(yamlConfig['ProjectName']))
        # TODO: probably only one of the pragmas is necessary?
        fout.write('\t#pragma HLS pipeline II = {}\n'.format(yamlConfig['ReuseFactor']))
        fout.write('\t#pragma HLS unroll factor = {}\n'.format(yamlConfig['ReuseFactor']))
        fout.write('\t#pragma HLS array_partition variable=x\n\n')
        fout.write('\t#pragma HLS array_partition variable=score\n\n')
        fout.write('\tbdt.decision_function(x, score);\n}')
        fout.close()

        ###################
        ## parameters.h
        ###################

        #f = open(os.path.join(filedir,'../hls-template/firmware/parameters.h'),'r')
        fout = open('{}/firmware/parameters.h'.format(yamlConfig['OutputDir']),'w')
        fout.write('#ifndef BDT_PARAMS_H__\n#define BDT_PARAMS_H__\n\n')
        fout.write('#include    "BDT.h"\n')
        fout.write('#include "ap_fixed.h"\n\n')
        fout.write('static const int n_trees = {};\n'.format(ensemble_dict['n_trees']))
        fout.write('static const int max_depth = {};\n'.format(ensemble_dict['max_depth']))
        fout.write('static const int n_features = {};\n'.format(ensemble_dict['n_features']))
        fout.write('static const int n_classes = {};\n'.format(ensemble_dict['n_classes']))
        fout.write('typedef {} input_t;\n'.format(yamlConfig['DefaultPrecision']))
        fout.write('typedef input_t input_arr_t[n_features];\n')
        fout.write('typedef {} score_t;\n'.format(yamlConfig['DefaultPrecision']))
        fout.write('typedef score_t score_arr_t[n_classes];\n')
        # TODO score_arr_t
        fout.write('typedef input_t threshold_t;\n\n')

        tree_fields = ['feature', 'threshold', 'value', 'children_left', 'children_right', 'parent']

        fout.write("static const BDT::BDT<n_trees, max_depth, n_classes, input_arr_t, score_t, threshold_t> bdt = \n")
        fout.write("{ // The struct\n")
        newline = "\t" + str(ensemble_dict['norm']) + ", // The normalisation\n"
        fout.write(newline)
        newline = "\t{"
        for iip, ip in enumerate(ensemble_dict['init_predict']):
            newline += str(ip)
            if iip < len(ensemble_dict['init_predict']) - 1:
                newline += ','
            else:
                newline += '}, // The init_predict\n'
        fout.write(newline)
        fout.write("\t{ // The array of trees\n")
        # loop over trees
        for itree, trees in enumerate(ensemble_dict['trees']):
            fout.write('\t\t{ // trees[' + str(itree) + ']\n')
            # loop over classes
            for iclass, tree in enumerate(trees):
                fout.write('\t\t\t{ // [' + str(iclass) + ']\n')
                # loop over fields
                for ifield, field in enumerate(tree_fields):
                    newline = '\t\t\t\t{'
                    newline += ','.join(map(str, tree[field]))
                    newline += '}'
                    if ifield < len(tree_fields) - 1:
                        newline += ','
                    newline += '\n'
                    fout.write(newline)
                newline = '\t\t\t}'
                if iclass < len(trees) - 1:
                        newline += ','
                newline += '\n'
                fout.write(newline)
            newline = '\t\t}'
            if itree < ensemble_dict['n_trees'] - 1:
                newline += ','
            newline += '\n'
            fout.write(newline)
        fout.write('\t}\n};')

        fout.write('\n#endif')
        fout.close()

        #######################
        ## myproject.h
        #######################

        f = open(os.path.join(filedir,'../hls-template/firmware/myproject.h'),'r')
        fout = open('{}/firmware/{}.h'.format(yamlConfig['OutputDir'], yamlConfig['ProjectName']),'w')

        for line in f.readlines():
            if 'MYPROJECT' in line:
                newline = line.replace('MYPROJECT',format(yamlConfig['ProjectName'].upper()))
            elif 'void myproject(' in line:
                newline = 'void {}(\n'.format(yamlConfig['ProjectName'])
            elif 'input_t data[N_INPUTS]' in line:
                newline = '\tinput_arr_t data,\n\tscore_arr_t score);'
            # Remove some lines
            elif ('result_t' in line) or ('unsigned short' in line):
                newline = ''
            else:
                newline = line
            fout.write(newline)

        f.close()
        fout.close()

        #######################
        ## myproject_test.cpp
        #######################

        fout = open('{}/{}_test.cpp'.format(yamlConfig['OutputDir'], yamlConfig['ProjectName']),'w')

        fout.write('#include "BDT.h"\n')
        fout.write('#include "firmware/parameters.h"\n')
        fout.write('#include "firmware/{}.h"\n'.format(yamlConfig['ProjectName']))

        fout.write('int main(){\n')
        fout.write('\tinput_arr_t x = {{{}}};\n'.format(str([0] * ensemble_dict['n_features'])[1:-1]));
        fout.write('\tscore_arr_t score;\n')
        fout.write('\t{}(x, score);\n'.format(yamlConfig['ProjectName']))
        fout.write('\tfor(int i = 0; i < n_classes; i++){\n')
        fout.write('\t\tstd::cout << score[i] << ", ";\n\t}\n')
        fout.write('\tstd::cout << std::endl;\n')
        fout.write('\treturn 0;\n}')
        fout.close()

        fout.close()

        #######################
        ## build_prj.tcl
        #######################

        bdtdir = os.path.abspath(os.path.join(filedir, "../bdt_utils"))
        relpath = os.path.relpath(bdtdir, start=yamlConfig['OutputDir'])

        f = open(os.path.join(filedir,'../hls-template/build_prj.tcl'),'r')
        fout = open('{}/build_prj.tcl'.format(yamlConfig['OutputDir']),'w')

        for line in f.readlines():

                line = line.replace('nnet_utils', relpath)
                line = line.replace('myproject', yamlConfig['ProjectName'])

                #if 'set_top' in line:
                #        line = line.replace('myproject', '{}_decision_function'.format(yamlConfig['ProjectName']))
                if 'set_part {xc7vx690tffg1927-2}' in line:
                        line = 'set_part {{{}}}\n'.format(yamlConfig['XilinxPart'])
                elif 'create_clock -period 5 -name default' in line:
                        line = 'create_clock -period {} -name default\n'.format(yamlConfig['ClockPeriod'])
                # Remove some lines
                elif ('weights' in line) or ('-tb firmware/weights' in line):
                        line = ''
                elif ('cosim_design' in line):
                        line = ''

                fout.write(line)
        f.close()
        fout.close()

class BDTWriterHDL(Writer):

    def write_hls(self, model):
        #######################################
        ## Print a BDT to VHDL
        #######################################
        #def bdt_writer_vhd(ensembleDict, yamlConfig):

        array_cast_text = """        constant value : tyArray2D(nTrees-1 downto 0)(nNodes-1 downto 0) := to_tyArray2D(value_int);
                constant threshold : txArray2D(nTrees-1 downto 0)(nNodes-1 downto 0) := to_txArray2D(threshold_int);"""

        filedir = os.path.dirname(os.path.abspath(__file__))

        dtype = yamlConfig['DefaultPrecision']
        if not 'ap_fixed' in dtype:
            print("Only ap_fixed is currently supported, exiting")
            sys.exit()
        dtype = dtype.replace('ap_fixed<', '').replace('>', '')
        dtype_n = int(dtype.split(',')[0].strip()) # total number of bits
        dtype_int = int(dtype.split(',')[1].strip()) # number of integer bits
        dtype_frac = dtype_n - dtype_int # number of fractional bits
        mult = 2**dtype_frac

        # binary classification only uses one set of trees
        n_classes = 1 if ensembleDict['n_classes'] == 2 else ensembleDict['n_classes']

        #######################################
        ## Write the tree constants
        #######################################

        fout = open('{}/firmware/Parameters.vhd'.format(yamlConfig['OutputDir']), 'w')
        fout.write('package Parameters is\n\n')
        #fout.write('    constant initPredict : txArray')
        fout.write('    constant TreeData : TreeDataArray2D := ((\n')
        # loop over trees
        for itree, trees in enumerate(ensemble_dict['trees']):
            fout.write('\t\t( -- trees[' + str(itree) + ']\n')
            # loop over classes
            for iclass, tree in enumerate(trees):
                fout.write('\t\t\t( -- [' + str(iclass) + ']\n')
                # loop over fields
                for ifield, field in enumerate(tree_fields):
                    newline = '\t\t\t\t('
                    newline += ','.join(map(str, tree[field]))
                    newline += ')'
                    if ifield < len(tree_fields) - 1:
                        newline += ','
                    newline += '\n'
                    fout.write(newline)
                newline = '\t\t\t)'
                if iclass < len(trees) - 1:
                        newline += ','
                newline += '\n'
                fout.write(newline)
            newline = '\t\t)'
            if itree < ensemble_dict['n_trees'] - 1:
                newline += ','
            newline += '\n'
            fout.write(newline)
        fout.write('\t)\n);')
        fout.write('end package Parameters;')

        f = open(os.path.join(filedir,'../templates/bdt_vhdl/run_bdt_test.sh'),'r')
        fout = open('{}/run_bdt_test.sh'.format(yamlConfig['OutputDir']),'w')
        for line in f.readlines():
            if 'insert arrays' in line:
                for i in range(n_classes):
                    newline = 'vcom -2008 -work BDT ./firmware/Arrays{}.vhd\n'.format(i)
                    fout.write(newline)
            else:
                fout.write(line)
        f.close()
        fout.close()

        f = open('{}/test.tcl'.format(yamlConfig['OutputDir']), 'w')
        f.write('vsim -L BDT -L xil_defaultlib xil_defaultlib.testbench\n')
        f.write('run 100 ns\n')
        f.write('quit -f\n')
        f.close()

        f = open('{}/SimulationInput.txt'.format(yamlConfig['OutputDir']), 'w')
        f.write(' '.join(map(str, [0] * ensembleDict['n_features'])))
        f.close()

        f = open(os.path.join(filedir,'../templates/bdt_vhdl/synth.tcl'),'r')
        fout = open('{}/synth.tcl'.format(yamlConfig['OutputDir']), 'w')
        for line in f.readlines():
            if 'hls4ml' in line:
                newline = "synth_design -top BDTTop -part {}\n".format(yamlConfig['XilinxPart'])
                fout.write(newline)
            else:
                fout.write(line)
        f.close()
        fout.close()

        f = open(os.path.join(filedir,'../templates/bdt_vhdl/BDTTop.vhd'),'r')
        fout = open('{}/firmware/BDTTop.vhd'.format(yamlConfig['OutputDir']),'w')
        for line in f.readlines():
            fout.write(line)
        f.close()
        fout.close()

        f = open(os.path.join(filedir, '../templates/bdt_vhdl/Constants.vhd'), 'r')
        fout = open('{}/firmware/Constants.vhd'.format(yamlConfig['OutputDir']), 'w')
        for line in f.readlines():
            if 'hls4ml' in line:
                newline = "  constant nTrees : integer := {};\n".format(ensembleDict['n_trees'])
                newline += "    constant maxDepth : integer := {};\n".format(ensembleDict['max_depth'])
                newline +=  "  constant nNodes : integer := {};\n".format(2 ** (ensembleDict['max_depth'] + 1) - 1)
                newline += "    constant nLeaves : integer := {};\n".format(2 ** ensembleDict['max_depth'])
                newline += "    constant nFeatures : integer := {};\n".format(ensembleDict['n_features'])
                newline += "    constant nClasses : integer := {};\n\n".format(n_classes)
                newline += "    subtype tx is signed({} downto 0);\n".format(dtype_n - 1)
                newline += "    subtype ty is signed({} downto 0);\n".format(dtype_n - 1)
                fout.write(newline)
            else:
                fout.write(line)
        f.close()
        fout.close()


