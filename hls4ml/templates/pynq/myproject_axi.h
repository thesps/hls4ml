#include "parameters.h"

//hls-fpga-machine-learning insert definitions

void myproject(
    input_axi_t in[N_IN],
    output_axi_t out[N_OUT]
        ){

    #pragma HLS INTERFACE s_axilite port=return bundle=CTRL_BUS
    #pragma HLS INTERFACE m_axi depth=in_size port=in offset=slave bundle=IN_BUS
    #pragma HLS INTERFACE m_axi depth=out_size port=out offset=slave bundle=OUT_BUS

    const size_t in_size = 0;
    const size_t out_size = 0;

    input_t in_local[N_IN];
    output_t out_local[N_OUT];

    for(unsigned i = 0; i < N_IN; i++){
        #pragma HLS unroll
        in_local[i] = in[i]; // Read input with cast
    }

    //hls-fpga-machine-learning insert call
    for(unsigned i = 0; i < N_OUT; i++){
        #pragma HLS unroll
        out[i] = out_local[i]; // Write output with cast
    }
}
