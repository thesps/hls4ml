#include "myproject_axi.h"

void myproject(
    input_axi_t in[N_IN],
    output_axi_t out[N_OUT]
        ){

    #pragma HLS INTERFACE s_axilite port=return bundle=CTRL_BUS
    #pragma HLS INTERFACE m_axi depth=N_IN port=in offset=slave bundle=IN_BUS
    #pragma HLS INTERFACE m_axi depth=N_OUT port=out offset=slave bundle=OUT_BUS

    unsigned short in_size = 0;
    unsigned short out_size = 0;

    //hls-fpga-machine-learning insert local vars

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
