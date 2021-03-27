#!/bin/bash

# We run all the tests, in the basic version (no extensive -test testing even if available)
# Therefore this can be inaccurate

echo "!!!!!!!!! Non extensive tests !!!!!!!!!!!!!!!!!!!"
PYTHON_BINARY="${PYTHON_BINARY:-python3}"

ERRORS=0
FAILED_TESTS=""
TESTS=0

bail() {
    ERRORSTR=$1
    /bin/echo -e "${RED}ERROR${NC} in $ERRORSTR" 1>&2
    ERRORS=`expr $ERRORS + 1`
    FAILED_TESTS="${FAILED_TESTS} $ERRORSTR\n"
}


tests=("test_relu_fpga" "test_gemm_fpga" "test_im2col_conv2d_fpga" "test_matmul_fpga"
        "test_maxpool2d_fpga" "test_reduce_sum_fpga" "test_reshape_fpga" "test_softmax_fpga" "test_streaming_conv_relu_mp")



for i in "${tests[@]}"
do
    TESTS=`expr $TESTS + 1`
    echo "################# Executing test $i #################"
    timeout 500s ${PYTHON_BINARY} $i.py
    if [ $? -ne 0 ]; then
      bail "$i"
    fi
done



PASSED=`expr $TESTS - $ERRORS`
echo "$PASSED / $TESTS tests passed"
if [ $ERRORS -ne 0 ]; then
    printf "Failed tests:\n${FAILED_TESTS}"
    exit 1
fi