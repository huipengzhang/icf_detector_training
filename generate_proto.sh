#!/bin/sh

echo "regenerate the .h and .cc file with proto file."
protoc -I=./ --cpp_out=./ ./icf_detector.proto
