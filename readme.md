# Written by Yu Ling in ZJU before 2016

# Prepared Libs

opencv

protobuf


# How to start

Use Commands Below

```
sh generate_proto.sh

mkdir build

cd build

cmake ..

ccmake . #modify CMAKE_BUILD_TYPE to Release

cmake .

make -j4
```
