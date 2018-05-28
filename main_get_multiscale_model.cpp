#include "icf_detector.pb.h"
#include <fstream>
#include <stdexcept>
#include <iostream>
#include <sstream>
using namespace std;

int main(int argn, char* argv[])
{
    ICFDetector::MultiScaleStrongModel outputmodel;
    int shrink_factor=-1;
    bool inited=false;
    for(int i=1;i<argn;i++)
    {
        ICFDetector::StrongModel m1;
        fstream fin(argv[i], ios::in | ios::binary);
        if (!m1.ParseFromIstream(&fin)) {
            throw std::runtime_error("Fail to Parse Model ProtoBuf File");
        }

        if(shrink_factor<0)
            shrink_factor = m1.shrink_factor();

        if(shrink_factor!=m1.shrink_factor())
        {
            cout<<argv[i]<<" shrinkd factor not equal to "<<shrink_factor<<endl;
            throw std::runtime_error("model input shrink factor error");
        }

        outputmodel.set_shrink_factor(shrink_factor);
        if(!inited)
        {
            outputmodel.set_scale0_model_width(m1.model_width());
            outputmodel.set_scale0_model_height(m1.model_height());
            inited = true;
        }
        outputmodel.add_models_width(m1.model_width());
        ICFDetector::StrongModel* m1p = outputmodel.add_strong_models();
        *m1p = m1;
        fin.close();
    }

    stringstream outss;
    outss<<"model_multiscale.proto.bin";
    fstream out(outss.str().c_str(), ios::out | ios::binary | ios::trunc);
    outputmodel.SerializeToOstream(&out);
    out.close();
}
