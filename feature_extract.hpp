#ifndef FEATURE_EXTRACT_HPP
#define FEATURE_EXTRACT_HPP

#include <vector>

class FeatureExtract
{
public:
    FeatureExtract(){};
    virtual ~FeatureExtract(){};

    virtual void setImage(const unsigned char* img_begin, int width, int height, int flags) = 0;
    virtual void compute() = 0;
    virtual const std::vector<double>& getFeatures() const = 0;


};


#endif


