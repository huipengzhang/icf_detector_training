#ifndef DATA_TYPES_HPP
#define DATA_TYPES_HPP
#include <cmath>
#include "icf_detector.pb.h"




struct SearchParam
{
    float minput_image_scale;
    float mmodel_scale;
    int mstart_x;
    int mstart_y;
    int mend_x;
    int mend_y;
    int mstride_x;
    int mstride_y;

    ICFDetector::StrongModel mmodel;
};











struct DetectResult
{
    DetectResult():mx(-1),my(-1),mw(-1),mh(-1),mscore(-1),mimgscale(-1),mmodelscale(-1){}

    int mx;
    int my;
    int mw;
    int mh;
    float mscore;
    float mimgscale;
    float mmodelscale;
};

inline bool DetectResultLarger(const DetectResult& a, const DetectResult&b)
{
//    return a.mw>b.mw;
    return a.mscore>b.mscore;
}


inline bool isOverlap(float ax0, float ay0,float ax1, float ay1,float bx0, float by0,float bx1,float by1,float threshold=0.5)
{
    float x0,x1,y0,y1,iw,ih;
    float inter_area,union_area;

    x0 = std::max(ax0,bx0);
    x1 = std::min(ax1,bx1);
    y0 = std::max(ay0,by0);
    y1 = std::min(ay1,by1);

    iw = (x1-x0>0)?(x1-x0):0;
    ih = (y1-y0>0)?(y1-y0):0;

    inter_area = iw*ih;
    union_area = (ay1-ay0)*(ax1-ax0)+(by1-by0)*(bx1-bx0)-inter_area;

    return (inter_area/union_area)>threshold?true:false;
}

inline bool isOverlapMin(float ax0, float ay0,float ax1, float ay1,float bx0, float by0,float bx1,float by1,float threshold=0.5)
{
    float x0,x1,y0,y1,iw,ih;
    float inter_area,min_area;
    x0 = std::max(ax0,bx0);
    x1 = std::min(ax1,bx1);
    y0 = std::max(ay0,by0);
    y1 = std::min(ay1,by1);

    iw = (x1-x0>0)?(x1-x0):0;
    ih = (y1-y0>0)?(y1-y0):0;


    inter_area = iw*ih;
    min_area = std::min((ay1-ay0)*(ax1-ax0),(by1-by0)*(bx1-bx0));

    return (inter_area/min_area)>threshold?true:false;
}

inline void detectResultScaleTo1(DetectResult& p)
{
    p.mx = static_cast<int>(round(p.mx/p.mimgscale));
    p.my = static_cast<int>(round(p.my/p.mimgscale));
    p.mw = static_cast<int>(round(p.mw/p.mimgscale));
    p.mh = static_cast<int>(round(p.mh/p.mimgscale));
    p.mimgscale = 1;
    return;
}

#endif
