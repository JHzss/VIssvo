//
// Created by jh on 18-10-27.
//

#include "global.hpp"
#include "map_point.hpp"

#ifndef SSVO_FEATUREINWINDOW_H
#define SSVO_FEATUREINWINDOW_H

namespace ssvo{

class FeatureInWindow {
public:
    FeatureInWindow();

public:
    vector<MapPoint::Ptr> mapPointsInWindow;

};


}




#endif //SSVO_FEATUREINWINDOW_H
