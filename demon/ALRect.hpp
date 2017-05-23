//
//  ALRect.hpp
//  shintao-recognize
//
//  Created by develop on 2017/5/23.
//  Copyright © 2017年 develop. All rights reserved.
//

#ifndef ALRect_hpp
#define ALRect_hpp

#include <stdio.h>

class ALRect {
public:
    ALRect();
    ALRect(int, int, int, int, int);
    
    int _ltx;
    int _lty;
    int _rdx;
    int _rdy;
    int _cx;
    int _cy;
    int _width;
    int _height;
    int _count;
    
    void SetLtx(int);
    void SetLty(int);
    void SetRdx(int);
    void SetRdy(int);
    void AddCount(int);
private:
    
};

#endif /* ALRect_hpp */
