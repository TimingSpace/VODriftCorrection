/*
@author Xiangwei Wang 
@contact wangxiangwei.cpp@gmail.com
*/

#include "edge/edge_unary_pointxyz.h"
#include <Eigen/StdVector>
#include <Eigen/Geometry>
#include <math.h>
#include <iostream>
namespace g2o{
/*
calculate point error RelU(-f(u,v)+dw) for right side road edge 
*/
void EdgeUnaryPointXYZ::computeError()
{
    const VertexPointXYZ* vertex  =static_cast<const VertexPointXYZ*> ( _vertices[0] );
    
    Eigen::Vector3d posi = vertex->estimate();
    // map 3D point into image by u = KX
    double u = fx_*posi[0]/posi[2]+cx_;
    double v = fy_*posi[1]/posi[2]+cy_;
    // loss is calculated by -f(u,v)+ dw
    double loss = -(fa_*v*v+fb_*v+fc_ - u-(0.5*wc_/(hc_))*(v-cy_));
    // scale for rescale the loss
    double scale = 1;// .75/(v-cy_+0.101023);
    loss = loss*scale;
    double e_loss = loss;
    // ReLU function
    if(loss<0) loss=0;
    _error(0,0) = e_loss;
}
/*calcualte the gradient of loss to posi*/
void EdgeUnaryPointXYZ::linearizeOplus()
{
    ////std::cout<<"test______________"<<std::endl;
    Eigen::Matrix<double, 1, 3> jacobian_l_X;
    const VertexPointXYZ* vertex  =static_cast<const VertexPointXYZ*> ( _vertices[0] );
    
    Eigen::Vector3d posi = vertex->estimate();
    double u = fx_*posi[0]/posi[2]+cx_;
    double v = fy_*posi[1]/posi[2]+cy_;
    double loss = -(fa_*v*v+fb_*v+fc_ - u-(wc_/(2*hc_))*(v-cy_));
    // gradient of (u,v) to (X,Y,Z)
    Eigen::Matrix<double,2,3> jacobian_U_X;
    jacobian_U_X(0,0) = fx_/posi[2];
    jacobian_U_X(0,2) = -fx_*posi[0]/(posi[2]*posi[2]);
    jacobian_U_X(1,1) = 0; // we set gradient to Y is zero, to keep Y unchanged
    jacobian_U_X(1,2) = -fy_*posi[1]/(posi[2]*posi[2]);
    jacobian_U_X(0,1) = 0;
    jacobian_U_X(1,0) = 0;

    //gradient of L to (u,v)
    Eigen::Matrix<double,1,2> jacobian_l_U;
    double scale = 1;//1.75/(v-cy_+0.101023);
    jacobian_l_U(0,1) =(-2*fa_*v+-fb_)+0.5*wc_/hc_;//*scale + loss*(-scale*scale/1.75);
    jacobian_l_U(0,0) =1*scale;
    jacobian_l_X = jacobian_l_U*jacobian_U_X;
    double jacobian_l = 1;
    //gradient of ReLU
    if(loss<0) jacobian_l = 0;

    _jacobianOplusXi = jacobian_l*jacobian_l_X;

}

bool EdgeUnaryPointXYZ::read( std::istream& in )
{
    return true;
}
bool EdgeUnaryPointXYZ::write( std::ostream& out ) const
{
    return true;
}
}//end namespace g2o
