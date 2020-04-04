#include "edge/edge_binary_pointxyz.h"
#include <Eigen/StdVector>
#include <Eigen/Geometry>
#include <math.h>
#include <iostream>
namespace g2o{
void EdgeBinaryPointXYZ::computeError()
{
    const VertexPointXYZ* vertex_x  =static_cast<const VertexPointXYZ*> ( _vertices[0] );
    const VertexPointXYZ* vertex_y  =static_cast<const VertexPointXYZ*> ( _vertices[1] );
    
    Eigen::Vector3d posi_x = vertex_x->estimate();
    Eigen::Vector3d posi_y = vertex_y->estimate();
    double u_x = fx_*posi_x[0]/posi_x[2]+cx_;
    double v_x = fy_*posi_x[1]/posi_x[2]+cy_;
    double u_y = fx_*posi_y[0]/posi_y[2]+cx_;
    double v_y = fy_*posi_y[1]/posi_y[2]+cy_;

    double d_x = -(fa_*v_x*v_x+fb_*v_x+fc_ - u_x);
    double scale_x = 1.75/(v_x-cy_+0.012);
    d_x = d_x*scale_x;
    
    double d_y = -(fa_*v_y*v_y+fb_*v_y+fc_ - u_y);
    double scale_y = 1.75/(v_y-cy_+0.012);
    d_y = d_y*scale_y;

    double loss = (d_y-d_x)*(d_y-d_x);
    std::cout<<"scale:  "<<scale_x<<" "<<scale_y<<" "<<loss<<std::endl;
    //std::cout<<"dx  "<<d_x<<"dy  "<<d_y<<"loss  "<<loss<<std::endl;
    _error(0,0) = loss;
}

void EdgeBinaryPointXYZ::linearizeOplus()
{
    Eigen::Matrix<double, 1, 3> jacobian_l_X;
    Eigen::Matrix<double, 1, 3> jacobian_l_Y;
    const VertexPointXYZ* vertex_x  =static_cast<const VertexPointXYZ*> ( _vertices[0] );
    const VertexPointXYZ* vertex_y  =static_cast<const VertexPointXYZ*> ( _vertices[1] );
    
    Eigen::Vector3d posi_x = vertex_x->estimate();
    Eigen::Vector3d posi_y = vertex_y->estimate();
    double u_x = fx_*posi_x[0]/posi_x[2]+cx_;
    double v_x = fy_*posi_x[1]/posi_x[2]+cy_;
    double u_y = fx_*posi_y[0]/posi_y[2]+cx_;
    double v_y = fy_*posi_y[1]/posi_y[2]+cy_;

    double d_x = -(fa_*v_x*v_x+fb_*v_x+fc_ - u_x);
    double scale_x = 1.75/(v_x-cy_+0.012);
    d_x = d_x*scale_x;
    
    double d_y = -(fa_*v_y*v_y+fb_*v_y+fc_ - u_y);
    double scale_y = 1.75/(v_y-cy_+0.012);
    d_y = d_y*scale_y;

    double d = (d_x-d_y);

    Eigen::Matrix<double,2,3> jacobian_U_X;
    jacobian_U_X(0,0) = fx_/posi_x[2];
    jacobian_U_X(0,2) = -fx_*posi_x[0]/(posi_x[2]*posi_x[2]);
    jacobian_U_X(1,0) = fy_/posi_x[2];
    jacobian_U_X(1,2) = -fy_*posi_x[1]/(posi_x[2]*posi_x[2]);
    jacobian_U_X(0,1) = 0;
    jacobian_U_X(1,0) = 0;

    //std::cout<<"gradient UX"<<jacobian_U_X<<std::endl;
    Eigen::Matrix<double,1,2> jacobian_l_U;
    double scale = 1.75/(v_x-cy_+10.12);
    jacobian_l_U(0,1) =(-2*fa_*v_x+-fb_)*scale + d_x*(-scale*scale/1.75);
    jacobian_l_U(0,0) =1*scale;
    //std::cout<<"gradient lU"<<jacobian_l_U<<std::endl;
    jacobian_l_X = jacobian_l_U*jacobian_U_X;
    //std::cout<<"gradient lX"<<jacobian_l_Y<<std::endl;
    double jacobian_l = 2*d;

    _jacobianOplusXi = jacobian_l*jacobian_l_Y;

    Eigen::Matrix<double,2,3> jacobian_U_Y;
    jacobian_U_Y(0,0) = fx_/posi_y[2];
    jacobian_U_Y(0,2) = -fx_*posi_y[0]/(posi_y[2]*posi_y[2]);
    jacobian_U_Y(1,0) = fy_/posi_y[2];
    jacobian_U_Y(1,2) = -fy_*posi_y[1]/(posi_y[2]*posi_y[2]);
    jacobian_U_Y(0,1) = 0;
    jacobian_U_Y(1,0) = 0;

    //std::cout<<"gradient UX"<<jacobian_U_Y<<std::endl;
    scale = 1.75/(v_y-cy_+10.12);
    jacobian_l_U(0,1) =(-2*fa_*v_y+-fb_)*scale + d_y*(-scale*scale/1.75);
    jacobian_l_U(0,0) =1*scale;
    //std::cout<<"gradient lU"<<jacobian_l_U<<std::endl;
    jacobian_l_Y = jacobian_l_U*jacobian_U_Y;
    //std::cout<<"gradient lX"<<jacobian_l_Y<<std::endl;
    jacobian_l = -2*d;

    _jacobianOplusXj = jacobian_l*jacobian_l_Y;

       

    //std::cout<<"gradient "<<_jacobianOplusXi<<std::endl;

}

bool EdgeBinaryPointXYZ::read( std::istream& in )
{
    return true;
}
bool EdgeBinaryPointXYZ::write( std::ostream& out ) const
{
    return true;
}
}//end namespace g2o
