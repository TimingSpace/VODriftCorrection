#ifndef G2O_EDGE_UNARY_POINTXYZ_H_XW
#define G2O_EDGE_UNARY_POINTXYZ_H_XW

#include "g2o/types/slam3d/vertex_pointxyz.h"
#include "g2o/config.h"
#include "g2o/core/base_unary_edge.h"
#include "g2o/types/slam3d/g2o_types_slam3d_api.h"

namespace g2o {

  class G2O_TYPES_SLAM3D_API EdgeUnaryPointXYZ : public BaseUnaryEdge<1, double, VertexPointXYZ>
  {
    public:
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        EdgeUnaryPointXYZ(){}
        EdgeUnaryPointXYZ(double cx, double cy, double fx,double fy,double fa,double fb,double fc,double wc,double
        hc):cx_(cx),cy_(cy),fx_(fx),fy_(fy),fa_(fa),fb_(fb),fc_(fc),wc_(wc),hc_(hc)
      {}

      virtual void computeError();
      virtual void linearizeOplus( ); 
      virtual bool read(std::istream& is);
      virtual bool write(std::ostream& os) const;
       // Pinhole camera intrinsics 
       double cx_=0, cy_=0, fx_=0, fy_=0; 
       //road function para  av^2+bv-u+c = 0 >0 mean right  <0 mean left
       double fa_=0,fb_=0,fc_=0;
        
       // car information width and height 
       double wc_=0,hc_=0;
  };


} // end namespace

#endif
