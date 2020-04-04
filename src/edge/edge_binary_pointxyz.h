#ifndef G2O_EDGE_BINARY_POINTXYZ_H_XW
#define G2O_EDGE_BINARY_POINTXYZ_H_XW

#include "g2o/types/slam3d/vertex_pointxyz.h"
#include "g2o/config.h"
#include "g2o/core/base_binary_edge.h"
#include "g2o/types/slam3d/g2o_types_slam3d_api.h"

namespace g2o {

  class G2O_TYPES_SLAM3D_API EdgeBinaryPointXYZ : public BaseBinaryEdge<1, double, VertexPointXYZ,VertexPointXYZ>
  {
    public:
      EIGEN_MAKE_ALIGNED_OPERATOR_NEW
        EdgeBinaryPointXYZ(){}
        EdgeBinaryPointXYZ(double cx, double cy, double fx,double fy,double fa,double fb,double fc):cx_(cx),cy_(cy),fx_(fx),fy_(fy),fa_(fa),fb_(fb),fc_(fc)
      {}

      virtual void computeError();
      virtual void linearizeOplus( ); 
      virtual bool read(std::istream& is);
      virtual bool write(std::ostream& os) const;
       // Pinhole camera intrinsics 
       double cx_=0, cy_=0, fx_=0, fy_=0; 
       //road function para  av^2+bv-u+c = 0 >0 mean left <0 mean right
       double fa_=0,fb_=0,fc_=0;

  };


} // end namespace

#endif
