#include <Eigen/StdVector>
#include <iostream>
#include <stdint.h>

#include <unordered_set>


#include "g2o/core/sparse_optimizer.h"
#include "g2o/core/block_solver.h"
#include "g2o/core/solver.h"
#include "g2o/core/robust_kernel_impl.h"
#include "g2o/core/optimization_algorithm_levenberg.h"
#include "g2o/solvers/eigen/linear_solver_eigen.h"
#include "g2o/solvers/dense/linear_solver_dense.h"

#include "g2o/solvers/csparse/linear_solver_csparse.h"
#include "g2o/solvers/cholmod/linear_solver_cholmod.h"
#include "g2o/types/sba/types_six_dof_expmap.h"
#include "g2o/types/sim3/types_seven_dof_expmap.h"
#include "g2o/types/sim3/sim3.h"
#include "g2o/types/slam3d/edge_pointxyz.h"
#include "edge/edge_unary_pointxyz.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
//
using namespace std;
namespace py=pybind11;
// make a short name for vector of vector
using vvd = std::vector<std::vector<double>>;

class EgoMotionCorrection{
    g2o::SparseOptimizer optimizer;
    // Pinhole camera intrinsics 
       double cx_=0, cy_=0, fx_=0, fy_=0; 
       //road function para  av^2+bv-u+c = 0 >0 mean right  <0 mean left
       double fa_=0,fb_=0,fc_=0;
        
       // car information width and height 
       double wc_=0,hc_=0;

    public:
    // set optimizer in construct function
    EgoMotionCorrection(){}
    EgoMotionCorrection(double fx,double fy,double cx,double cy,double wc,double
    hc):fx_(fx),fy_(fy),cx_(cx),cy_(cy),wc_(wc),hc_(hc)
    {
        optimizer.setVerbose(false);
        std::unique_ptr<g2o::BlockSolver_3_2::LinearSolverType> linearSolver;
        bool DENSE = false;
        if (DENSE)
        {
            linearSolver = g2o::make_unique<g2o::LinearSolverDense<g2o::BlockSolver_3_2::PoseMatrixType>>();
        } else
        {
            linearSolver = g2o::make_unique<g2o::LinearSolverCholmod<g2o::BlockSolver_3_2::PoseMatrixType>>();
        }

        g2o::OptimizationAlgorithmLevenberg* solver = new g2o::OptimizationAlgorithmLevenberg(
            g2o::make_unique<g2o::BlockSolver_3_2>(std::move(linearSolver))
            );
        optimizer.setAlgorithm(solver);
    }
    /*
    @function predict: correct path by considering road edge
    @input trans shape [n,3] n position,relative to the first frame
    @input fa,fb,fc, line function f(u,v)=fa*v*v+fb*v+c-u
    @return corrected position shape is [n,3]
    */
    vvd correct(vvd trans,vector<double> lamdas ,double fa,double fb,double fc)
    {

        Eigen::Vector3d t = Eigen::Vector3d(trans[0].data());
        g2o::VertexPointXYZ *vP = new g2o::VertexPointXYZ();
        vP->setEstimate(t);
        vP->setId(0);
        vP->setFixed(true);
        optimizer.addVertex(vP);
        const Eigen::Matrix<double,1,1> matLambda = Eigen::Matrix<double,1,1>::Identity();
        // add vertices
        for(int i=1; i<trans.size(); i++)
        {
            Eigen::Vector3d t = Eigen::Vector3d(trans[i].data());
            g2o::VertexPointXYZ *vP = new g2o::VertexPointXYZ();
            vP->setEstimate(t);
            vP->setId(i);
            vP->setFixed(false);
            optimizer.addVertex(vP);
        
        }

        // add edge
        // unary edge to keep vehicle on road
        for(int i = 1;i<trans.size();i++)    {
            g2o::EdgeUnaryPointXYZ *e = new g2o::EdgeUnaryPointXYZ(cx_,cy_,fx_,fy_,fa,fb,fc,wc_,hc_);
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(i)));
            e->information() = matLambda;
            optimizer.addEdge(e);
        }
        // add edge
        // binary edge to consider original motion
       const Eigen::Matrix<double,3,3> matLambda3 = Eigen::Matrix<double,3,3>::Identity();
       for(int i = 1;i<trans.size();i++)    {
            g2o::EdgePointXYZ *e = new g2o::EdgePointXYZ();
            e->setVertex(0, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(i-1)));
            e->setVertex(1, dynamic_cast<g2o::OptimizableGraph::Vertex*>(optimizer.vertex(i)));
            e->information() = lamdas[i]*matLambda3;
            Eigen::Vector3d me; 
            for(int j=0;j<3;j++){me[j] = trans[i][j]-trans[i-1][j];}
            e->setMeasurement(me);
            optimizer.addEdge(e);
        }
        optimizer.setVerbose(true);
        optimizer.initializeOptimization();
        vvd result;
        optimizer.optimize(20);
        for(int i = 0;i<trans.size();i++)    {
            g2o::VertexPointXYZ *vP = static_cast<g2o::VertexPointXYZ*>(optimizer.vertex(i));
            Eigen::Vector3d t =  vP->estimate();
            std::vector<double> re;
            re.push_back(t[0]);
            re.push_back(t[1]);
            re.push_back(t[2]);
            result.push_back(re);
        }
            optimizer.edges().clear();
            optimizer.vertices().clear();
        return result;
    }
};

PYBIND11_MODULE(vocorrection,m)
{   
    py::class_<EgoMotionCorrection>(m,"EgoMotionCorrection")
        .def(py::init<>())
        .def(py::init<double,double ,double ,double,double ,double>())
        .def("correct",&EgoMotionCorrection::correct);
}

//demo
int main(int argc, char** argv)
{
    EgoMotionCorrection * vo_predict = new EgoMotionCorrection();
    vvd quats;
    vvd trans;
    vector<double> lamdas;
    for(int i =0;i<10;i++)
    {
        std::vector<double> t;
        t.push_back(0);
        t.push_back(1.7);
        t.push_back(i+5);
        trans.push_back(t);
        lamdas.push_back(i);
    }
    vvd res = vo_predict->correct(trans,lamdas,0.0,1.0,-80.0);
    return 0;
}
