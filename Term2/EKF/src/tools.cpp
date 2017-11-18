#include <iostream>
#include <math.h>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth) {
  /**
  TODO:
    * Calculate the RMSE here.
  */

	VectorXd rmse(4);
		rmse << 0,0,0,0;

		// check the validity of the following inputs:
		//  * the estimation vector size should not be zero
		//  * the estimation vector size should equal ground truth vector size
		if(estimations.size() != ground_truth.size()
				|| estimations.size() == 0){
			cout << "Invalid estimation or ground_truth data" << endl;
			return rmse;
		}

		//accumulate squared residuals
		for(unsigned int i=0; i < estimations.size(); ++i){

			VectorXd residual = estimations[i] - ground_truth[i];

			//coefficient-wise multiplication
			residual = residual.array()*residual.array();
			rmse += residual;
		}

		//calculate the mean
		rmse = rmse/estimations.size();

		//calculate the squared root
		rmse = rmse.array().sqrt();

		//return the result
		return rmse;
}

MatrixXd Tools::CalculateJacobian(const VectorXd& x_state) {

	MatrixXd Hj(3,4);

    if ( x_state.size() != 4 ) {
	  cout << "ERROR - CalculateJacobian () - The state vector must have size 4." << endl;
	  return Hj;
    }
	//recover state parameters
	float px = x_state(0);
	float py = x_state(1);
	float vx = x_state(2);
	float vy = x_state(3);

	//pre-compute a set of terms to avoid repeated calculation
	float c1 = px*px+py*py;
	float c2 = sqrt(c1);
	float c3 = (c1*c2);

	//check division by zero
	if(fabs(c1) < EPS){
		cout << "CalculateJacobian () - Error - Division by Zero" << endl;
		return Hj;
	}

	//compute the Jacobian matrix
	Hj << (px/c2), (py/c2), 0, 0,
		  -(py/c1), (px/c1), 0, 0,
		  py*(vx*py - vy*px)/c3, px*(px*vy - py*vx)/c3, px/c2, py/c2;

	return Hj;
}



VectorXd Tools::Polar2Cart(const VectorXd x_state) {
	VectorXd x_cart(4);

    double rho = x_state[0]; // range
	double phi = x_state[1]; // bearing
	double rho_dot = x_state[2]; // velocity of rho
	  // Coordinates convertion from polar to cartesian
	double x = rho * cos(phi);

    if ( x < EPS ) {
      x = EPS;
    }

	double y = rho * sin(phi);
    if ( y < EPS ) {
      y = EPS;
    }

	double vx = rho_dot * cos(phi);
	double vy = rho_dot * sin(phi);

	x_cart << x, y, vx , vy;

	return x_cart;
}
