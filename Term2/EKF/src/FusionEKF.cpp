#include "FusionEKF.h"
#include "tools.h"
#include "math.h"
#include "Eigen/Dense"
#include <iostream>
#include "kalman_filter.h"

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/*
 * Constructor.
 */
FusionEKF::FusionEKF() {
  is_initialized_ = false;
  previous_timestamp_ = 0;

  // initializing matrices
  R_laser_ = MatrixXd(2, 2);
  R_radar_ = MatrixXd(3, 3);
  H_laser_ = MatrixXd(2, 4);
  Hj_ = MatrixXd(3, 4);

  //measurement covariance matrix - laser
  R_laser_ << 0.0225, 0,
        0, 0.0225;

  //measurement covariance matrix - radar
  R_radar_ << 0.09, 0, 0,
        0, 0.0009, 0,
        0, 0, 0.09;

  // Initializing P
  ekf_.P_ = MatrixXd(4, 4);
  ekf_.P_ << 1, 0, 0, 0,
             0, 1, 0, 0,
             0, 0, 1000, 0,
             0, 0, 0, 1000;

  H_laser_ << 1, 0, 0, 0,
              0, 1, 0, 0;
}

/**
* Destructor.
*/
FusionEKF::~FusionEKF() {}

void FusionEKF::ProcessMeasurement(const MeasurementPackage &measurement_pack) {
  /*****************************************************************************
   *  Initialization
   ****************************************************************************/

   if (!is_initialized_) {
    /**
    TODO:
      * Initialize the state ekf_.x_ with the first measurement.
      * Create the covariance matrix.
      * Remember: you'll need to convert radar from polar to cartesian coordinates.
    */

    cout << "EKF: " << endl;
    ekf_.x_ = VectorXd(4);
    ekf_.x_ << 1, 1, 1, 1;

    // first measurement
    if(measurement_pack.sensor_type_ == MeasurementPackage::RADAR)
    {
    	ekf_.x_=tools.Polar2Cart(measurement_pack.raw_measurements_);
    }
    else if (measurement_pack.sensor_type_ == MeasurementPackage::LASER)
    {
    	ekf_.x_ << measurement_pack.raw_measurements_[0], measurement_pack.raw_measurements_[1], 0, 0;
    }
    if (fabs(ekf_.x_(0)) < EPS and fabs(ekf_.x_(1)) < EPS){
		ekf_.x_(0) = EPS;
		ekf_.x_(1) = EPS;
	}

    previous_timestamp_ = measurement_pack.timestamp_;

    is_initialized_ = true;
    return;
  }

  /*****************************************************************************
   *  Prediction
   ****************************************************************************/

	double dt = (measurement_pack.timestamp_ - previous_timestamp_) / 1000000.0;	//dt - expressed in seconds
	previous_timestamp_ = measurement_pack.timestamp_;

	double dt_2 = dt * dt;
	double dt_3 = dt_2 * dt;
	double dt_4 = dt_3 * dt;
	double  noise_ax = 9;
	double  noise_ay = 9;

	//Modify the F matrix so that the time is integrated
	 // State transition matrix update
     ekf_.F_ = MatrixXd(4, 4);
     ekf_.F_ << 1, 0, dt, 0,
			 0, 1, 0, dt,
			 0, 0, 1, 0,
			 0, 0, 0, 1;

	//set the process covariance matrix Q
    ekf_.Q_ = MatrixXd(4, 4);

	ekf_.Q_ <<  dt_4/4*noise_ax, 0, dt_3/2*noise_ax, 0,
			   0, dt_4/4*noise_ay, 0, dt_3/2*noise_ay,
			   dt_3/2*noise_ax, 0, dt_2*noise_ax, 0,
			   0, dt_3/2*noise_ay, 0, dt_2*noise_ay;

  ekf_.Predict();

  /*****************************************************************************
   *  Update
   ****************************************************************************/

  if (measurement_pack.sensor_type_ == MeasurementPackage::RADAR) {
	  ekf_.H_ = tools.CalculateJacobian(ekf_.x_);
	  ekf_.R_=R_radar_;
    ekf_.UpdateEKF(measurement_pack.raw_measurements_);
  } else {
    // Laser updates
	  ekf_.R_=R_laser_;
	  ekf_.H_=H_laser_;
	  ekf_.Update(measurement_pack.raw_measurements_);
  }

  // print the output
  cout << "x_ = " << ekf_.x_ << endl;
  cout << "P_ = " << ekf_.P_ << endl;
}
