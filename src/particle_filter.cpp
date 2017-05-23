/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h>

#include "particle_filter.h"

std::default_random_engine gen;

// Calculate Multivariate-Gaussian probability 
double MGP(double x, 
	double y, 
	double mx, 
	double my,
	double sig_x,
	double sig_y) {

	double p1 = 1/ (2 * M_PI * sig_x * sig_y);
	double p2x = ((x - mx) * (x - mx)) / (2 * sig_x * sig_x);
	double p2y = ((y - my) * (y - my)) / (2 * sig_y * sig_y);
	double p2 = -1 * (p2x + p2y);
	double p3 = exp(p2);
	return p1 * p3;
}

void ParticleFilter::init(double x, 
	double y, 
	double theta, 
	double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1.
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

	num_particles = 1000; 
	weights = std::vector<double>(num_particles);
	
	// Set of current particles
	particles = std::vector<Particle>(num_particles);	
	
	// Create normal (Gaussian) distribution for x, y and theta
	std::normal_distribution<double> dist_x(x, std[0]);
	std::normal_distribution<double> dist_y(y, std[1]);
	std::normal_distribution<double> dist_theta(theta, std[2]);
	
	for (int i = 0; i < num_particles; i++) {
		particles[i].id = i;
		particles[i].x = dist_x(gen);
		particles[i].y = dist_y(gen);
		particles[i].theta = dist_theta(gen);
		particles[i].weight = 1.0;
		weights[i] = 1.0;
	}

	// Flag initialized
	is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, 
	double std_pos[], 
	double velocity, 
	double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

	for (int p = 0; p < particles.size(); p++) {
		if (fabs(yaw_rate) < 1e-6) {
			particles[p].x = particles[p].x + 
				velocity * cos(particles[p].theta) * delta_t;
			particles[p].y = particles[p].y + 
				velocity * sin(particles[p].theta) * delta_t;
			particles[p].theta = particles[p].theta;
		} else {
			particles[p].x = particles[p].x + 
				(velocity / yaw_rate) * (sin(particles[p].theta + 
				yaw_rate * delta_t) - sin(particles[p].theta));
			particles[p].y = particles[p].y + 
				velocity / yaw_rate * (cos(particles[p].theta) - 
				cos(particles[p].theta + yaw_rate * delta_t));
			particles[p].theta = particles[p].theta + 
				yaw_rate * delta_t;
		}

		std::normal_distribution<double> dist_x(particles[p].x, std_pos[0]);
		std::normal_distribution<double> dist_y(particles[p].y, std_pos[1]);
		std::normal_distribution<double> dist_theta(particles[p].theta, std_pos[2]);

		particles[p].x = dist_x(gen);
		particles[p].y = dist_y(gen);
		particles[p].theta = dist_theta(gen);
	}
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.

	for (int ob=0; ob < observations.size(); ob++) {
		double min = std::numeric_limits<double>::max();
		for (int p=0; p < predicted.size(); p++) {
			double distance = dist(observations[ob].x, 
				observations[ob].y, 
				predicted[p].x, 
				predicted[p].y);
			if (distance < min) {
				observations[ob].id = predicted[p].id;
				min = distance;
			}
		}
	}
}

void ParticleFilter::updateWeights(double sensor_range, 
	double std_landmark[],
	std::vector<LandmarkObs> observations, 
	Map map_landmarks) {
	// TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation
	//   3.33. Note that you'll need to switch the minus sign in that equation to a plus to account
	//   for the fact that the map's y-axis actually points downwards.)
	//   http://planning.cs.uiuc.edu/node99.html

	weights.clear();
	for (int p=0; p < particles.size(); p++) {
		std::vector<LandmarkObs> observations_ground_frame;
		for (int ob=0; ob < observations.size(); ob++) {
			LandmarkObs obs;
			obs.x = observations[ob].x * cos(particles[p].theta) -
				observations[ob].y * sin(particles[p].theta) +
				particles[p].x;
			obs.y = observations[ob].x * sin(particles[p].theta) +
				observations[ob].y * cos(particles[p].theta) +
				particles[p].y;
			obs.id = -1;
			observations_ground_frame.push_back(obs);
		}

		std::vector<LandmarkObs> predicted_measurement;
		for (int l=0; l < map_landmarks.landmark_list.size(); l++) {
			double distance = dist(particles[p].x,
				particles[p].y,
				map_landmarks.landmark_list[l].x_f,
				map_landmarks.landmark_list[l].y_f);
		 	if (distance <= sensor_range) {
				predicted_measurement.push_back(LandmarkObs{map_landmarks.landmark_list[l].id_i,
					map_landmarks.landmark_list[l].x_f,
					map_landmarks.landmark_list[l].y_f});
		 	}
		 }

		dataAssociation(predicted_measurement, observations_ground_frame);

		particles[p].weight = 1.0;
		for (int pm=0; pm < predicted_measurement.size(); pm++) {
			int min_index = -1;
			double minimum_distance = std::numeric_limits<double>::max();

			for (int ob=0; ob < observations_ground_frame.size(); ob++) {
				if (predicted_measurement[pm].id == observations_ground_frame[ob].id) {
					double distance = dist(predicted_measurement[pm].x,
						predicted_measurement[pm].y,
						observations_ground_frame[ob].x,
						observations_ground_frame[ob].y);
					if (distance < minimum_distance) {
						min_index = ob;
						minimum_distance = distance;
					}
				}
			}

			if (min_index != -1) {
			 	particles[p].weight *= MGP(predicted_measurement[pm].x, 
					predicted_measurement[pm].y,
					observations_ground_frame[min_index].x, 
					observations_ground_frame[min_index].y,
					std_landmark[0], 
					std_landmark[1]);
			}
		}
		weights.push_back(particles[p].weight);
	}
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight.
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

	std::discrete_distribution<int> distribution_weights(weights.begin(), weights.end());
	std::vector<Particle> resampled_particles;
	
	for(int i=0; i < num_particles; i++) {
		resampled_particles.push_back(particles[distribution_weights(gen)]);
	}
	particles = resampled_particles;
}

void ParticleFilter::write(std::string filename) {
	// You don't need to modify this file.
	std::ofstream dataFile;
	dataFile.open(filename, std::ios::app);
	for (int i = 0; i < num_particles; ++i) {
		dataFile << particles[i].x << " " << particles[i].y << " " << particles[i].theta << "\n";
	}
	dataFile.close();
}