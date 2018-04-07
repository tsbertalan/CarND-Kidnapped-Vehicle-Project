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
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

    num_particles = 100;

    default_random_engine gen;
    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_t(theta, std[2]);

    for(int i=0; i<num_particles; i++) {
        Particle p;
        p.id = 0;
        p.x = dist_x(gen);
        p.y = dist_y(gen);
        p.theta = dist_y(gen);
        p.weight = 1;

        particles.push_back(p);
    }
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

    default_random_engine gen;
    normal_distribution<double> dist_x(0, std_pos[0]);
    normal_distribution<double> dist_y(0, std_pos[1]);
    normal_distribution<double> dist_t(0, std_pos[2]);

    for(int i=0; i<num_particles; i++) {
        Particle p = particles[i];
        p.x += velocity / yaw_rate * (sin(p.theta + yaw_rate * delta_t) - sin(p.theta));
        p.y += velocity / yaw_rate * (cos(p.theta) - cos(p.theta + yaw_rate * delta_t));
        p.theta += yaw_rate * delta_t;

        // Add noise.
        p.x += dist_x(gen);
        p.y += dist_y(gen);
        p.theta += dist_t(gen);
    }
    cout << "Propagated " << num_particles << " particles." << endl;

}



vector<Deviation> ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

    // Brute force method.
    LandmarkObs obsv, pred;
    vector<Deviation> deviations;
    Deviation d;

    // Loop over all the landmark observations.
    for(int iobsv=0; iobsv<observations.size(); iobsv++) {
        obsv = observations[iobsv];

        // Find the nearest map landmark.
        double bestdist = INFINITY;
        int ibest;

        // Loop over all the map landmarks, transformed into car coordinates.
        for(int ipred=0; ipred<predicted.size(); ipred++) {
            pred = predicted[ipred];

            // Find the distance to this map landmark.
            d.dx = pred.x - obsv.x;
            d.dy = pred.y - obsv.y;

            // Record the closest map landmark for this observed landmark.
            double r = d.r();
            if( r < bestdist ) {
                obsv.id = pred.id;
                bestdist = r;
            }
        }

        // Save the deviation vector for computing likelihood later.
        deviations.push_back(d);
    }

    return deviations;
}

//Map::single_landmark_s car2map(Particle car, LandmarkObs obs) {
//    Map::single_landmark_s map_obs;
//    map_obs.x = obs.x + cos(car.theta) * obs.x - sin(car.theta) * obs.y;
//    map_obs.y = obs.y + sin(car.theta) * obs.x + cos(car.theta) * obs.y;
//    return map_obs;
//}

LandmarkObs map2car(Particle car, Map::single_landmark_s map_obs) {
    LandmarkObs car_obs;
    car_obs.x = (map_obs.x_f - car.x) * cos(car.theta) + (map_obs.x_f - car.y) * sin(car.theta);
    car_obs.y = (map_obs.x_f - car.y) * cos(car.theta) - (map_obs.x_f - car.x) * sin(car.theta);
    // Copy the map ID.
    car_obs.id = map_obs.id_i;
    return car_obs;
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
	// Update the weights of each particle using a mult-variate Gaussian distribution. You can read
	//   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
	// NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
	//   according to the MAP'S coordinate system. You will need to transform between the two systems.
	//   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
	//   The following is a good resource for the theory:
	//   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
	//   and the following is a good resource for the actual equation to implement (look at equation 
	//   3.33
	//   http://planning.cs.uiuc.edu/node99.html

    // For each particle ...
    for(int iparticle=0; iparticle<num_particles; iparticle++) {

        // Predict where the landmarks would be for this particle:
        // transform landmarks to car coordinates.
        vector<LandmarkObs> car_landmarks;
        for(int imap=0; imap<map_landmarks.landmark_list.size(); imap++) {
            car_landmarks.push_back(map2car(particles[iparticle], map_landmarks.landmark_list[imap]));
        }

        // Associate each observation with one landmark.
        // Since observations is const, copy it here ... ?
        vector<LandmarkObs> assigned_observations = observations;
        vector<Deviation> deviations = dataAssociation(car_landmarks, assigned_observations);

        // With the distances between observations and predicted landmark locations; calculate a likelihood for each.
        // Compute the product likelihood for the particle.
        double likelihood = 1.0;
        double exponent;
        for(int iobs=0; iobs<deviations.size(); iobs++) {
            Deviation d = deviations[iobs];
            exponent = pow(d.dx, 2) / 2 / std_landmark[0] + pow(d.dy, 2) / 2 / std_landmark[1];
            exponent *= -1;
            // Don't bother normalizing the likelihoods.
            likelihood *= exp(exponent);// / 2 / M_PI / std_landmark[0] / std_landmark[1];
        }

        // Let the weight for the particle be just the product likelihood.
        particles[iparticle].weight = likelihood;

    }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best)
{
	vector<int> v = best.associations;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
	vector<double> v = best.sense_x;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
	vector<double> v = best.sense_y;
	stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}

double Deviation::r() {
    return sqrt(pow(dx, 2) + pow(dy, 2));
}
