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
#include <chrono>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// Set the number of particles. Initialize all particles to first position (based on estimates of
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).


    num_particles = 300;

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    default_random_engine gen(seed);
    normal_distribution<double> dist_x(x, std[0]);
    normal_distribution<double> dist_y(y, std[1]);
    normal_distribution<double> dist_t(theta, std[2]);

    for(int i=0; i<num_particles; i++) {
        Particle p;
        p.id = 0;
        p.x = dist_x(gen);
        p.y = dist_y(gen);
        p.theta = dist_t(gen);
        p.weight = 1;

        particles.push_back(p);
    }

    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/


    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    default_random_engine gen(seed);
    normal_distribution<double> dist_x(0, std_pos[0]);
    normal_distribution<double> dist_y(0, std_pos[1]);
    normal_distribution<double> dist_t(0, std_pos[2]);

    std::ofstream f;
    f.open("particle_histories.out", std::ios_base::app);

    for(auto& p : particles) {
        cout << "Particle " << &p << " moved from " << p.describe();
        double xf, yf, tf;
        xf = p.x + velocity / yaw_rate * (sin(p.theta + yaw_rate * delta_t) - sin(p.theta));
        yf = p.y + velocity / yaw_rate * (cos(p.theta) - cos(p.theta + yaw_rate * delta_t));
        tf = p.theta + yaw_rate * delta_t;

        p.x = xf;
        p.y = yf;
        p.theta = tf;
        cout << " to " << p.describe();

        // Add noise.
        double nx, ny, nt;
        nx = dist_x(gen);
        ny = dist_y(gen);
        nt = dist_t(gen);

        p.x += nx;
        p.y += ny;
        p.theta += nt;


        cout << " to " << p.describe() << "." << endl;
        f << p.describe();
        f.flush();

        cout << "noise added: " << nx << "," << ny << "," << nt << endl;
    }

    f << endl;

    f.close();

}



vector<Deviation> ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// Find the predicted measurement that is closest to each observed measurement and assign the
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to
	//   implement this method and use it as a helper during the updateWeights phase.


    // Brute force method.
    vector<Deviation> deviations;
    Deviation deviation, smallest_deviation;

    // Loop over all the landmark observations.
    for(auto& observation : observations) {

        // Find the nearest map landmark.
        double smallest_distance = INFINITY;

        // Loop over all the map landmarks, transformed into car coordinates.
        for(auto & prediction : predicted) {

            // Find the distance to this map landmark.
            deviation.dx = prediction.x - observation.x;
            deviation.dy = prediction.y - observation.y;

            // Record the closest map landmark for this observed landmark.
            double r = deviation.r();
            if( r < smallest_distance ) {
                smallest_deviation.dx = deviation.dx;
                smallest_deviation.dy = deviation.dy;
                observation.id = prediction.id;
                smallest_distance = r;
            }
        }

        // Save the deviation vector for computing likelihood later.
        deviations.push_back(smallest_deviation);
    }


    return deviations;
}

Map::single_landmark_s car2map(Particle& car, LandmarkObs obs) {
//    msg("car2map");
    Map::single_landmark_s map_obs;
    map_obs.x_f = car.x + cos(car.theta) * obs.x - sin(car.theta) * obs.y;
    map_obs.y_f = car.y + sin(car.theta) * obs.x + cos(car.theta) * obs.y;
    return map_obs;
}

LandmarkObs map2car(Particle& car, Map::single_landmark_s map_obs) {
//    msg("map2car");
    LandmarkObs car_obs;
//    cout << "Map observation " << map_obs.x_f << "," << map_obs.y_f << endl;
//    cout << "With car particle " << car.x<<","<<car.y<<","<<car.theta<<endl;
    car_obs.x = (map_obs.x_f - car.x) * cos(car.theta) + (map_obs.x_f - car.y) * sin(car.theta);
    car_obs.y = (map_obs.y_f - car.y) * cos(car.theta) - (map_obs.x_f - car.x) * sin(car.theta);
    // Copy the map ID.
    car_obs.id = map_obs.id_i;
//    cout << "Gives car observation " << car_obs.x << "," << car_obs.y << endl;
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
    for(Particle &particle : particles) {

        // Predict where the landmarks would be for this particle:
        // transform landmarks to car coordinates.
        vector<LandmarkObs> car_landmarks;
        for(auto& landmark : map_landmarks.landmark_list) {
            car_landmarks.push_back(map2car(particle, landmark));
        }

        // Associate each observation with one landmark.
        // Since observations is const, copy it here ... ?
        vector<LandmarkObs> assigned_observations = observations;
        vector<Deviation> deviations = dataAssociation(car_landmarks, assigned_observations);

        // Since we did the association now, I guess we should set it in the particle.
        vector<int> associations;
        vector<double> sense_x;
        vector<double> sense_y;
        for(auto& obs : assigned_observations) {
            associations.push_back(obs.id);
            Map::single_landmark_s sense = car2map(particle, obs);
            sense_x.push_back(sense.x_f);
            sense_y.push_back(sense.y_f);
        }
        SetAssociations(
                particle,
                associations,
                sense_x,
                sense_y
        );



        // With the distances between observations and predicted landmark locations; calculate a likelihood for each.
        // Compute the product likelihood for the particle.
        double likelihood = 1.0;
        double exponent;
//        cout << "Weight for particle " << &particle << " is... " << endl;
        for(auto& d : deviations) {
            double dx, dy;
            if(d.r() < sensor_range) {
                dx = d.dx;
                dy = d.dy;
            } else {
                dx = sqrt(pow(sensor_range, 2)/2.0);
                dy = dx;
            }
//            cout << "    (" << dx << "," << dy << ")->";
            exponent =
                      pow(dx, 2) / 2 / std_landmark[0] / std_landmark[0]
                    + pow(dy, 2) / 2 / std_landmark[1] / std_landmark[1];
            exponent *= -1;
            likelihood *= exp(exponent) / 2 / M_PI / std_landmark[0] / std_landmark[1];
//            cout << likelihood << endl;
        }
//        cout << endl;

        // Let the weight for the particle be just the product likelihood.
        particle.weight = likelihood;

    }
}

void ParticleFilter::resample() {
    // TODO: Resample particles with replacement with probability proportional to their weight.
    // NOTE: You may find std::discrete_distribution helpful here.
    //   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

    vector<double> weights;
//    cout << "weights = [";
//    for (int iparticle = 0; iparticle < particles.size(); iparticle++) {
    for(auto &particle : particles) {
        weights.push_back(particle.weight);
//        cout << particle.weight << ",";
    }
//    cout << "]" << endl;

    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    default_random_engine gen(seed);
    discrete_distribution<> dist(weights.begin(), weights.end());

    vector<Particle> resampled;
    for(int _iparticle=0; _iparticle<num_particles; _iparticle++) {
        int isample = dist(gen);
        resampled.push_back(particles[isample]);
    }

    particles = resampled;
}

void ParticleFilter::SetAssociations(
        Particle& particle,
        const std::vector<int>& associations,
        const std::vector<double>& sense_x,
        const std::vector<double>& sense_y
)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates
    particle.associations = associations;
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

//Particle::~Particle() {
//    cout << "    Deleting particle at " << this << "; #" << id << " at (" << x << "," << y << "," << theta << ")." << endl;
//}
//
//Particle::Particle() {
//    cout << "    Created particle at " << this << "." << endl;
//}
//
//Particle::Particle(const Particle &p2) {
//    // Copy constructor.
//    cout << "    Copied particle " << &p2 << " into " << this << "." << endl;
//    id = p2.id;
//    x = p2.x;
//    y = p2.y;
//    theta = p2.theta;
//    associations = p2.associations;
//    sense_x = p2.sense_x;
//    sense_y = p2.sense_y;
//}
void msg(std::string m) {
    std::cout << m << std::endl;
}

std::string Particle::describe() {
    std::ostringstream ss;
    ss << "(" << x << "," << y << "," << theta << ")";
    return ss.str();
}
