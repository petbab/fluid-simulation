#include "fluid_simulation.h"

#include <random>
#include <glm/gtc/constants.hpp>
#include <algorithm>
//#include <iostream>


void FluidSimulation::init_simulation() {
    velocities.resize(positions.size());
    densities.resize(positions.size());
    predicted_densities.resize(positions.size());
    alphas.resize(positions.size());
    divergence_errors.resize(positions.size());
    divergence_kappas.resize(positions.size());
    density_kappas.resize(positions.size());

    point_set_index = n_search->add_point_set(reinterpret_cast<float*>(positions.data()), positions.size());
    n_search->z_sort();
    n_search->find_neighbors();

    compute_densities();
    compute_alphas();
}

void FluidSimulation::simulation_step(double delta) {
    delta = adapt_time_step(delta);

    predict_velocities(delta);

    correct_density_error(delta);

    update_positions(delta);

    n_search->find_neighbors();

    compute_densities();
    compute_alphas();

    correct_divergence_error(delta);

    resolve_collisions(delta);

    first_iteration = false;
}

void FluidSimulation::compute_densities() {
    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < densities.size(); ++i) {
        float density = kernel.W(glm::vec3{0.f});
        glm::vec3 xi = positions[i];

        for_neighbors(i, [&](unsigned j){
            density += kernel.W(xi - positions[j]);
        });

        densities[i] = density * PARTICLE_MASS;
    }
}

void FluidSimulation::compute_alphas() {
    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < alphas.size(); ++i) {
        float sum_grad_sq = 0.f;
        glm::vec3 sum_grad{0.f};
        glm::vec3 xi = positions[i];

        for_neighbors(i, [&](unsigned j){
            glm::vec3 grad = -PARTICLE_MASS * kernel.grad_W(xi - positions[j]);
            sum_grad_sq += glm::dot(grad, grad);
            sum_grad -= grad;
        });

        sum_grad_sq += glm::dot(sum_grad, sum_grad);
        alphas[i] = sum_grad_sq > ALPHA_DENOM_EPSILON ? 1.f / sum_grad_sq : 0.f;
    }
}

double FluidSimulation::adapt_time_step(double delta) const {
    float max_velocity = glm::length(std::ranges::max(velocities, std::less{}, [](const glm::vec3 &v){
        return glm::length(v);
    }));

//    std::cout << "MAX VELOCITY: " << max_velocity << '\n';

    if (max_velocity < 1.e-9)
        return MAX_TIME_STEP;

    float cfl_max_time_step = CFL_FACTOR * PARTICLE_SPACING / max_velocity;

    return std::min(std::clamp(static_cast<float>(delta), MIN_TIME_STEP, MAX_TIME_STEP), cfl_max_time_step);
}

void FluidSimulation::predict_velocities(double delta) {
    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < velocities.size(); ++i) {
        glm::vec3 velocity_laplacian{0.f};
        glm::vec3 xi = positions[i];
        glm::vec3 vi = velocities[i];

        for_neighbors(i, [&](unsigned j){
            glm::vec3 x_ij = xi - positions[j];
            glm::vec3 v_ij = vi - velocities[j];

            velocity_laplacian += glm::dot(v_ij, x_ij) * kernel.grad_W(x_ij) / (densities[j] * (glm::dot(x_ij, x_ij) + 0.01f * SUPPORT_RADIUS * SUPPORT_RADIUS));
        });

        velocity_laplacian *= 10 * PARTICLE_MASS;

        velocities[i] += static_cast<float>(delta) * (GRAVITY + VISCOSITY * velocity_laplacian);
    }
}

void FluidSimulation::update_positions(double delta) {
    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < positions.size(); ++i)
        positions[i] += static_cast<float>(delta) * velocities[i];
}

void FluidSimulation::correct_density_error(double delta) {
    if (!first_iteration)
        warm_start_density(delta);
    std::ranges::fill(density_kappas, 0.f);

    float density_error = MAX_DENSITY_ERROR + 1.f;
    for (int iter = 0; (density_error > MAX_DENSITY_ERROR || iter < 2) && iter < MAX_DENSITY_ITERATIONS; ++iter) {
        density_error = 0.f;

        // Predict densities
        for (std::size_t i = 0; i < densities.size(); ++i) {
            float density_delta = 0.f;
            glm::vec3 xi = positions[i];
            glm::vec3 vi = velocities[i];

            for_neighbors(i, [&](unsigned j){
                density_delta += glm::dot(vi - velocities[j],
                                          kernel.grad_W(xi - positions[j]));
            });

            predicted_densities[i] = densities[i] + static_cast<float>(delta) * PARTICLE_MASS * density_delta;

            density_error += predicted_densities[i];
        }
        density_error = std::abs(density_error / static_cast<float>(densities.size()) - REST_DENSITY) / REST_DENSITY;

        // Adapt velocities
        for (std::size_t i = 0; i < velocities.size(); ++i) {
            float kappa_i = std::max(predicted_densities[i] - REST_DENSITY, 0.f) * alphas[i] / static_cast<float>(delta * delta);
            density_kappas[i] += kappa_i;

            glm::vec3 vel_correction{0.f};
            glm::vec3 xi = positions[i];
            float di = densities[i];

            for_neighbors(i, [&](unsigned j){
                float kappa_j = std::max(predicted_densities[j] - REST_DENSITY, 0.f) * alphas[j] / static_cast<float>(delta * delta);
                vel_correction += (kappa_i / (di * di) + kappa_j / (densities[j] * densities[j]))
                                  * kernel.grad_W(xi - positions[j]);
            });

            velocities[i] -= static_cast<float>(delta) * PARTICLE_MASS * vel_correction;
        }
    }
//    std::cout << "DENSITY ERROR: " << density_error << '\n';
}

void FluidSimulation::correct_divergence_error(double delta) {
    if (!first_iteration)
        warm_start_divergence(delta);
    std::ranges::fill(divergence_kappas, 0.f);

    float divergence_error = MAX_DIVERGENCE_ERROR + 1.f;
    for (int iter = 0; (divergence_error > MAX_DIVERGENCE_ERROR || iter < 1) && iter < MAX_DIVERGENCE_ITERATIONS; ++iter) {
        divergence_error = 0.f;

        for (std::size_t i = 0; i < divergence_errors.size(); ++i) {
            // Compute divergence error of particle i (Equation 9.)
            float divergence = 0.f;
            glm::vec3 xi = positions[i];
            glm::vec3 vi = velocities[i];

            for_neighbors(i, [&](unsigned j){
                divergence += glm::dot(vi - velocities[j],
                                       kernel.grad_W(xi - positions[j]));
            });

            divergence *= PARTICLE_MASS;
            divergence_errors[i] = divergence;

            divergence_error += std::abs(divergence);
        }
        divergence_error /= static_cast<float>(positions.size());

        // Adapt velocities
        for (std::size_t i = 0; i < velocities.size(); ++i) {
            float kappa_i = divergence_errors[i] * alphas[i] / static_cast<float>(delta);
            divergence_kappas[i] += kappa_i;

            glm::vec3 vel_correction{0.f};
            glm::vec3 xi = positions[i];
            float di = densities[i];

            for_neighbors(i, [&](unsigned j){
                float kappa_j = divergence_errors[j] * alphas[j] / static_cast<float>(delta);
                vel_correction += (kappa_i / (di * di) + kappa_j / (densities[j] * densities[j]))
                                  * kernel.grad_W(xi - positions[j]);
            });

            velocities[i] -= static_cast<float>(delta) * PARTICLE_MASS * vel_correction;
        }
    }
//    std::cout << "DIVERGENCE ERROR: " << divergence_error << '\n';
}

void FluidSimulation::warm_start_density(double delta) {
    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < velocities.size(); ++i) {
        float kappa_i = density_kappas[i];

        glm::vec3 vel_correction{0.f};
        glm::vec3 xi = positions[i];
        float di = densities[i];

        for_neighbors(i, [&](unsigned j){
            float kappa_j = density_kappas[j];
            vel_correction += (kappa_i / (di * di) + kappa_j / (densities[j] * densities[j]))
                              * kernel.grad_W(xi - positions[j]);
        });

        velocities[i] -= static_cast<float>(delta) * PARTICLE_MASS * vel_correction;
    }
}

void FluidSimulation::warm_start_divergence(double delta) {
    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < velocities.size(); ++i) {
        float kappa_i = divergence_kappas[i];

        glm::vec3 vel_correction{0.f};
        glm::vec3 xi = positions[i];
        float di = densities[i];

        for_neighbors(i, [&](unsigned j){
            float kappa_j = divergence_kappas[j];
            vel_correction += (kappa_i / (di * di) + kappa_j / (densities[j] * densities[j]))
                              * kernel.grad_W(xi - positions[j]);
        });

        velocities[i] -= static_cast<float>(delta) * PARTICLE_MASS * vel_correction;
    }
}

void FluidSimulation::resolve_collisions(double) {
    #pragma omp parallel for schedule(static)
    for (unsigned i = 0; i < positions.size(); ++i) {
        if (positions[i].x - PARTICLE_RADIUS < bounding_box.min.x) {
            positions[i].x = bounding_box.min.x + PARTICLE_RADIUS;
            velocities[i].x *= -ELASTICITY;
        } else if (positions[i].x + PARTICLE_RADIUS > bounding_box.max.x) {
            positions[i].x = bounding_box.max.x - PARTICLE_RADIUS;
            velocities[i].x *= -ELASTICITY;
        }
        if (positions[i].y - PARTICLE_RADIUS < bounding_box.min.y) {
            positions[i].y = bounding_box.min.y + PARTICLE_RADIUS;
            velocities[i].y *= -ELASTICITY;
        } else if (positions[i].y + PARTICLE_RADIUS > bounding_box.max.y) {
            positions[i].y = bounding_box.max.y - PARTICLE_RADIUS;
            velocities[i].y *= -ELASTICITY;
        }
        if (positions[i].z - PARTICLE_RADIUS < bounding_box.min.z) {
            positions[i].z = bounding_box.min.z + PARTICLE_RADIUS;
            velocities[i].z *= -ELASTICITY;
        } else if (positions[i].z + PARTICLE_RADIUS > bounding_box.max.z) {
            positions[i].z = bounding_box.max.z - PARTICLE_RADIUS;
            velocities[i].z *= -ELASTICITY;
        }
    }
}

void FluidSimulation::for_neighbors(unsigned int i, auto f) {
    CompactNSearch::PointSet &ps = n_search->point_set(point_set_index);
    for (unsigned j = 0; j < ps.n_neighbors(point_set_index, i); ++j)
        f(ps.neighbor(point_set_index, i, j));
}
