#include "dfsph.h"

#include <algorithm>
//#include <iostream>


DFSPHSimulator::DFSPHSimulator(const opts_t &opts)
    : SPHBase(opts, SUPPORT_RADIUS),
      kernel{SUPPORT_RADIUS} {
    predicted_densities.resize(particle_count);
    alphas.resize(particle_count);
    divergence_errors.resize(particle_count);
    divergence_kappas.resize(particle_count);
    density_kappas.resize(particle_count);

    compute_densities();
    compute_alphas();
}

void DFSPHSimulator::update(float delta) {
    delta = adapt_time_step(delta, MIN_TIME_STEP, MAX_TIME_STEP, SUPPORT_RADIUS);

    apply_non_pressure_forces(delta);

    correct_density_error(delta);

    update_positions(delta);

    find_neighbors();

    compute_densities();
    compute_alphas();

    correct_divergence_error(delta);

    resolve_collisions();

    first_iteration = false;
}

void DFSPHSimulator::compute_alphas() {
    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < particle_count; ++i) {
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

void DFSPHSimulator::correct_density_error(float delta) {
    if (!first_iteration)
        warm_start_density(delta);
    std::ranges::fill(density_kappas, 0.f);

    float density_error = MAX_DENSITY_ERROR + 1.f;
    for (int iter = 0; (density_error > MAX_DENSITY_ERROR || iter < 2) && iter < MAX_DENSITY_ITERATIONS; ++iter) {
        density_error = 0.f;

        // Predict densities
        for (std::size_t i = 0; i < particle_count; ++i) {
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
        density_error = std::abs(density_error / static_cast<float>(particle_count) - REST_DENSITY) / REST_DENSITY;

        // Adapt velocities
        for (std::size_t i = 0; i < particle_count; ++i) {
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

void DFSPHSimulator::correct_divergence_error(float delta) {
    if (!first_iteration)
        warm_start_divergence(delta);
    std::ranges::fill(divergence_kappas, 0.f);

    float divergence_error = MAX_DIVERGENCE_ERROR + 1.f;
    for (int iter = 0; (divergence_error > MAX_DIVERGENCE_ERROR || iter < 1) && iter < MAX_DIVERGENCE_ITERATIONS; ++iter) {
        divergence_error = 0.f;

        for (std::size_t i = 0; i < particle_count; ++i) {
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
        divergence_error /= static_cast<float>(particle_count);

        // Adapt velocities
        for (std::size_t i = 0; i < particle_count; ++i) {
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

void DFSPHSimulator::warm_start_density(float delta) {
    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < particle_count; ++i) {
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

void DFSPHSimulator::warm_start_divergence(float delta) {
    #pragma omp parallel for schedule(static)
    for (std::size_t i = 0; i < particle_count; ++i) {
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

void DFSPHSimulator::reset() {
    SPHBase::reset();

    std::ranges::fill(predicted_densities, 0);
    std::ranges::fill(alphas, 0);
    std::ranges::fill(divergence_errors, 0);
    std::ranges::fill(divergence_kappas, 0);
    std::ranges::fill(density_kappas, 0);

    first_iteration = true;
}
