#pragma once

#include <thrust/device_vector.h>
#include <cuda/simulator.h>
#include <cuda/nsearch/nsearch.h>
#include <cuda/math.cuh>
#include <thrust/transform_reduce.h>
#include <cuda/tuning/compute_densities.cuh>
#include <cuda/tuning/update_positions.cuh>
#include "cuda/tuning/apply_external_forces.cuh"
#include "cuda/tuning/compute_boundary_mass.cuh"
#include "cuda/tuning/compute_surface_normals.cuh"
#include "cuda/tuning/compute_surface_tension.cuh"
#include "cuda/tuning/compute_viscosity.cuh"
#include "cuda/tuning/update_velocities.cuh"


class CUDASPHBase : public CUDASimulator {
public:
    ///////////////////////////////////////////////////////////////////////////////
    ////                         SIMULATION PARAMETERS                         ////
    ///////////////////////////////////////////////////////////////////////////////
    static constexpr float REST_DENSITY = 1000.f;
    static constexpr float SUPPORT_RADIUS = 2.f * PARTICLE_SPACING;
    static constexpr float PARTICLE_VOLUME = PARTICLE_SPACING * PARTICLE_SPACING * PARTICLE_SPACING * 0.8;
    static constexpr float PARTICLE_MASS = REST_DENSITY * PARTICLE_VOLUME;

    static constexpr float XSPH_ALPHA = 0.f;

    static constexpr float4 GRAVITY{0, -9.81f, 0, 0};

    static constexpr float CFL_FACTOR = 0.4f;
    static constexpr float NON_PRESSURE_MAX_TIME_STEP = 0.015;
    ///////////////////////////////////////////////////////////////////////////////

    struct float4_length_sq {
        __host__ __device__ float operator()(const float4& v) const {
            return length(v);
        }
    };

    CUDASPHBase(const opts_t &opts)
    : CUDASimulator(opts),
      density(fluid_particles),
      boundary_mass(boundary_particles),
      velocity(fluid_particles),
      n_search{2.f * SUPPORT_RADIUS, total_particles},
      non_pressure_accel(fluid_particles),
      normal(fluid_particles),
      density_tuner(fluid_particles, total_particles, thrust::raw_pointer_cast(density.data()),
          thrust::raw_pointer_cast(boundary_mass.data()), n_search.dev_ptr()),
      update_positions_tuner(fluid_particles, thrust::raw_pointer_cast(velocity.data()), bounding_box),
      update_velocities_tuner(fluid_particles, thrust::raw_pointer_cast(velocity.data()),
          thrust::raw_pointer_cast(non_pressure_accel.data())),
      compute_boundary_mass_tuner(fluid_particles, boundary_particles,
          thrust::raw_pointer_cast(boundary_mass.data()), n_search.dev_ptr()),
      compute_viscosity_tuner(fluid_particles, thrust::raw_pointer_cast(velocity.data()),
          thrust::raw_pointer_cast(density.data()), thrust::raw_pointer_cast(non_pressure_accel.data()),
          n_search.dev_ptr()),
      compute_surface_normals_tuner(fluid_particles, thrust::raw_pointer_cast(density.data()),
          thrust::raw_pointer_cast(normal.data()), n_search.dev_ptr()),
      compute_surface_tension_tuner(fluid_particles, thrust::raw_pointer_cast(density.data()),
          thrust::raw_pointer_cast(normal.data()), thrust::raw_pointer_cast(non_pressure_accel.data()),
          n_search.dev_ptr()),
      apply_external_forces_tuner(fluid_particles, thrust::raw_pointer_cast(non_pressure_accel.data()),
          opts.external_force),
      apply_external_force(!opts.external_force.empty()) {
    }

protected:
    void compute_densities(float* positions_dev_ptr) {
        density_tuner.run(positions_dev_ptr);
    }

    void update_positions(float* positions_dev_ptr, float delta) {
        update_positions_tuner.run(positions_dev_ptr, delta);
    }

    void apply_non_pressure_forces(float* positions_dev_ptr, float delta) {
        apply_external_forces(positions_dev_ptr);

        compute_viscosity(positions_dev_ptr);
        compute_surface_tension(positions_dev_ptr);

        update_velocities(delta);

        compute_XSPH(positions_dev_ptr);
    }

    void compute_boundary_mass(float* positions_dev_ptr) {
        compute_boundary_mass_tuner.run(positions_dev_ptr);
    }

    void reset() override {
        FluidSimulator::reset();

        thrust::fill(density.begin(), density.end(), 0);
        thrust::fill(boundary_mass.begin(), boundary_mass.end(), 0);
        thrust::fill(velocity.begin(), velocity.end(), make_float4(0));
        thrust::fill(non_pressure_accel.begin(), non_pressure_accel.end(), make_float4(0));
        thrust::fill(normal.begin(), normal.end(), make_float4(0));
    }

    // Adapt the time step size according to the Courant-Friedrich-Levy (CFL) condition
    float adapt_time_step(float delta, float min_step, float max_step) const {
        float max_velocity = sqrtf(thrust::transform_reduce(
            velocity.begin(),
            velocity.end(),
            float4_length_sq(),      // unary transform
            0.0f,                    // initial value
            thrust::maximum<float>() // reduction op
        ));

        if (max_velocity < 1.e-9)
            return max_step;

        float cfl_max_time_step = CFL_FACTOR * PARTICLE_SPACING / max_velocity;

        return std::min(std::clamp(delta, min_step, max_step), cfl_max_time_step);
    }

private:
    void update_velocities(float delta) {
        delta = std::min(delta, NON_PRESSURE_MAX_TIME_STEP);
        update_velocities_tuner.run(delta);
    }

    /**
     * Simulates viscosity by smoothing the velocity field [SPH Tutorial, eq. 103].
     */
    void compute_XSPH(const float* positions_dev_ptr) {
        if constexpr (XSPH_ALPHA == 0.f)
            return;
        // TODO
    }

    /**
     * Computes and applies the viscous force using an explicit viscosity model.
     * Approximates the Laplacian of the velocity field via finite differences
     * [SPH Tutorial, eq. 102].
     */
    void compute_viscosity(float* positions_dev_ptr) {
        compute_viscosity_tuner.run(positions_dev_ptr);
    }

    /**
     * Simulates surface tension using the macroscopic approach by Akinci et al. (2013).
     */
    void compute_surface_tension(float* positions_dev_ptr) {
        compute_surface_normals(positions_dev_ptr);

        compute_surface_tension_tuner.run(positions_dev_ptr);
    }

    /**
     * Computes surface normals used to calculate the curvature force
     * in compute_surface_tension [SPH Tutorial, eq. 125].
     */
    void compute_surface_normals(float* positions_dev_ptr) {
        compute_surface_normals_tuner.run(positions_dev_ptr);
    }

    void apply_external_forces(float* positions_dev_ptr) {
        if (apply_external_force) {
            apply_external_forces_tuner.run(positions_dev_ptr);
        } else {
            thrust::fill(non_pressure_accel.begin(), non_pressure_accel.end(), GRAVITY);
        }
    }

protected:
    thrust::device_vector<float> density, boundary_mass;
    thrust::device_vector<float4> velocity;
    NSearchWrapper n_search;

private:
    thrust::device_vector<float4> non_pressure_accel, normal;
    DensityTuner density_tuner;
    UpdatePositionsTuner update_positions_tuner;
    UpdateVelocitiesTuner update_velocities_tuner;
    ComputeBoundaryMassTuner compute_boundary_mass_tuner;
    ComputeViscosityTuner compute_viscosity_tuner;
    ComputeSurfaceNormalsTuner compute_surface_normals_tuner;
    ComputeSurfaceTensionTuner compute_surface_tension_tuner;
    ApplyExternalForcesTuner apply_external_forces_tuner;
    bool apply_external_force;
};
