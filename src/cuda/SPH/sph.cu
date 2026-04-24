#include "sph.cuh"

#include <ranges>
#include <thrust/transform_reduce.h>


template <class T>
static T* dev_ptr(thrust::device_vector<T>& vec) {
    return thrust::raw_pointer_cast(vec.data());
}

CUDASPHSimulator::CUDASPHSimulator(const opts_t& opts)
    : CUDASimulator(opts),
      density(fluid_particles),
      boundary_mass(boundary_particles),
      pressure(fluid_particles),
      velocity(fluid_particles),
      non_pressure_accel(fluid_particles),
      normal(fluid_particles),
      n_search(2.f * SUPPORT_RADIUS, total_particles),
      density_tuner(fluid_particles, total_particles, dev_ptr(density), dev_ptr(boundary_mass), n_search.dev_ptr()),
      update_positions_tuner(fluid_particles, dev_ptr(velocity)),
      update_velocities_tuner(fluid_particles, dev_ptr(velocity), dev_ptr(non_pressure_accel)),
      compute_viscosity_tuner(fluid_particles, dev_ptr(velocity), dev_ptr(density), dev_ptr(non_pressure_accel),
                              n_search.dev_ptr()),
      compute_surface_normals_tuner(fluid_particles, dev_ptr(density), dev_ptr(normal), n_search.dev_ptr()),
      compute_surface_tension_tuner(fluid_particles, dev_ptr(density), dev_ptr(normal), dev_ptr(non_pressure_accel),
                                    n_search.dev_ptr()),
      compute_pressure_tuner(fluid_particles, dev_ptr(density), dev_ptr(pressure)),
      apply_pressure_force_tuner(fluid_particles, boundary_particles, dev_ptr(density), dev_ptr(pressure),
                                 dev_ptr(velocity), dev_ptr(boundary_mass), n_search.dev_ptr()),
      active_tuners{
          {DENSITY_TUNER, &density_tuner},
          {UPDATE_POSITIONS_TUNER, &update_positions_tuner},
          {UPDATE_VELOCITIES_TUNER, &update_velocities_tuner},
          {COMPUTE_VISCOSITY_TUNER, &compute_viscosity_tuner},
          {COMPUTE_SURFACE_NORMALS_TUNER, &compute_surface_normals_tuner},
          {COMPUTE_SURFACE_TENSION_TUNER, &compute_surface_tension_tuner},
          {COMPUTE_PRESSURE_TUNER, &compute_pressure_tuner},
          {APPLY_PRESSURE_FORCE_TUNER, &apply_pressure_force_tuner},
      },
      tuning_budget{0.1f} {
    if (boundary_particles > 0) {
        compute_boundary_mass_tuner = std::make_unique<ComputeBoundaryMassTuner>(
            fluid_particles, boundary_particles, dev_ptr(boundary_mass), n_search.dev_ptr());
        active_tuners[COMPUTE_BOUNDARY_MASS_TUNER] = compute_boundary_mass_tuner.get();
    }
    if (!opts.external_force.empty()) {
        apply_external_forces_tuner = std::make_unique<ApplyExternalForcesTuner>(
            fluid_particles, dev_ptr(non_pressure_accel), opts.external_force);
        active_tuners[APPLY_EXTERNAL_FORCES_TUNER] = apply_external_forces_tuner.get();
    }

    scheduler = std::make_unique<TuningScheduler>(active_tuners.size(), tuning_budget);
}

void CUDASPHSimulator::update(float delta) {
    scheduler->schedule();

    auto lock = cuda_gl_positions->lock();
    float* positions_ptr = static_cast<float*>(lock.get_ptr());

    n_search.rebuild(positions_ptr, is_scheduled(REBUILD_N_SEARCH_TUNER));

    compute_boundary_mass(positions_ptr);

    compute_densities(positions_ptr);

    apply_non_pressure_forces(positions_ptr, delta);

    delta = adapt_time_step(delta, MIN_TIME_STEP, MAX_TIME_STEP);

    compute_pressure();
    apply_pressure_force(positions_ptr, delta);

    update_positions(positions_ptr, delta);
}

void CUDASPHSimulator::visualize(Shader* shader) {
    // visualizer->visualize(shader, dev_ptr(density),
    //     REST_DENSITY * 0.5f, REST_DENSITY * 1.2f);
    // visualizer->visualize(shader, dev_ptr(velocity));
    // visualizer->visualize(shader, dev_ptr(boundary_mass),
    //     0.f, PARTICLE_MASS * 2.f, true);
}

void CUDASPHSimulator::compute_densities(float* positions_dev_ptr) {
    density_tuner.run(positions_dev_ptr, is_scheduled(DENSITY_TUNER));
}

void CUDASPHSimulator::update_positions(float* positions_dev_ptr, float delta) {
    update_positions_tuner.run(positions_dev_ptr, delta, bounding_box, is_scheduled(UPDATE_POSITIONS_TUNER));
}

void CUDASPHSimulator::apply_non_pressure_forces(float* positions_dev_ptr, float delta) {
    apply_external_forces(positions_dev_ptr);

    compute_viscosity(positions_dev_ptr);
    compute_surface_tension(positions_dev_ptr);

    update_velocities(delta);

    compute_XSPH(positions_dev_ptr);
}

void CUDASPHSimulator::compute_boundary_mass(float* positions_dev_ptr) {
    if (compute_boundary_mass_tuner != nullptr)
        compute_boundary_mass_tuner->run(positions_dev_ptr, is_scheduled(COMPUTE_BOUNDARY_MASS_TUNER));
}

void CUDASPHSimulator::reset() {
    FluidSimulator::reset();

    thrust::fill(density.begin(), density.end(), 0);
    thrust::fill(boundary_mass.begin(), boundary_mass.end(), 0);
    thrust::fill(velocity.begin(), velocity.end(), make_float4(0));
    thrust::fill(non_pressure_accel.begin(), non_pressure_accel.end(), make_float4(0));
    thrust::fill(normal.begin(), normal.end(), make_float4(0));
    thrust::fill(pressure.begin(), pressure.end(), 0);
}

struct float4_length_sq {
    __host__ __device__ float operator()(const float4& v) const {
        return length(v);
    }
};

float CUDASPHSimulator::adapt_time_step(float delta, float min_step, float max_step) const {
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

void CUDASPHSimulator::update_velocities(float delta) {
    delta = std::min(delta, NON_PRESSURE_MAX_TIME_STEP);
    update_velocities_tuner.run(delta, is_scheduled(UPDATE_VELOCITIES_TUNER));
}

void CUDASPHSimulator::compute_XSPH(const float* positions_dev_ptr) {
    if constexpr (XSPH_ALPHA == 0.f)
        return;
    // TODO
}

void CUDASPHSimulator::compute_viscosity(float* positions_dev_ptr) {
    compute_viscosity_tuner.run(positions_dev_ptr, is_scheduled(COMPUTE_VISCOSITY_TUNER));
}

void CUDASPHSimulator::compute_surface_tension(float* positions_dev_ptr) {
    compute_surface_normals(positions_dev_ptr);

    compute_surface_tension_tuner.run(positions_dev_ptr, is_scheduled(COMPUTE_SURFACE_TENSION_TUNER));
}

void CUDASPHSimulator::compute_surface_normals(float* positions_dev_ptr) {
    compute_surface_normals_tuner.run(positions_dev_ptr, is_scheduled(COMPUTE_SURFACE_NORMALS_TUNER));
}

void CUDASPHSimulator::apply_external_forces(float* positions_dev_ptr) {
    if (apply_external_forces_tuner != nullptr) {
        apply_external_forces_tuner->run(positions_dev_ptr, is_scheduled(APPLY_EXTERNAL_FORCES_TUNER));
    } else {
        thrust::fill(non_pressure_accel.begin(), non_pressure_accel.end(), GRAVITY);
    }
}

void CUDASPHSimulator::compute_pressure() {
    compute_pressure_tuner.run(is_scheduled(COMPUTE_PRESSURE_TUNER));
}

void CUDASPHSimulator::apply_pressure_force(float* positions_dev_ptr, float delta) {
    apply_pressure_force_tuner.run(positions_dev_ptr, delta, is_scheduled(APPLY_PRESSURE_FORCE_TUNER));
}

bool CUDASPHSimulator::is_scheduled(tuners tuner_i) const {
    auto it = active_tuners.find(tuner_i);
    int i = std::distance(active_tuners.begin(), it);
    return scheduler->is_scheduled(i);
}

std::pair<int, int> CUDASPHSimulator::tuning_stats() const {
    std::pair stats{0, 0};
    for (const Tuner *tnr : active_tuners | std::views::values) {
        auto [s, t] = tnr->tuning_stats();
        stats.first += s;
        stats.second += t;
    }

    return stats;
}

void CUDASPHSimulator::set_tuning_budget(float tb) {
    tuning_budget = tb;
    scheduler->set_tune_iterations_per_frame(tb);
}
