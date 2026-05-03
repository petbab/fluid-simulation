#include "sph.cuh"

#include <ranges>
#include <thrust/transform_reduce.h>


CUDASPHSimulator::CUDASPHSimulator(const opts_t& opts)
    : FluidSimulator(opts),
      particle_data{fluid_particles, boundary_particles, CELL_SIZE},
      particle_data_visualizer{&particle_data, total_particles, fluid_particles},
      fluid_n_search(CELL_SIZE, fluid_particles),
      density_tuner(fluid_particles, total_particles),
      update_positions_tuner(fluid_particles),
      compute_viscosity_tuner(fluid_particles),
      compute_surface_normals_tuner(fluid_particles),
      compute_surface_tension_tuner(fluid_particles),
      compute_pressure_tuner(fluid_particles),
      apply_pressure_force_tuner(fluid_particles, boundary_particles),
      active_tuners{
          {DENSITY_TUNER, &density_tuner},
          {UPDATE_POSITIONS_TUNER, &update_positions_tuner},
          {COMPUTE_VISCOSITY_TUNER, &compute_viscosity_tuner},
          {COMPUTE_SURFACE_NORMALS_TUNER, &compute_surface_normals_tuner},
          {COMPUTE_SURFACE_TENSION_TUNER, &compute_surface_tension_tuner},
          {COMPUTE_PRESSURE_TUNER, &compute_pressure_tuner},
          {APPLY_PRESSURE_FORCE_TUNER, &apply_pressure_force_tuner},
      },
      tuning_budget{0.1f} {
    if (boundary_particles > 0) {
        compute_boundary_mass_tuner = std::make_unique<ComputeBoundaryMassTuner>(boundary_particles);
    }
    if (!opts.external_force.empty()) {
        apply_external_forces_tuner = std::make_unique<ApplyExternalForcesTuner>(
            fluid_particles, opts.external_force);
        active_tuners[APPLY_EXTERNAL_FORCES_TUNER] = apply_external_forces_tuner.get();
    }

    scheduler = std::make_unique<TuningScheduler>(active_tuners.size(), tuning_budget);
}

void CUDASPHSimulator::init_positions(GLuint pos_vao_a, GLuint pos_vao_b) {
    particle_data.init_positions(pos_vao_a, pos_vao_b);
    if (has_boundary())
        init_boundary();
}

void CUDASPHSimulator::update(float delta) {
    scheduler->schedule();

    auto lock_src = particle_data.position().lock_src();
    auto lock_dst = particle_data.position().lock_dst();
    float4* positions_src = static_cast<float4*>(lock_src.get_ptr());
    float4* positions_dst = static_cast<float4*>(lock_dst.get_ptr());

    // Sort saves updated positions into 'positions_dst'
    particle_data.sort(positions_src, positions_dst);

    fluid_n_search.rebuild(positions_dst, is_scheduled(REBUILD_N_SEARCH_TUNER));

    compute_densities(positions_dst);

    apply_external_forces(positions_dst);
    compute_viscosity(positions_dst);
    compute_surface_tension(positions_dst);

    float np_delta = std::min(delta, NON_PRESSURE_MAX_TIME_STEP);
    delta = adapt_time_step(delta, MIN_TIME_STEP, MAX_TIME_STEP);

    compute_pressure();
    apply_pressure_force(positions_dst, delta);

    update_positions(positions_dst, delta, np_delta);

    particle_data_visualizer.update();
}

void CUDASPHSimulator::visualize(Shader* shader) {
    particle_data_visualizer.visualize(shader);
}

void CUDASPHSimulator::compute_densities(float4* positions_dev_ptr) {
    density_tuner.run(
        positions_dev_ptr, particle_data.density(), particle_data.boundary_mass(), fluid_n_search.dev_ptr(),
        total_particles, fluid_particles,
        has_boundary() ? boundary_n_search->dev_ptr() : nullptr,
        is_scheduled(DENSITY_TUNER));
}

void CUDASPHSimulator::update_positions(float4* positions_dev_ptr, float delta, float np_delta) {
    update_positions_tuner.run(positions_dev_ptr, particle_data.velocity(),
        particle_data.non_pressure_accel(), fluid_particles, delta, np_delta,
        bounding_box, is_scheduled(UPDATE_POSITIONS_TUNER));
}

void CUDASPHSimulator::compute_boundary_mass(float4* positions_dev_ptr) {
    assert(has_boundary());
    compute_boundary_mass_tuner->run(
        positions_dev_ptr + fluid_particles, particle_data.boundary_mass(),
        boundary_particles, boundary_n_search->dev_ptr(), false);
}

void CUDASPHSimulator::reset() {
    FluidSimulator::reset();

    thrust::fill(particle_data.density_vec().begin(), particle_data.density_vec().end(), 0);
    thrust::fill(particle_data.velocity_vec().begin(), particle_data.velocity_vec().end(), make_float4(0));
    thrust::fill(particle_data.non_pressure_accel_vec().begin(), particle_data.non_pressure_accel_vec().end(),
                 make_float4(0));
    thrust::fill(particle_data.normal_vec().begin(), particle_data.normal_vec().end(), make_float4(0));
    thrust::fill(particle_data.pressure_vec().begin(), particle_data.pressure_vec().end(), 0);
}

void CUDASPHSimulator::init_boundary() {
    assert(has_boundary());

    auto lock_src = particle_data.position().lock_src();
    auto lock_dst = particle_data.position().lock_dst();
    float4* positions_src = static_cast<float4*>(lock_src.get_ptr());
    float4* positions_dst = static_cast<float4*>(lock_dst.get_ptr());

    particle_data.sort_boundary(positions_src, positions_dst);

    build_boundary_n_search(positions_dst);
    compute_boundary_mass(positions_dst);
}

void CUDASPHSimulator::build_boundary_n_search(float4* positions_dev_ptr) {
    assert(has_boundary());
    boundary_n_search = std::make_unique<NSearchWrapper>(CELL_SIZE, boundary_particles, true);
    boundary_n_search->rebuild(positions_dev_ptr + fluid_particles, false);
}

bool CUDASPHSimulator::has_boundary() const {
    return boundary_particles > 0;
}

struct float4_length_sq {
    __host__ __device__ float operator()(const float4& v) const {
        return dot(v, v);
    }
};

float CUDASPHSimulator::adapt_time_step(float delta, float min_step, float max_step) const {
    float max_velocity = sqrtf(thrust::transform_reduce(
        particle_data.velocity_vec().begin(),
        particle_data.velocity_vec().end(),
        float4_length_sq(), // unary transform
        0.0f, // initial value
        thrust::maximum<float>() // reduction op
    ));

    if (max_velocity < 1.e-9)
        return max_step;

    float cfl_max_time_step = CFL_FACTOR * PARTICLE_SPACING / max_velocity;

    return std::min(std::clamp(delta, min_step, max_step), cfl_max_time_step);
}

void CUDASPHSimulator::compute_viscosity(float4* positions_dev_ptr) {
    compute_viscosity_tuner.run(positions_dev_ptr, particle_data.velocity(), particle_data.density(),
                                particle_data.non_pressure_accel(),
                                fluid_particles, fluid_n_search.dev_ptr(), is_scheduled(COMPUTE_VISCOSITY_TUNER));
}

void CUDASPHSimulator::compute_surface_tension(float4* positions_dev_ptr) {
    compute_surface_normals(positions_dev_ptr);

    compute_surface_tension_tuner.run(positions_dev_ptr, particle_data.density(), particle_data.normal(),
                                      particle_data.non_pressure_accel(),
                                      fluid_particles, fluid_n_search.dev_ptr(), is_scheduled(COMPUTE_SURFACE_TENSION_TUNER));
}

void CUDASPHSimulator::compute_surface_normals(float4* positions_dev_ptr) {
    compute_surface_normals_tuner.run(positions_dev_ptr, particle_data.density(), particle_data.normal(),
                                      fluid_particles,
                                      fluid_n_search.dev_ptr(), is_scheduled(COMPUTE_SURFACE_NORMALS_TUNER));
}

void CUDASPHSimulator::apply_external_forces(float4* positions_dev_ptr) {
    if (apply_external_forces_tuner != nullptr) {
        apply_external_forces_tuner->run(positions_dev_ptr, particle_data.non_pressure_accel(), fluid_particles,
                                         is_scheduled(APPLY_EXTERNAL_FORCES_TUNER));
    } else {
        thrust::fill(particle_data.non_pressure_accel_vec().begin(), particle_data.non_pressure_accel_vec().end(),
                     GRAVITY);
    }
}

void CUDASPHSimulator::compute_pressure() {
    compute_pressure_tuner.run(particle_data.density(), particle_data.pressure(), fluid_particles,
                               is_scheduled(COMPUTE_PRESSURE_TUNER));
}

void CUDASPHSimulator::apply_pressure_force(float4* positions_dev_ptr, float delta) {
    apply_pressure_force_tuner.run(positions_dev_ptr, particle_data.density(), particle_data.pressure(),
                                   particle_data.velocity(),
                                   particle_data.boundary_mass(), fluid_particles, boundary_particles, delta,
                                   fluid_n_search.dev_ptr(),
                                   has_boundary() ? boundary_n_search->dev_ptr() : nullptr,
                                   is_scheduled(APPLY_PRESSURE_FORCE_TUNER));
}

bool CUDASPHSimulator::is_scheduled(tuners tuner_i) const {
    auto it = active_tuners.find(tuner_i);
    int i = std::distance(active_tuners.begin(), it);
    return scheduler->is_scheduled(i);
}

std::pair<int, int> CUDASPHSimulator::tuning_stats() const {
    std::pair stats{0, 0};
    for (const Tuner* tnr : active_tuners | std::views::values) {
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

void CUDASPHSimulator::reset_tuning() {
    for (Tuner* tnr : active_tuners | std::views::values)
        tnr->clear_configuration_data();
}
