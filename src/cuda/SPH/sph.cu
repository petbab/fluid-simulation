#include "sph.cuh"

#include <ranges>
#include <thrust/transform_reduce.h>
#include <thrust/reduce.h>
#include <thrust/functional.h>


CUDASPHSimulator::CUDASPHSimulator(const opts_t& opts)
    : FluidSimulator(opts),
      particle_data{fluid_particles, boundary_particles, CELL_SIZE},
      particle_data_visualizer{&particle_data, total_particles, fluid_particles},
      step_tuner(fluid_particles, boundary_particles, opts.external_force),
      active_tuners{
          {STEP_TUNER, &step_tuner},
      },
      tuning_budget{0.1f} {
    if (boundary_particles > 0) {
        compute_boundary_mass_tuner = std::make_unique<ComputeBoundaryMassTuner>(boundary_particles);
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

    float np_delta = std::min(delta, NON_PRESSURE_MAX_TIME_STEP);
    delta = adapt_time_step(delta, MIN_TIME_STEP, MAX_TIME_STEP);

    step_tuner.run([&](float cell_size_mult) {
            particle_data.set_cell_size_mult(cell_size_mult);
            particle_data.sort(positions_src, positions_dst);
        },
        has_boundary() ? boundary_n_search->dev_ptr() : nullptr,
        positions_dst,
        particle_data.velocity_dst(),
        particle_data.density(),
        particle_data.pressure(),
        particle_data.pressure_accel(),
        particle_data.non_pressure_accel(),
        particle_data.normal(),
        has_boundary() ? particle_data.boundary_mass() : nullptr,
        delta, np_delta, bounding_box,
        is_scheduled(STEP_TUNER)
    );

    particle_data_visualizer.update();
}

void CUDASPHSimulator::visualize(Shader* shader) {
    particle_data_visualizer.visualize(shader);
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
    boundary_n_search = std::make_unique<NSearchWrapperTuned>(
        std::bit_ceil(boundary_particles), CELL_SIZE, boundary_particles, true);
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

void CUDASPHSimulator::set_frozen_config(ktt::KernelConfiguration cfg) {
    step_tuner.set_frozen_config(std::move(cfg));
}

void CUDASPHSimulator::set_result_out(std::optional<std::filesystem::path> out) {
    step_tuner.set_results_out(std::move(out));
}

struct float4_length {
    __host__ __device__ float operator()(const float4& v) const {
        return sqrtf(dot(v, v));
    }
};

struct float4_ke {
    __host__ __device__ float operator()(const float4& v) const {
        float speed_sq = dot(v, v);
        return 0.5f * CUDASPHSimulator::PARTICLE_MASS * speed_sq;
    }
};

std::pair<float, float> CUDASPHSimulator::compute_state_metrics() const {
    const auto& vel = particle_data.velocity_vec();
    float mean_speed = thrust::transform_reduce(
        vel.begin(), vel.end(),
        float4_length(),
        0.0f,
        thrust::plus<float>()
    ) / static_cast<float>(fluid_particles);

    float ke = thrust::transform_reduce(
        vel.begin(), vel.end(),
        float4_ke(),
        0.0f,
        thrust::plus<float>()
    );

    return {mean_speed, ke};
}
