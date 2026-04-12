#include "sph.cuh"
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
      update_positions_tuner(fluid_particles, dev_ptr(velocity), bounding_box),
      update_velocities_tuner(fluid_particles, dev_ptr(velocity), dev_ptr(non_pressure_accel)),
      compute_boundary_mass_tuner(fluid_particles, boundary_particles, dev_ptr(boundary_mass), n_search.dev_ptr()),
      compute_viscosity_tuner(fluid_particles, dev_ptr(velocity), dev_ptr(density), dev_ptr(non_pressure_accel),
                              n_search.dev_ptr()),
      compute_surface_normals_tuner(fluid_particles, dev_ptr(density), dev_ptr(normal), n_search.dev_ptr()),
      compute_surface_tension_tuner(fluid_particles, dev_ptr(density), dev_ptr(normal), dev_ptr(non_pressure_accel),
                                    n_search.dev_ptr()),
      compute_pressure_tuner(fluid_particles, dev_ptr(density), dev_ptr(pressure)),
      apply_pressure_force_tuner(fluid_particles, boundary_particles, dev_ptr(density), dev_ptr(pressure),
                                 dev_ptr(velocity), dev_ptr(boundary_mass), n_search.dev_ptr()),
      apply_external_forces_tuner(fluid_particles, dev_ptr(non_pressure_accel), opts.external_force),
      apply_external_force(!opts.external_force.empty()) {
}

void CUDASPHSimulator::update(float delta) {
    auto lock = cuda_gl_positions->lock();
    float* positions_ptr = static_cast<float*>(lock.get_ptr());

    n_search.rebuild(positions_ptr);

    compute_boundary_mass(positions_ptr);

    compute_densities(positions_ptr);

    apply_non_pressure_forces(positions_ptr, delta);

    delta = adapt_time_step(delta, MIN_TIME_STEP, MAX_TIME_STEP);

    compute_pressure();
    apply_pressure_force(positions_ptr, delta);

    update_positions(positions_ptr, delta);
}

void CUDASPHSimulator::visualize(Shader* shader) {
    visualizer->visualize(shader, dev_ptr(density),
        REST_DENSITY * 0.5f, REST_DENSITY * 1.2f);
    // visualizer->visualize(shader, dev_ptr(velocity));
    // visualizer->visualize(shader, dev_ptr(boundary_mass),
    //     0.f, PARTICLE_MASS * 2.f, true);
}

void CUDASPHSimulator::compute_densities(float* positions_dev_ptr) {
    density_tuner.run(positions_dev_ptr);
}

void CUDASPHSimulator::update_positions(float* positions_dev_ptr, float delta) {
    update_positions_tuner.run(positions_dev_ptr, delta);
}

void CUDASPHSimulator::apply_non_pressure_forces(float* positions_dev_ptr, float delta) {
    apply_external_forces(positions_dev_ptr);

    compute_viscosity(positions_dev_ptr);
    compute_surface_tension(positions_dev_ptr);

    update_velocities(delta);

    compute_XSPH(positions_dev_ptr);
}

void CUDASPHSimulator::compute_boundary_mass(float* positions_dev_ptr) {
    compute_boundary_mass_tuner.run(positions_dev_ptr);
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
    update_velocities_tuner.run(delta);
}

void CUDASPHSimulator::compute_XSPH(const float* positions_dev_ptr) {
    if constexpr (XSPH_ALPHA == 0.f)
        return;
    // TODO
}

void CUDASPHSimulator::compute_viscosity(float* positions_dev_ptr) {
    compute_viscosity_tuner.run(positions_dev_ptr);
}

void CUDASPHSimulator::compute_surface_tension(float* positions_dev_ptr) {
    compute_surface_normals(positions_dev_ptr);

    compute_surface_tension_tuner.run(positions_dev_ptr);
}

void CUDASPHSimulator::compute_surface_normals(float* positions_dev_ptr) {
    compute_surface_normals_tuner.run(positions_dev_ptr);
}

void CUDASPHSimulator::apply_external_forces(float* positions_dev_ptr) {
    if (apply_external_force) {
        apply_external_forces_tuner.run(positions_dev_ptr);
    } else {
        thrust::fill(non_pressure_accel.begin(), non_pressure_accel.end(), GRAVITY);
    }
}

void CUDASPHSimulator::compute_pressure() {
    compute_pressure_tuner.run();
}

void CUDASPHSimulator::apply_pressure_force(float* positions_dev_ptr, float delta) {
    apply_pressure_force_tuner.run(positions_dev_ptr, delta);
}
