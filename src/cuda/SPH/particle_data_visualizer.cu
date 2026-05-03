#include "particle_data_visualizer.cuh"
#include "sph.cuh"


const std::map<ParticleDataVisualizer::mode_t, ParticleDataVisualizer::mode_spec_t> ParticleDataVisualizer::modes = {
    {mode_t::pretty, mode_spec_t{false}},
    {mode_t::none, mode_spec_t{false}},
    {mode_t::density, mode_spec_t{true, 0.f, 3.f * CUDASPHSimulator::REST_DENSITY}},
    {mode_t::pressure, mode_spec_t{true, 0.f, 4.f}},
    {mode_t::velocity, mode_spec_t{true, 0.f, 4.f}},
    {mode_t::non_pressure_accel, mode_spec_t{true, 0.f, 20.f}},
    {mode_t::normal, mode_spec_t{true, 0.f, 2.f}},
    {mode_t::indices, mode_spec_t{true}},
    {mode_t::boundary_mass, mode_spec_t{true, 0.f, 3.f * CUDASPHSimulator::PARTICLE_MASS}},
    {mode_t::boundary_indices, mode_spec_t{true}},
};

const auto ParticleDataVisualizer::mode_strings = std::vector{
    "Pretty",
    "None",
    "Density",
    "Pressure",
    "Velocity",
    "Non-Pressure Acceleration",
    "Normal",
    "Indices",
    "Boundary Mass",
    "Boundary Indices",
};
