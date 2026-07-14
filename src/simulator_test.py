from simulator import Simulator

if __name__ == "__main__":
    sim: Simulator = Simulator(19, 200, 50)
    sim.run_simulation()

    sim.kpi_handler.print_kpis(50)
    print(sim.power_consumption_handler.calculate_energy_efficiency(sim.sleep_mode_manager, sim.kpi_handler, 50))