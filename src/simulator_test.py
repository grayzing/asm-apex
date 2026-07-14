from simulator import Simulator

if __name__ == "__main__":
    sim: Simulator = Simulator(19, 570, 50)
    sim.run_simulation()

    sim.kpi_handler.print_kpis(50)