from simulator import Simulator

if __name__ == "__main__":
    sim = Simulator(31, 500, 500)
    sim.run_simulation()
    sim.kpi_handler.print_kpis(500)
    