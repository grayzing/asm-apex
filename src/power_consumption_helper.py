import numpy as np

from sector_manager import SectorManager
from sleep_mode_manager import SleepModeManager
from simulation_kpis_handler import SimulationKPIHandler

class RadioUnitPowerConsumptionHelper:
    # This is from Usman et al.
    def __init__(self, num_sectors: int):
        self.power_consumption_sector_matrix: np.ndarray = np.zeros((num_sectors, ), dtype=np.float16)

        # constants
        self.power_amplifier_transmit_power: float = np.full((num_sectors, ), 40.0, dtype=np.float32) # watts
        self.total_efficiency_of_power_amplifier: float = np.full((num_sectors, ), 1.0, dtype=np.float32)

    def calculate_digital_components_power(self):
        pass

    def calculate_analog_components_power(self):
        pass

    def calculate_power_amplifier_power(self) -> np.ndarray:
        return self.power_amplifier_transmit_power / (self.total_efficiency_of_power_amplifier * self.power_amplifier_transmit_power)

    def calculate_total_power(self, sleep_mode_manager: SleepModeManager) -> np.float32:
        return np.sum(self.calculate_analog_components_power() + self.calculate_digital_components_power() + self.calculate_power_amplifier_power())

    def calculate_energy_efficiency(self, sleep_mode_manager: SleepModeManager, kpi_handler: SimulationKPIHandler, step: int):
        # Mbits/Joule
        return kpi_handler.calculate_throughput_mbps(step) / self.calculate_total_power(sleep_mode_manager)
