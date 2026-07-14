import numpy as np

from sector_manager import SectorManager
from sleep_mode_manager import SleepModeManager
from simulation_kpis_handler import SimulationKPIHandler

class RadioUnitPowerConsumptionHelper:
    def __init__(self, num_sectors: int):
        self.num_sectors = num_sectors
        self.power_consumption_sector_matrix: np.ndarray = np.zeros((num_sectors, ), dtype=np.float16)

        # 1. Map Sectors to Physical O-RU Hardware IDs
        # Sectors 0, 1, 2 -> physical_ru_id 0 (64T64R Macro RU)
        # Sectors 3+      -> physical_ru_id 1, 2, 3... (Independent 4T4R RUs)
        self.sector_to_ru_map = np.zeros(num_sectors, dtype=np.int32)
        
        # Sector 0, 1, 2 belong to the first Macro RU (ID 0)
        self.sector_to_ru_map[0:3] = 0 
        
        # Subsequent sectors get their own individual 4T4R O-RUs (IDs 1, 2, 3...)
        for sec_idx in range(3, num_sectors):
            self.sector_to_ru_map[sec_idx] = sec_idx - 2  # Shift IDs to keep them unique

        # Determine total physical RUs
        self.num_physical_rus = int(np.max(self.sector_to_ru_map) + 1)

        # 2. Define Hardware Configuration per Physical RU
        # Index 0 is our 64T64R Macro, all others are 4T4R
        self.is_macro_ru = np.zeros(self.num_physical_rus, dtype=bool)
        self.is_macro_ru[0] = True

        # 3. Base Hardware Profiles (Usman et al. baseline scales)
        # Columns: [Active, SM1, SM2, SM3, SM4]
        self.rf_power_4t4r = np.array([365.0, 61.0, 25.0, 18.0, 10.0], dtype=np.float32)
        self.digital_power_4t4r = np.array([32.0, 27.0, 15.0, 10.0, 5.0], dtype=np.float32)

        self.rf_power_64t64r = np.array([1200.0, 270.0, 130.0, 75.0, 25.0], dtype=np.float32)
        self.digital_power_64t64r = np.array([150.0, 50.0, 30.0, 15.0, 10.0], dtype=np.float32)

    def calculate_digital_components_power(self, sleep_mode_manager: SleepModeManager) -> np.ndarray:
        sleep_modes = sleep_mode_manager.sector_sleep_mode_matrix
        clamped_modes = np.clip(sleep_modes, 0, 4)

        # Resolve digital power at the physical O-RU level
        digital_power_per_ru = np.zeros(self.num_physical_rus, dtype=np.float32)

        # Determine active sleep mode per physical RU (the minimum sleep mode = shallowest active state)
        ru_sleep_modes = np.zeros(self.num_physical_rus, dtype=np.int32)
        for ru_id in range(self.num_physical_rus):
            # Find all sectors belonging to this RU
            associated_sectors = np.where(self.sector_to_ru_map == ru_id)[0]
            # Digital unit runs at the rate of the least-sleeping sector
            ru_sleep_modes[ru_id] = np.min(clamped_modes[associated_sectors])

        # Apply profile specs based on RU hardware classification
        for ru_id in range(self.num_physical_rus):
            mode = ru_sleep_modes[ru_id]
            if self.is_macro_ru[ru_id]:
                digital_power_per_ru[ru_id] = self.digital_power_64t64r[mode]
            else:
                digital_power_per_ru[ru_id] = self.digital_power_4t4r[mode]

        # Map RU-level digital power back to individual sectors (sharing the load evenly)
        sector_digital_power = np.zeros(self.num_sectors, dtype=np.float32)
        for ru_id in range(self.num_physical_rus):
            associated_sectors = np.where(self.sector_to_ru_map == ru_id)[0]
            # Distribute the digital baseline across its active sectors
            sector_digital_power[associated_sectors] = digital_power_per_ru[ru_id] / len(associated_sectors)

        return sector_digital_power

    def calculate_rf_processing_power(self, sleep_mode_manager: SleepModeManager) -> np.ndarray:
        sleep_modes = sleep_mode_manager.sector_sleep_mode_matrix
        clamped_modes = np.clip(sleep_modes, 0, 4)

        sector_rf_power = np.zeros(self.num_sectors, dtype=np.float32)
        for sec_idx in range(self.num_sectors):
            ru_id = self.sector_to_ru_map[sec_idx]
            mode = clamped_modes[sec_idx]
            
            if self.is_macro_ru[ru_id]:
                sector_rf_power[sec_idx] = self.rf_power_64t64r[mode]
            else:
                sector_rf_power[sec_idx] = self.rf_power_4t4r[mode]

        return sector_rf_power

    def calculate_analog_components_power(self, sleep_mode_manager: SleepModeManager) -> np.ndarray:
        return 0.25 * self.calculate_rf_processing_power(sleep_mode_manager)

    def calculate_power_amplifier_power(self, sleep_mode_manager: SleepModeManager) -> np.ndarray:
        return 0.75 * self.calculate_rf_processing_power(sleep_mode_manager)

    def calculate_total_power(self, sleep_mode_manager: SleepModeManager) -> np.float32:
        analog_power = self.calculate_analog_components_power(sleep_mode_manager)
        digital_power = self.calculate_digital_components_power(sleep_mode_manager)
        pa_power = self.calculate_power_amplifier_power(sleep_mode_manager)
        
        self.power_consumption_sector_matrix = (analog_power + digital_power + pa_power).astype(np.float16)
        
        return np.float32(np.sum(self.power_consumption_sector_matrix))

    def calculate_energy_efficiency(self, sleep_mode_manager: SleepModeManager, kpi_handler: SimulationKPIHandler, step: int) -> float:
        throughput_mbps = np.sum(kpi_handler.calculate_throughput_mbps(step))
        total_power_watts = self.calculate_total_power(sleep_mode_manager)
        
        if total_power_watts == 0:
            return 0.0
            
        return throughput_mbps / total_power_watts