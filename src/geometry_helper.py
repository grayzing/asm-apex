import numpy as np
from device_manager import DeviceManager
from sector_manager import SectorManager
from base_station_manager import BaseStationManager

class GeometryHelper:
    def __init__(self, num_sectors: int, num_devices: int) -> None:
        self.num_sectors = num_sectors
        self.num_devices = num_devices

        self.relative_azimuth_angle_deg_matrix: np.ndarray = np.zeros((num_sectors, num_devices), dtype=np.float16)
        self.relative_zenith_angle_deg_matrix: np.ndarray = np.zeros((num_sectors, num_devices), dtype=np.float16)
        self.distance_matrix_meters_matrix: np.ndarray = np.zeros((num_sectors, num_devices), dtype=np.float16)

    def _resolve_sector_parent_base_station_vector(self, sector_manager: SectorManager, base_station_manager: BaseStationManager) -> np.ndarray:
        parent_bs_vector = np.asarray(sector_manager.sector_parent_base_station_vector, dtype=np.int64)
        if parent_bs_vector.size == 0:
            return np.zeros(self.num_sectors, dtype=np.int64)

        if parent_bs_vector.size < self.num_sectors:
            repeat_count = max(1, int(np.ceil(self.num_sectors / parent_bs_vector.size)))
            expanded = np.repeat(parent_bs_vector, repeat_count)[:self.num_sectors]
        else:
            expanded = parent_bs_vector[:self.num_sectors]

        num_base_stations = len(base_station_manager.get_all_positions())
        if num_base_stations <= 0:
            return np.zeros(self.num_sectors, dtype=np.int64)

        return np.clip(expanded, 0, num_base_stations - 1)

    def _wrap_signed_degrees(self, values: np.ndarray) -> np.ndarray:
        wrapped = ((values + 180.0) % 360.0) - 180.0
        return np.where((wrapped > 0) & (values < 0), wrapped - 180.0, wrapped).astype(np.float32)

    def update_distance_matrix(self, device_manager: DeviceManager, sector_manager: SectorManager, base_station_manager: BaseStationManager) -> None:
        # 1. Pull 3D positions from the managers
        # device_positions shape: (U, 3) where each row is [x, y, z]
        # bs_positions shape: (B, 3) where each row is [x, y, z]
        device_positions = device_manager.get_all_positions()
        bs_positions = base_station_manager.get_all_positions()
        
        # 2. Vectorized 3D displacement matrix via broadcasting
        # Dimensions: (1, B, 3) - (U, 1, 3) -> Matrix Shape: (U, B, 3)
        # Calculates the vector [dx, dy, dz] from every Base Station to every Device
        disp_matrix = bs_positions[np.newaxis, :, :] - device_positions[:, np.newaxis, :]
        
        # 3. Calculate Euclidean distance from Devices to Base Stations
        # Square components, sum along the coordinate axis, and take the square root
        # Shape: (U, B) -> Rows are UEs, Columns are Base Stations
        device_to_bs_distance = np.sqrt(np.sum(disp_matrix**2, axis=2))
        
        # Transpose to shape (B, U) -> Base Stations x Devices
        bs_to_device_distance = device_to_bs_distance.T
        
        # 4. Map Base Station rows to Sector dimensions via advanced indexing
        # Pull the 1D parent index vector from SectorManager. Shape: (M,)
        # Example: [0, 0, 0, 1, 1, 1] means sectors 0-2 are at BS 0, 3-5 are at BS 1
        parent_bs_vector = self._resolve_sector_parent_base_station_vector(sector_manager, base_station_manager)
        
        # Advanced indexing expands (B, U) directly into (M, U) by duplicating rows 
        # to perfectly match which sector belongs to which base station tower.
        self.distance_matrix_meters_matrix = bs_to_device_distance[parent_bs_vector, :].astype(np.float32)

    def update_relative_azimuth_matrix(self, device_manager: DeviceManager, sector_manager: SectorManager, base_station_manager: BaseStationManager) -> None:
        """
        Updates the M x U relative azimuth matrix (Sectors x Devices).
        Calculates absolute angles from base station towers, replicates them per sector,
        and subtracts each sector's unique boresight orientation.
        """
        # 1. Pull spatial coordinates from managers
        # device_positions shape: (U, 3) -> [x, y, z] per device
        # bs_positions shape: (B, 3) -> [x, y, z] per base station tower
        device_positions = device_manager.get_all_positions()
        bs_positions = base_station_manager.get_all_positions()
        
        # 2. Compute 2D displacement vectors (dx, dy) from Towers to UEs via broadcasting
        # Dimensions: (1, B) - (U, 1) -> Resulting Matrix Shape: (U, B)
        dx = device_positions[:, 0][:, np.newaxis] - bs_positions[:, 0][np.newaxis, :]
        dy = device_positions[:, 1][:, np.newaxis] - bs_positions[:, 1][np.newaxis, :]
        
        # 3. Calculate Absolute Grid Angles (Towers x UEs)
        # np.arctan2 inputs are (y, x) and outputs are in radians -> transpose to (B, U)
        theta_abs_rad = np.arctan2(dy, dx).T
        theta_abs_deg = np.degrees(theta_abs_rad) # Shape: (B, U)
        
        # 4. Map Base Station Grid to Sector Rows via Advanced Indexing
        # parent_bs_vector shape: (M,) -> maps each sector to its parent tower index
        parent_bs_vector = self._resolve_sector_parent_base_station_vector(sector_manager, base_station_manager)
        sector_theta_abs = theta_abs_deg[parent_bs_vector, :] # Expands shape from (B, U) to (M, U)
        
        # 5. Subtract Sector Boresight Heading and Normalize Boundaries
        # azimuth_orientation_deg_matrix shape: (M,) -> reshape to (M, 1) for column broadcasting
        boresight_angles = sector_manager.sector_azimuth_angle_deg_matrix[:, np.newaxis]
        
        # Relative Azimuth = Absolute Angle to User - Boresight Heading of Sector Antenna
        raw_relative_azimuth = sector_theta_abs - boresight_angles
        
        # Mathematical wrap to clip output strictly between [-180, 180] degrees
        # This prevents mathematical boundary jumps at the edge of the circular coordinate space
        self.relative_azimuth_angle_deg_matrix = self._wrap_signed_degrees(raw_relative_azimuth)

    def update_relative_zenith_matrix(self, device_manager: DeviceManager, sector_manager: SectorManager, base_station_manager: BaseStationManager) -> None:
        """
        Updates the M x U relative zenith matrix (Sectors x Devices).
        Computes absolute 3GPP Zenith angles from tower heights, replicates them per sector,
        and scales them against each sector's electrical downtilt angle configuration.
        """
        # 1. Pull 3D coordinates from managers
        device_positions = device_manager.get_all_positions()  # Shape: (U, 3) -> [x, y, z]
        bs_positions = base_station_manager.get_all_positions()  # Shape: (B, 3) -> [x, y, z]
        
        # 2. Compute vertical displacement vector (dz) from Towers to UEs via broadcasting
        # Dimensions: (1, B) - (U, 1) -> Matrix Shape: (U, B)
        dz = device_positions[:, 2][:, np.newaxis] - bs_positions[:, 2][np.newaxis, :]
        
        # 3. Calculate 2D ground distances from Towers to UEs
        dx = device_positions[:, 0][:, np.newaxis] - bs_positions[:, 0][np.newaxis, :]
        dy = device_positions[:, 1][:, np.newaxis] - bs_positions[:, 1][np.newaxis, :]
        ground_dist = np.sqrt(dx**2 + dy**2) + 1e-6  # Epsilon prevents division by zero
        
        # 4. Calculate Absolute 3GPP Zenith Angle of Departure (ZOD)
        # ZOD is measured downward from the positive Z-axis. 
        # Formula: arctan2(ground_dist, dz) -> Transpose to get Shape (B, U)
        theta_zod_rad = np.arctan2(ground_dist, dz).T
        theta_zod_deg = np.degrees(theta_zod_rad)  # Shape: (B, U)
        theta_zod_deg = np.clip(theta_zod_deg, 0.0, 180.0)
        
        # 5. Map Base Station Grid to Sector Rows via Advanced Indexing
        parent_bs_vector = self._resolve_sector_parent_base_station_vector(sector_manager, base_station_manager)
        sector_theta_zod = theta_zod_deg[parent_bs_vector, :]  # Expands shape from (B, U) to (M, U)
        
        # 6. Adjust for Sector Electrical Downtilt to calculate Relative Zenith Angle
        # downtilt_deg_matrix shape: (M,) -> reshape to (M, 1) for column broadcasting
        # 3GPP Convention: Relative offset = Absolute ZOD - (90° + Downtilt)
        # This centers the relative angle at 0.0 when a user hits the peak of the downtilted beam.
        downtilts = sector_manager.downtilt_deg_matrix[:, np.newaxis]
        
        raw_relative_zenith = sector_theta_zod - (90.0 + downtilts)
        
        # Mathematical wrap to clip output strictly between [-180, 180] degrees
        self.relative_zenith_angle_deg_matrix = self._wrap_signed_degrees(raw_relative_zenith)
