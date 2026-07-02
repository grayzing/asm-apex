from geometry_helper import GeometryHelper
from base_station_manager import BaseStationManager
from sector_manager import SectorManager
from device_manager import DeviceManager
from radio_channel_model import RadioChannelModel
from handover_manager import RSRPBasedHandoverManager
from traffic_generator import TrafficGenerator, ParetoDistributionTrafficGenerator
from scheduler import QueueAwareProportionalFairPhysicalResourceBlockScheduler

if __name__ == "__main__":
    num_sectors = 3
    num_devices = 5

    base_station_manager = BaseStationManager(num_base_stations=2)
    base_station_manager.base_station_position_matrix[0] = [0.0, 0.0, 25.0]  # Base Station 1 at (0, 0, 30)
    base_station_manager.base_station_position_matrix[1] = [100.0, 0.0, 25.0]  # Base Station 2 at (100, 0, 30)
    sector_manager = SectorManager(num_sectors=num_sectors)
    sector_manager.sector_parent_base_station_vector[0] = 0  # Sector 1 belongs to Base Station 1
    sector_manager.sector_parent_base_station_vector[1] = 0  # Sector 2 belongs to Base Station 1
    sector_manager.sector_parent_base_station_vector[2] = 1  # Sector 3 belongs to Base Station 1
    device_manager = DeviceManager(num_devices=num_devices)
    device_manager.device_position_matrix[0] = [560.0, 56.0, 1.5]   # Device 1
    device_manager.device_position_matrix[1] = [210.0, -140.0, 1.5]  # Device 2
    device_manager.device_position_matrix[2] = [50.0, 50.0, 1.5]   # Device 3
    device_manager.device_position_matrix[3] = [280.0, -165.0, 1.5]  # Device 4
    device_manager.device_position_matrix[4] = [120.0, 180.0, 1.5]

    geometry_helper = GeometryHelper(num_sectors=num_sectors, num_devices=num_devices)

    geometry_helper.update_distance_matrix(device_manager, sector_manager, base_station_manager)
    geometry_helper.update_relative_azimuth_angle_deg_matrix(device_manager, sector_manager, base_station_manager)
    geometry_helper.update_relative_zenith_angle_deg_matrix(device_manager, sector_manager, base_station_manager)

    radio_channel_model = RadioChannelModel(num_sectors=num_sectors, num_devices=num_devices)
    radio_channel_model.update_path_loss_matrix(geometry_helper, sector_manager, base_station_manager, device_manager)
    radio_channel_model.update_directional_gain_matrix(geometry_helper, sector_manager, device_manager)
    radio_channel_model.update_received_power_matrix_per_resource_element(sector_manager)
    radio_channel_model.update_sinr_dbm_matrix_per_slot(sector_manager)
    radio_channel_model.update_spectral_efficiency_matrix()
    traffic_generator = ParetoDistributionTrafficGenerator(num_devices=num_devices, window_length=5)
    traffic_generator.generate_device_downlink_bits_matrix()
    handover_manager = RSRPBasedHandoverManager(num_sectors=num_sectors, num_devices=num_devices)
    handover_manager.handover(sector_manager, device_manager, radio_channel_model)
    print("Directional Gain Matrix (dB):")
    print(radio_channel_model.directional_gain_matrix)
    print("Distance Matrix (meters):")
    print(geometry_helper.distance_matrix_meters_matrix)

    print("\nRelative Azimuth Angle Matrix (degrees):")
    print(geometry_helper.relative_azimuth_angle_deg_matrix)

    print("\nRelative Zenith Angle Matrix (degrees):")
    print(geometry_helper.relative_zenith_angle_deg_matrix)

    scheduler = QueueAwareProportionalFairPhysicalResourceBlockScheduler(num_sectors=num_sectors, num_devices=num_devices)
    scheduler.schedule(sector_manager, radio_channel_model, traffic_generator, handover_manager, device_manager, 0)
    print(device_manager.device_physical_resource_block_allocation_vector)
    print(sector_manager.sector_physical_resource_block_utilization)

