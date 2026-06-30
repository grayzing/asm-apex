import numpy as np

from src.base_station_manager import BaseStationManager
from src.device_manager import DeviceManager
from src.network_topology_helper import HexagonalNetworkTopologyHelper
from src.sector_manager import SectorManager


def test_hexagonal_network_topology_setup():
    helper = HexagonalNetworkTopologyHelper(intersite_distance_meters=500.0, num_rings=1)

    num_sites = 1 + 3 * 1 * (1 + 1)
    num_sectors = num_sites * 3

    device_mgr = DeviceManager(num_devices=1)
    bs_mgr = BaseStationManager(num_base_stations=num_sites, num_sectors=num_sectors)
    sector_mgr = SectorManager(num_sectors=num_sectors, num_base_stations=num_sites)

    helper.setup_network(device_mgr, bs_mgr, sector_mgr)

    assert bs_mgr.base_station_position_matrix.shape == (num_sites, 3)
    assert sector_mgr.sector_parent_base_station_vector.shape == (num_sectors,)
    assert np.all(sector_mgr.sector_parent_base_station_vector[:num_sectors] == np.repeat(np.arange(num_sites), 3))
    assert np.all(bs_mgr.base_station_sector_adjacency_matrix[:num_sectors, :num_sites] >= 0)
