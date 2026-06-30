import numpy as np

from src.handover_manager import HandoverManager
from src.sleep_mode_manager import SleepModeManager
from src.sector_manager import SectorManager
from src.scheduler import Scheduler
from src.traffic_model import ParetoDistributionTrafficModel


def test_scheduler_skips_sm3_sectors():
    sector_mgr = SectorManager(num_sectors=2, num_base_stations=1)
    traffic_model = ParetoDistributionTrafficModel(time_frame_ms=1, num_devices=2, seed=5)
    handover_mgr = HandoverManager(num_sectors=2, num_devices=2)
    handover_mgr.handover_device(device_id=0, sector_id=0)
    handover_mgr.handover_device(device_id=1, sector_id=1)

    sleep_mgr = SleepModeManager(num_sectors=2)
    sleep_mgr.set_sleep_mode(1, 3)

    traffic_model.num_bits_in_queue = np.array([1000, 1000], dtype=np.int64)
    scheduler = Scheduler(
        sector_manager=sector_mgr,
        traffic_model=traffic_model,
        handover_manager=handover_mgr,
        num_devices=2,
        num_sectors=2,
        num_prbs_per_sector=1,
        sleep_mode_manager=sleep_mgr,
    )

    rates = np.array([
        [1.0, 0.1],
        [0.1, 1.0],
    ], dtype=np.float32)
    allocation = scheduler.allocate_prbs(frame_index=0, sector_device_rates=rates)

    assert allocation.shape == (2, 2)
    assert allocation[0, 0] == 1
    assert allocation[1, 1] == 0
    assert allocation[1].sum() == 0
