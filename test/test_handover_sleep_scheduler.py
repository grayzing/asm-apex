import numpy as np

from src.handover_manager import HandoverManager
from src.sleep_mode_manager import SleepModeManager
from src.sector_manager import SectorManager
from src.device_manager import DeviceManager
from src.scheduler import Scheduler
from src.traffic_model import ParetoDistributionTrafficModel


class DummyRadioModel:
    def __init__(self, rx_power):
        self.downlink_received_signal_power_dbm_matrix = np.array(rx_power, dtype=np.float32)


def test_handover_assigns_best_sector():
    handover = HandoverManager(num_sectors=2, num_devices=2)
    sector_mgr = SectorManager(num_sectors=2, num_base_stations=1)
    device_mgr = DeviceManager(num_devices=2)
    radio = DummyRadioModel([[0.0, -5.0], [-2.0, -1.0]])

    handover.handover_event(sector_mgr, device_mgr, radio)

    assert handover.get_attached_sector(0) == 0
    assert handover.get_attached_sector(1) == 1
    assert handover.is_device_attached(0, 0)
    assert not handover.is_device_attached(0, 1)


def test_scheduler_allocates_only_attached_devices():
    sector_mgr = SectorManager(num_sectors=2, num_base_stations=1)
    traffic_model = ParetoDistributionTrafficModel(time_frame_ms=1, num_devices=3, seed=1)
    handover_mgr = HandoverManager(num_sectors=2, num_devices=3)
    handover_mgr.handover_device(device_id=0, sector_id=0)
    handover_mgr.handover_device(device_id=1, sector_id=0)
    handover_mgr.handover_device(device_id=2, sector_id=1)

    traffic_model.num_bits_in_queue = np.array([1000, 2000, 3000], dtype=np.int64)
    scheduler = Scheduler(
        sector_manager=sector_mgr,
        traffic_model=traffic_model,
        handover_manager=handover_mgr,
        num_devices=3,
        num_sectors=2,
        num_prbs_per_sector=1,
    )

    rates = np.array([
        [1.0, 1.0, 10.0],
        [2.0, 2.0, 1.0],
    ], dtype=np.float32)
    allocation = scheduler.allocate_prbs(frame_index=0, sector_device_rates=rates)

    assert allocation.shape == (2, 3)
    assert allocation[0, 2] == 0
    assert allocation[1, 2] == 1


def test_sleep_mode_transitions_require_countdown_for_sm3():
    sleep_mgr = SleepModeManager(num_sectors=1)

    sleep_mgr.set_sleep_mode(0, 3)
    assert sleep_mgr.sector_sleep_mode_transition_counter[0] == 20
    assert not sleep_mgr.can_change_state(0)

    for _ in range(20):
        sleep_mgr.decrement_counters()

    assert sleep_mgr.sector_sleep_mode_transition_counter[0] == 0
    assert sleep_mgr.can_change_state(0)

    sleep_mgr.set_sleep_mode(0, 1)
    assert sleep_mgr.sector_sleep_mode_transition_counter[0] == 0
    assert sleep_mgr.sector_sleep_mode_vector[0] == 1


def test_sleep_mode_exit_from_sm3_requires_countdown():
    sleep_mgr = SleepModeManager(num_sectors=1)
    sleep_mgr.set_sleep_mode(0, 3)
    sleep_mgr.set_sleep_mode(0, 1)
    assert sleep_mgr.sector_sleep_mode_vector[0] == 1
    assert sleep_mgr.sector_sleep_mode_transition_counter[0] == 20


def test_handover_ignores_sm3_sectors_when_selecting_best():
    sector_mgr = SectorManager(num_sectors=2, num_base_stations=1)
    device_mgr = DeviceManager(num_devices=1)

    sleep_mgr = SleepModeManager(num_sectors=2)
    sleep_mgr.set_sleep_mode(1, 3)

    handover_mgr = HandoverManager(num_sectors=2, num_devices=1, sleep_mode_manager=sleep_mgr)
    radio = DummyRadioModel([[0.0], [10.0]])

    handover_mgr.handover_event(sector_mgr, device_mgr, radio)

    assert handover_mgr.get_attached_sector(0) == 0
    assert not handover_mgr.is_device_attached(0, 1)
