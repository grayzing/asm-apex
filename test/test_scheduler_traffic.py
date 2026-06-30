import numpy as np

from src.handover_manager import HandoverManager
from src.sector_manager import SectorManager
from src.scheduler import Scheduler
from src.traffic_model import ParetoDistributionTrafficModel


def test_pareto_distribution_traffic_model_updates_queue_backlog():
    model = ParetoDistributionTrafficModel(
        time_frame_ms=2,
        num_devices=2,
        alpha=1.5,
        scale_bits=1000,
        max_bits_per_frame=2000,
        seed=42,
    )

    assert np.array_equal(model.num_bits_in_queue, np.zeros(2, dtype=np.int64))

    backlog = model.generate_bits_in_queue(frame_index=0)
    assert backlog.shape == (2,)
    assert np.all(backlog >= 0)
    assert np.array_equal(model.queue_size_matrix[:, 0], backlog)

    backlog_second = model.generate_bits_in_queue(frame_index=1)
    assert np.all(backlog_second >= backlog)
    assert np.array_equal(model.queue_size_matrix[:, 1], backlog_second)


def test_scheduler_allocates_prbs_with_priority_and_consumes_bits():
    sector_mgr = SectorManager(num_sectors=1, num_base_stations=1)
    traffic_model = ParetoDistributionTrafficModel(time_frame_ms=1, num_devices=2, seed=7)
    traffic_model.num_bits_in_queue = np.array([5000, 1000], dtype=np.int64)

    handover_mgr = HandoverManager(num_sectors=1, num_devices=2)
    handover_mgr.handover_device(device_id=0, sector_id=0)
    handover_mgr.handover_device(device_id=1, sector_id=0)
    scheduler = Scheduler(
        sector_manager=sector_mgr,
        traffic_model=traffic_model,
        handover_manager=handover_mgr,
        num_devices=2,
        num_sectors=1,
        num_prbs_per_sector=1,
    )

    rates = np.array([[1.0, 2.0]], dtype=np.float32)
    allocation = scheduler.allocate_prbs(frame_index=0, sector_device_rates=rates)

    assert allocation.shape == (1, 2)
    assert allocation.dtype == np.int16
    assert allocation[0, 0] == 1
    assert allocation[0, 1] == 0
    assert traffic_model.num_bits_in_queue[0] == 4000
    assert traffic_model.num_bits_in_queue[1] == 1000


def test_scheduler_respects_banned_devices():
    sector_mgr = SectorManager(num_sectors=1, num_base_stations=1)
    traffic_model = ParetoDistributionTrafficModel(time_frame_ms=1, num_devices=2, seed=9)
    traffic_model.num_bits_in_queue = np.array([5000, 1000], dtype=np.int64)

    handover_mgr = HandoverManager(num_sectors=1, num_devices=2)
    handover_mgr.handover_device(device_id=0, sector_id=0)
    handover_mgr.handover_device(device_id=1, sector_id=0)
    scheduler = Scheduler(
        sector_manager=sector_mgr,
        traffic_model=traffic_model,
        handover_manager=handover_mgr,
        num_devices=2,
        num_sectors=1,
        num_prbs_per_sector=1,
        banned_devices=np.array([0], dtype=np.int32),
    )

    rates = np.array([[1.0, 1.0]], dtype=np.float32)
    allocation = scheduler.allocate_prbs(frame_index=0, sector_device_rates=rates)

    assert allocation[0, 0] == 0
    assert allocation[0, 1] == 1


def test_traffic_consume_bits_clamps_to_zero():
    traffic_model = ParetoDistributionTrafficModel(time_frame_ms=1, num_devices=2, seed=0)
    traffic_model.num_bits_in_queue = np.array([500, 1000], dtype=np.int64)

    traffic_model.consume_bits(
        device_ids=np.array([0, 1], dtype=np.int32),
        bits=np.array([600, 500], dtype=np.int64),
    )

    assert traffic_model.num_bits_in_queue[0] == 0
    assert traffic_model.num_bits_in_queue[1] == 500
