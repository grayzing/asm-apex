import logging

import matplotlib.pyplot as plt
import numpy as np

from src.handover_manager import HandoverManager
from src.network_topology_helper import HexagonalNetworkTopologyHelper
from src.mobility_helper import RandomWalkMobilityHelper
from src.radio_channel_model import RadioChannelModel
from src.scheduler import Scheduler
from src.sector_manager import SectorManager
from src.sleep_mode_manager import SleepModeManager
from src.simulation import Simulation
from src.traffic_model import ParetoDistributionTrafficModel


class SimpleRadioChannelModel(RadioChannelModel):
    def __init__(self, num_sectors: int, num_devices: int) -> None:
        super().__init__(num_sectors, num_devices)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    num_gnbs = 1
    num_devices = 50
    num_sectors_per_gnb = 3
    num_steps = 500

    traffic_model = ParetoDistributionTrafficModel(time_frame_ms=num_steps, num_devices=num_devices, seed=42)
    handover_mgr = HandoverManager(num_sectors=num_gnbs * num_sectors_per_gnb, num_devices=num_devices)
    sleep_mgr = SleepModeManager(num_sectors=num_gnbs * num_sectors_per_gnb)
    radio_model = SimpleRadioChannelModel(num_sectors=num_gnbs * num_sectors_per_gnb, num_devices=num_devices)
    sector_mgr = SectorManager(num_sectors=num_gnbs * num_sectors_per_gnb, num_base_stations=num_gnbs)
    scheduler = Scheduler(
        sector_manager=sector_mgr,
        traffic_model=traffic_model,
        handover_manager=handover_mgr,
        num_devices=num_devices,
        num_sectors=num_gnbs * num_sectors_per_gnb,
        num_prbs_per_sector=4,
        sleep_mode_manager=sleep_mgr,
    )
    topology_helper = HexagonalNetworkTopologyHelper(intersite_distance_meters=200.0, num_rings=0)
    mobility_helper = RandomWalkMobilityHelper()

    sim = Simulation(
        num_gnbs=num_gnbs,
        num_devices=num_devices,
        num_sectors_per_gnb=num_sectors_per_gnb,
        topology_helper=topology_helper,
        mobility_helper=mobility_helper,
        traffic_model=traffic_model,
        radio_channel_model=radio_model,
        scheduler=scheduler,
        handover_manager=handover_mgr,
        sleep_mode_manager=sleep_mgr,
    )

    sim.setup_network()
    sim.run(end_frame=num_steps)

    metrics = sim.get_metrics()

    plt.figure(figsize=(12, 8))
    plt.plot(metrics["total_throughput_mbps"], label="Total Throughput (Mbps)")
    plt.plot(metrics["average_throughput_mbps"], label="Average Throughput (Mbps)")
    plt.plot(metrics["p10_throughput_mbps"], label="10th Percentile Throughput (Mbps)")
    plt.xlabel("Frame")
    plt.ylabel("Megabits per second")
    plt.title("Simulation Throughput Metrics")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("simulation_metrics.png")
    plt.show()

    print("Metrics summary")
    print("Mean SINR:", np.mean(metrics["sinr"]))
    print("Mean total throughput (Mbps):", np.mean(metrics["total_throughput_mbps"]))
    print("Mean average throughput (Mbps):", np.mean(metrics["average_throughput_mbps"]))
    print("Mean p10 throughput (Mbps):", np.mean(metrics["p10_throughput_mbps"]))


if __name__ == "__main__":
    main()
