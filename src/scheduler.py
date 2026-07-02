import numpy as np

from sector_manager import SectorManager
from radio_channel_model import RadioChannelModel
from device_manager import DeviceManager
from traffic_generator import TrafficGenerator
from handover_manager import HandoverManager

from abc import ABC, abstractmethod

class PhysicalResourceBlockScheduler(ABC):
    def __init__(self, num_sectors: int, num_devices: int) -> None:
        self.num_sectors = num_sectors
        self.num_devices = num_devices

    @abstractmethod
    def schedule(self, sector_manager: SectorManager, radio_channel_model: RadioChannelModel, traffic_generator: TrafficGenerator, handover_manager: HandoverManager):
        raise NotImplementedError("Subclasses must implement this method.")
    
class QueueAwareProportionalFairPhysicalResourceBlockScheduler(PhysicalResourceBlockScheduler):
    def __init__(self, num_sectors: int, num_devices: int) -> None:
        super().__init__(num_sectors, num_devices)
        self.historical_throughput_matrix: np.ndarray = np.ones((num_devices, 1), dtype=np.float32) # in bits/sec
        self.alpha = 1.0
        self.ema_beta = 0.95  # Smoothing history window parameter for Proportional Fair

    def schedule(self, sector_manager: SectorManager, radio_channel_model: RadioChannelModel, traffic_generator: TrafficGenerator, handover_manager: HandoverManager, device_manager: DeviceManager, step: int):
        # 1. Map current SINR array values into the underlying link spectral efficiencies (bits/RE)
        radio_channel_model.update_spectral_efficiency_matrix()
        
        # 2. Extract current downlink backlog frames for this active simulation interval step
        # Matrix shape: (num_devices, window_length) -> slicing yields a 1D vector of shape (num_devices,)
        current_queues = traffic_generator.device_downlink_bits_matrix[:, step].copy()

        # 3. Calculate explicit bit payload metrics per single physical resource block
        # 12 subcarriers * 14 symbols * 80% (deducting 20% control channel and reference overhead)
        res_per_prb = int(12 * 14 * 0.80) 
        
        # Calculate individual PRB capacity array. Shape: (num_sectors, num_devices)
        bits_per_prb_matrix = radio_channel_model.spectral_efficiency_matrix * res_per_prb

        # 4. Generate the QAPF tracking metrics array: (R_prb / T_historical) * Queue^alpha
        historical_throughput_flat = self.historical_throughput_matrix.squeeze()
        pf_base_metric = bits_per_prb_matrix / historical_throughput_flat[np.newaxis, :]
        qapf_metric_matrix = pf_base_metric * (current_queues[np.newaxis, :] ** self.alpha)

        # 5. Determine total physical resource blocks structurally available per sector bandwidth allocation
        # Standard configuration maps 1 MHz to roughly 5 PRBs (e.g., 20 MHz -> 100 PRBs available)
        total_prbs_per_sector = (sector_manager.bandwidth_mhz_matrix * 5).astype(np.int32)

        # Output generation matrices
        prb_allocation_matrix = np.zeros((self.num_sectors, self.num_devices), dtype=np.int32)
        actual_transmitted_bits_matrix = np.zeros((self.num_sectors, self.num_devices), dtype=np.float32)

        # 6. Frequency Domain Loop Allocation
        for sector_idx in range(self.num_sectors):
            max_available_prbs = total_prbs_per_sector[sector_idx]
            if max_available_prbs <= 0:
                continue

            # Identify valid devices associated with this specific target sector via Handover Manager map matrix
            associated_device_mask = (handover_manager.sector_device_association_matrix[sector_idx, :] == 1)
            if not np.any(associated_device_mask):
                continue

            attached_device_indices = np.where(associated_device_mask)[0]

            # stronger fairness: give one PRB to each attached device first (if they have queue and any positive rate)
            for dev_idx in attached_device_indices:
                if max_available_prbs <= 0:
                    break

                queue_remaining = current_queues[dev_idx]
                bits_per_single_prb = bits_per_prb_matrix[sector_idx, dev_idx]
                if queue_remaining <= 0 or bits_per_single_prb <= 0:
                    continue

                prb_allocation_matrix[sector_idx, dev_idx] = 1
                max_available_prbs -= 1

                transmitted_bits = min(queue_remaining, bits_per_single_prb)
                actual_transmitted_bits_matrix[sector_idx, dev_idx] += transmitted_bits
                current_queues[dev_idx] -= transmitted_bits

            # Isolate metric rows strictly belonging to valid attached devices, mask independent paths to -1.0
            sector_metrics = np.where(associated_device_mask, qapf_metric_matrix[sector_idx, :], -1.0)
            
            # Sort utility vectors (highest prioritizing ranks mapped first)
            ranked_device_indices = np.argsort(sector_metrics)[::-1]

            for dev_idx in ranked_device_indices:
                if max_available_prbs <= 0 or sector_metrics[dev_idx] < 0:
                    break

                queue_remaining = current_queues[dev_idx]
                if queue_remaining <= 0:
                    continue

                bits_per_single_prb = bits_per_prb_matrix[sector_idx, dev_idx]
                if bits_per_single_prb <= 0:
                    continue

                # Deduce exactly how many physical resource blocks this device requires to exhaust its buffer queues
                prbs_needed = int(np.ceil(queue_remaining / bits_per_single_prb))
                prbs_assigned = min(prbs_needed, max_available_prbs)

                # Track systemic hardware distribution changes
                prb_allocation_matrix[sector_idx, dev_idx] += prbs_assigned
                max_available_prbs -= prbs_assigned

                # Process buffer data drops
                max_bits_capacity = prbs_assigned * bits_per_single_prb
                transmitted_bits = min(max_bits_capacity, queue_remaining)
                
                actual_transmitted_bits_matrix[sector_idx, dev_idx] += transmitted_bits

                current_queues[dev_idx] -= transmitted_bits

        total_tx_bits_per_device = np.sum(actual_transmitted_bits_matrix, axis=0)
        traffic_generator.device_downlink_bits_matrix[:, step] -= total_tx_bits_per_device

        self.historical_throughput_matrix *= self.ema_beta
        self.historical_throughput_matrix[:, 0] += (1 - self.ema_beta) * total_tx_bits_per_device
        self.historical_throughput_matrix = np.maximum(self.historical_throughput_matrix, 1e-3)

        allocated_prbs_per_sector = np.sum(prb_allocation_matrix, axis=1)
        
        safe_total_prbs = np.where(total_prbs_per_sector == 0, 1, total_prbs_per_sector)
        
        sector_manager.sector_physical_resource_block_utilization = np.maximum(0.05, (allocated_prbs_per_sector / safe_total_prbs)).astype(np.float32)
        device_manager.device_physical_resource_block_allocation_vector = np.sum(prb_allocation_matrix, axis=0).astype(np.int16)
        return prb_allocation_matrix, total_tx_bits_per_device
