"""Top-level package for the asm-apex simulation components."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

# Make the package's sibling modules importable under their legacy top-level names.
package_dir = Path(__file__).resolve().parent
if str(package_dir) not in sys.path:
    sys.path.insert(0, str(package_dir))

for module_name in ("device_manager", "sector_manager", "base_station_manager"):
    sys.modules.setdefault(module_name, importlib.import_module(f"src.{module_name}"))

from .base_station_manager import BaseStationManager
from .device_manager import DeviceManager
from .geometry_helper import GeometryHelper
from .sector_manager import SectorManager
from .scheduler import Scheduler
from .sleep_mode_manager import SleepModeManager
from .handover_manager import HandoverManager
from .simulation import Simulation
from .traffic_model import TrafficModel, ParetoDistributionTrafficModel

__all__ = [
    "BaseStationManager",
    "DeviceManager",
    "GeometryHelper",
    "SectorManager",
    "Scheduler",
    "SleepModeManager",
    "HandoverManager",
    "Simulation",
    "TrafficModel",
    "ParetoDistributionTrafficModel",
]
