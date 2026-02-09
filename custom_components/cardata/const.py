# Copyright (c) 2025, Renaud Allard <renaud@allard.it>, Kris Van Biesen <kvanbiesen@gmail.com>, Jyri Saukkonen <jyri.saukkonen+jjyksi@gmail.com>
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice,
#    this list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
# ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE
# LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR
# CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF
# SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS
# INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
# CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE)
# ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE, EVEN IF ADVISED OF THE
# POSSIBILITY OF SUCH DAMAGE.

"""Constants for the BMW CarData integration."""

DOMAIN = "cardata"

# Location descriptors
LOCATION_LATITUDE_DESCRIPTOR = "vehicle.cabin.infotainment.navigation.currentLocation.latitude"
LOCATION_LONGITUDE_DESCRIPTOR = "vehicle.cabin.infotainment.navigation.currentLocation.longitude"
LOCATION_HEADING_DESCRIPTOR = "vehicle.cabin.infotainment.navigation.currentLocation.heading"
LOCATION_ALTITUDE_DESCRIPTOR = "vehicle.cabin.infotainment.navigation.currentLocation.altitude"

# Window descriptors for sensor icons
WINDOW_DESCRIPTORS = (
    "vehicle.cabin.window.row1.driver.status",
    "vehicle.cabin.window.row1.passenger.status",
    "vehicle.cabin.window.row2.driver.status",
    "vehicle.cabin.window.row2.passenger.status",
    "vehicle.body.trunk.window.isOpen",
)

# Battery descriptors for device class detection
BATTERY_DESCRIPTORS = {
    "vehicle.drivetrain.batteryManagement.header",
    "vehicle.drivetrain.electricEngine.charging.level",
    "vehicle.powertrain.electric.battery.stateOfCharge.target",
    "vehicle.trip.segment.end.drivetrain.batteryManagement.hvSoc",
}

# Predicted SOC sensor (calculated during charging)
PREDICTED_SOC_DESCRIPTOR = "vehicle.predicted_soc"

# Magic SOC sensor (driving consumption prediction)
MAGIC_SOC_DESCRIPTOR = "vehicle.magic_soc"

DEFAULT_SCOPE = "authenticate_user openid cardata:api:read cardata:streaming:read"
DEVICE_CODE_URL = "https://customer.bmwgroup.com/gcdm/oauth/device/code"
TOKEN_URL = "https://customer.bmwgroup.com/gcdm/oauth/token"
API_BASE_URL = "https://api-cardata.bmwgroup.com"
API_VERSION = "v1"
BASIC_DATA_ENDPOINT = "/customers/vehicles/{vin}/basicData"
DEFAULT_STREAM_HOST = "customer.streaming-cardata.bmwgroup.com"
DEFAULT_STREAM_PORT = 9000
# How often to refresh the auth tokens in seconds
DEFAULT_REFRESH_INTERVAL = 45 * 60
MQTT_KEEPALIVE = 30
DEBUG_LOG = False
DIAGNOSTIC_LOG_INTERVAL = 30  # How often we print stream logs in seconds
BOOTSTRAP_COMPLETE = "bootstrap_complete"
# Force telematics poll if no data (MQTT or telematics) received for this duration (seconds)
DATA_STALE_THRESHOLD = 2 * 60 * 60  # 2 hours
HTTP_TIMEOUT = 30  # Timeout for HTTP API requests in seconds
VEHICLE_METADATA = "vehicle_metadata"
OPTION_MQTT_KEEPALIVE = "mqtt_keepalive"
OPTION_DEBUG_LOG = "debug_log"
OPTION_DIAGNOSTIC_INTERVAL = "diagnostic_log_interval"
OPTION_ENABLE_MAGIC_SOC = "enable_magic_soc"

# Error message constants (for consistent error detection)
ERR_TOKEN_REFRESH_IN_PROGRESS = "Token refresh already in progress"

# Container Management
# If True, search for existing containers to reuse (prevents accumulation)
CONTAINER_REUSE_EXISTING = True
# If False, always create new container (saves 1 API call but may accumulate containers)
# Set to False for testing if you frequently change descriptors

HV_BATTERY_CONTAINER_NAME = "BMW CarData HV Battery"
HV_BATTERY_CONTAINER_PURPOSE = "High voltage battery telemetry"
HV_BATTERY_DESCRIPTORS = [
    # Current high-voltage battery state of charge
    "vehicle.drivetrain.batteryManagement.header",
    "vehicle.drivetrain.electricEngine.charging.acAmpere",
    "vehicle.drivetrain.electricEngine.charging.acVoltage",
    "vehicle.powertrain.electric.battery.preconditioning.automaticMode.statusFeedback",
    "vehicle.vehicle.avgAuxPower",
    "vehicle.powertrain.tractionBattery.charging.port.anyPosition.flap.isOpen",
    "vehicle.powertrain.tractionBattery.charging.port.anyPosition.isPlugged",
    "vehicle.drivetrain.electricEngine.charging.timeToFullyCharged",
    "vehicle.powertrain.electric.battery.charging.acLimit.selected",
    "vehicle.drivetrain.electricEngine.charging.method",
    "vehicle.body.chargingPort.plugEventId",
    "vehicle.drivetrain.electricEngine.charging.phaseNumber",
    "vehicle.trip.segment.end.drivetrain.batteryManagement.hvSoc",
    "vehicle.trip.segment.accumulated.drivetrain.electricEngine.recuperationTotal",
    "vehicle.drivetrain.electricEngine.remainingElectricRange",
    "vehicle.drivetrain.electricEngine.charging.timeRemaining",
    "vehicle.drivetrain.electricEngine.charging.hvStatus",
    "vehicle.drivetrain.electricEngine.charging.lastChargingReason",
    "vehicle.drivetrain.electricEngine.charging.lastChargingResult",
    "vehicle.powertrain.electric.battery.preconditioning.manualMode.statusFeedback",
    "vehicle.drivetrain.electricEngine.charging.reasonChargingEnd",
    "vehicle.powertrain.electric.battery.stateOfCharge.target",
    "vehicle.body.chargingPort.lockedStatus",
    "vehicle.drivetrain.electricEngine.charging.level",
    "vehicle.powertrain.electric.battery.stateOfHealth.displayed",
    "vehicle.vehicleIdentification.basicVehicleData",
    "vehicle.drivetrain.batteryManagement.batterySizeMax",
    "vehicle.drivetrain.batteryManagement.maxEnergy",
    "vehicle.powertrain.electric.battery.charging.power",
    "vehicle.drivetrain.electricEngine.charging.status",
]

# Minimum number of telemetry descriptors required to consider a vehicle as "real"
# Vehicles with fewer descriptors are likely "ghost" cars from family sharing with limited access
MIN_TELEMETRY_DESCRIPTORS = 5

# SOC Learning parameters
# Learning rate for Exponential Moving Average (0.2 = 20% new, 80% old)
LEARNING_RATE = 0.2
# Minimum SOC gain required to learn from a session (percentage)
MIN_LEARNING_SOC_GAIN = 5.0
# Valid efficiency bounds - reject outliers outside this range
MIN_VALID_EFFICIENCY = 0.82
MAX_VALID_EFFICIENCY = 0.98
# Tolerance for matching target SOC (percentage) - if within this, finalize immediately
TARGET_SOC_TOLERANCE = 2.0
# Grace period for BMW SOC update after charge ends (minutes)
DC_SESSION_FINALIZE_MINUTES = 5.0
AC_SESSION_FINALIZE_MINUTES = 15.0
# Storage key and version for learned efficiency data
SOC_LEARNING_STORAGE_KEY = "cardata.soc_learning"
SOC_LEARNING_STORAGE_VERSION = 2
# Maximum gap between energy readings before skipping integration (seconds)
MAX_ENERGY_GAP_SECONDS = 600

# Driving consumption learning parameters
DEFAULT_CONSUMPTION_KWH_PER_KM = 0.21  # BMW BEV fleet average
MIN_VALID_CONSUMPTION = 0.10
MAX_VALID_CONSUMPTION = 0.40
MIN_LEARNING_TRIP_DISTANCE_KM = 5.0
MIN_LEARNING_SOC_DROP = 2.0
DRIVING_SOC_CONTINUITY_SECONDS = 300  # 5 min window for isMoving flap tolerance
DRIVING_SESSION_MAX_AGE_SECONDS = 4 * 60 * 60  # 4 hours
GPS_MAX_STEP_DISTANCE_M = 2000  # Max single GPS step (m) — reject jumps after tunnel/lost signal
AUX_EXTRAPOLATION_MAX_SECONDS = 600  # Stop extrapolating aux power after 10 min without update
MAX_AUX_POWER_KW = 10.0  # Sanity cap: reject aux power readings above this (bogus data)
REFERENCE_LEARNING_TRIP_KM = 30.0  # Reference distance for weighting learning: short trips contribute less

# Model-to-consumption mapping (kWh/km, real-world averages)
# Keys matched by prefix against modelName/series, longest match first
DEFAULT_CONSUMPTION_BY_MODEL: dict[str, float] = {
    "iX1": 0.17,
    "iX2": 0.17,
    "iX3": 0.20,
    "iX M60": 0.26,
    "iX xDrive50": 0.24,
    "iX xDrive40": 0.22,
    "iX": 0.23,
    "i4 M50": 0.21,
    "i4": 0.19,
    "i5 M60": 0.22,
    "i5": 0.20,
    "i7 M70": 0.26,
    "i7 xDrive60": 0.25,
    "i7": 0.24,
}

# Key for storing deduplicated allowed VINs in entry data
ALLOWED_VINS_KEY = "allowed_vins"
