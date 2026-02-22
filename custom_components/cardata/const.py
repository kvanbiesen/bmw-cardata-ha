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

# Individual descriptor constants (used across 3+ files)
DESC_SOC_HEADER = "vehicle.drivetrain.batteryManagement.header"
DESC_MAX_ENERGY = "vehicle.drivetrain.batteryManagement.maxEnergy"
DESC_BATTERY_SIZE_MAX = "vehicle.drivetrain.batteryManagement.batterySizeMax"
DESC_CHARGING_AC_VOLTAGE = "vehicle.drivetrain.electricEngine.charging.acVoltage"
DESC_CHARGING_AC_AMPERE = "vehicle.drivetrain.electricEngine.charging.acAmpere"
DESC_CHARGING_PHASES = "vehicle.drivetrain.electricEngine.charging.phaseNumber"
DESC_CHARGING_STATUS = "vehicle.drivetrain.electricEngine.charging.status"
DESC_CHARGING_LEVEL = "vehicle.drivetrain.electricEngine.charging.level"
DESC_CHARGING_POWER = "vehicle.powertrain.electric.battery.charging.power"
DESC_REMAINING_FUEL = "vehicle.drivetrain.fuelSystem.remainingFuel"
DESC_FUEL_LEVEL = "vehicle.drivetrain.fuelSystem.level"
DESC_TRAVELLED_DISTANCE = "vehicle.vehicle.travelledDistance"

# Lock acquisition timeout (seconds) — used for connect, credential, and token refresh locks
LOCK_ACQUIRE_TIMEOUT = 60.0

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
    DESC_SOC_HEADER,
    DESC_CHARGING_LEVEL,
    "vehicle.powertrain.electric.battery.stateOfCharge.target",
    "vehicle.trip.segment.end.drivetrain.batteryManagement.hvSoc",
}

# Predicted SOC sensor (calculated during charging)
PREDICTED_SOC_DESCRIPTOR = "vehicle.predicted_soc"

# Magic SOC sensor (driving consumption prediction)
MAGIC_SOC_DESCRIPTOR = "vehicle.magic_soc"

# Manual battery capacity (user input, takes priority over automatic detection)
MANUAL_CAPACITY_DESCRIPTOR = "vehicle.manual_battery_capacity"

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
# Telematic polling budget — target ~24 scheduled API polls/day, leaving headroom
# for bootstrap, trip-end events, etc. within BMW's 50-call daily quota.
# When daily optional features (charging history, tyre diagnosis) are enabled,
# the polling budget is reduced to keep total calls constant.
TARGET_DAILY_POLLS = 24
HTTP_TIMEOUT = 30  # Timeout for HTTP API requests in seconds
TRIP_POLL_COOLDOWN_SECONDS = 600  # Min seconds between trip-end polls per VIN
VEHICLE_METADATA = "vehicle_metadata"
OPTION_MQTT_KEEPALIVE = "mqtt_keepalive"
OPTION_DEBUG_LOG = "debug_log"

# Custom MQTT broker options
OPTION_CUSTOM_MQTT_ENABLED = "custom_mqtt_enabled"
OPTION_CUSTOM_MQTT_HOST = "custom_mqtt_host"
OPTION_CUSTOM_MQTT_PORT = "custom_mqtt_port"
OPTION_CUSTOM_MQTT_USERNAME = "custom_mqtt_username"
OPTION_CUSTOM_MQTT_PASSWORD = "custom_mqtt_password"
OPTION_CUSTOM_MQTT_TLS = "custom_mqtt_tls"  # "off", "tls", "tls_insecure"
OPTION_CUSTOM_MQTT_TOPIC_PREFIX = "custom_mqtt_topic_prefix"
DEFAULT_CUSTOM_MQTT_PORT = 1883
DEFAULT_CUSTOM_MQTT_TOPIC_PREFIX = "bmw/"
OPTION_DIAGNOSTIC_INTERVAL = "diagnostic_log_interval"
OPTION_ENABLE_MAGIC_SOC = "enable_magic_soc"
OPTION_ENABLE_CHARGING_HISTORY = "enable_charging_history"
OPTION_ENABLE_TYRE_DIAGNOSIS = "enable_tyre_diagnosis"

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
    DESC_SOC_HEADER,
    DESC_CHARGING_AC_AMPERE,
    DESC_CHARGING_AC_VOLTAGE,
    "vehicle.powertrain.electric.battery.preconditioning.automaticMode.statusFeedback",
    "vehicle.vehicle.avgAuxPower",
    "vehicle.powertrain.tractionBattery.charging.port.anyPosition.flap.isOpen",
    "vehicle.powertrain.tractionBattery.charging.port.anyPosition.isPlugged",
    "vehicle.drivetrain.electricEngine.charging.timeToFullyCharged",
    "vehicle.powertrain.electric.battery.charging.acLimit.selected",
    "vehicle.drivetrain.electricEngine.charging.method",
    "vehicle.body.chargingPort.plugEventId",
    DESC_CHARGING_PHASES,
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
    DESC_CHARGING_LEVEL,
    "vehicle.powertrain.electric.battery.stateOfHealth.displayed",
    "vehicle.vehicleIdentification.basicVehicleData",
    DESC_BATTERY_SIZE_MAX,
    DESC_MAX_ENERGY,
    DESC_CHARGING_POWER,
    DESC_CHARGING_STATUS,
]

# Minimum number of telemetry descriptors required to consider a vehicle as "real"
# Vehicles with fewer descriptors are likely "ghost" cars from family sharing with limited access
MIN_TELEMETRY_DESCRIPTORS = 5

# SOC Learning parameters
# Default DC charging efficiency (used before learning)
DEFAULT_DC_EFFICIENCY = 0.93
# Learning rate for Exponential Moving Average (0.2 = 20% new, 80% old)
LEARNING_RATE = 0.2
# Minimum SOC gain required to learn from a session (percentage)
MIN_LEARNING_SOC_GAIN = 5.0
# Valid efficiency bounds - reject outliers outside this range
MIN_VALID_EFFICIENCY = 0.40
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
REFERENCE_LEARNING_TRIP_KM = 30.0  # Reference distance for weighting learning: short trips contribute less

# Model-to-consumption mapping (kWh/km, real-world averages)
# Keys matched by prefix against modelName/series, longest match first
DEFAULT_CONSUMPTION_BY_MODEL: dict[str, float] = {
    # iX1 family (WLTP ~15.4-18.1)
    "iX1 xDrive30": 0.18,
    "iX1": 0.17,
    # iX2 family (WLTP ~15.6-17.7)
    "iX2 xDrive30": 0.18,
    "iX2": 0.17,
    # iX3 (old G08: WLTP ~18.5-18.9)
    "iX3": 0.20,
    # iX family (WLTP ~19.3-24.7)
    "iX M60": 0.24,
    "iX xDrive60": 0.21,
    "iX xDrive50": 0.22,
    "iX xDrive40": 0.22,
    "iX": 0.22,
    # i4 family (WLTP ~15.1-22.5)
    "i4 M50": 0.21,
    "i4 eDrive40": 0.18,
    "i4 eDrive35": 0.17,
    "i4": 0.18,
    # i5 family (WLTP ~15.1-20.6)
    "i5 M60": 0.20,
    "i5 eDrive40": 0.18,
    "i5 xDrive40": 0.18,
    "i5": 0.18,
    # i7 family (WLTP ~18.4-23.8)
    "i7 M70": 0.23,
    "i7 xDrive60": 0.21,
    "i7 eDrive50": 0.20,
    "i7": 0.21,
}

# Model-based default battery capacities (usable kWh, not gross)
DEFAULT_CAPACITY_BY_MODEL: dict[str, float] = {
    # iX1 family
    "iX1 xDrive30": 64.7,
    "iX1": 64.7,
    # iX2 family
    "iX2 xDrive30": 64.7,
    "iX2": 64.7,
    # iX3 (G08)
    "iX3": 74.0,
    # iX family
    "iX M60": 105.2,
    "iX xDrive60": 105.2,
    "iX xDrive50": 105.2,
    "iX xDrive40": 71.0,
    "iX": 76.6,
    # i4 family
    "i4 M50": 80.7,
    "i4 eDrive40": 80.7,
    "i4 eDrive35": 59.4,
    "i4": 80.7,
    # i5 family
    "i5 M60": 81.2,
    "i5 eDrive40": 81.2,
    "i5 xDrive40": 81.2,
    "i5": 81.2,
    # i7 family
    "i7 M70": 101.7,
    "i7 xDrive60": 101.7,
    "i7 eDrive50": 101.7,
    "i7": 101.7,
}

# Daily fetch interval for optional endpoints (charging history, tyre diagnosis)
DAILY_FETCH_INTERVAL = 86400  # 24 hours

# Key for storing deduplicated allowed VINs in entry data
ALLOWED_VINS_KEY = "allowed_vins"
