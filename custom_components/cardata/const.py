"""Constants for the BMW CarData integration."""

from enum import Enum


class ConnectionState(Enum):
    """MQTT connection states."""
    DISCONNECTED = "disconnected"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    DISCONNECTING = "disconnecting"
    FAILED = "failed"


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

DEFAULT_SCOPE = "authenticate_user openid cardata:api:read cardata:streaming:read"
DEVICE_CODE_URL = "https://customer.bmwgroup.com/gcdm/oauth/device/code"
TOKEN_URL = "https://customer.bmwgroup.com/gcdm/oauth/token"
API_BASE_URL = "https://api-cardata.bmwgroup.com"
API_VERSION = "v1"
BASIC_DATA_ENDPOINT = "/customers/vehicles/{vin}/basicData"
DEFAULT_STREAM_HOST = "customer.streaming-cardata.bmwgroup.com"
DEFAULT_STREAM_PORT = 9000
DEFAULT_REFRESH_INTERVAL = 45 * 60  #How often to refresh the auth tokens in seconds
MQTT_KEEPALIVE = 30
DEBUG_LOG = False
DIAGNOSTIC_LOG_INTERVAL = 30 # How often we print stream logs in seconds
BOOTSTRAP_COMPLETE = "bootstrap_complete"
REQUEST_LOG = "request_log"
REQUEST_LOG_VERSION = 1
REQUEST_LIMIT = 400  # API Quota - 80% of BMW's ~500/day limit (leaves safety margin)
REQUEST_WINDOW_SECONDS = 24 * 60 * 60  # How long API Quota is reserved after API Call in seconds

# Quota thresholds for warnings (percentages of REQUEST_LIMIT)
QUOTA_WARNING_THRESHOLD = 280   # Warn at 70% of REQUEST_LIMIT
QUOTA_CRITICAL_THRESHOLD = 360  # Critical at 90% of REQUEST_LIMIT
TELEMATIC_POLL_INTERVAL = 40 * 60 # How often to call the Telematic API in seconds
HTTP_TIMEOUT = 30  # Timeout for HTTP API requests in seconds
VEHICLE_METADATA = "vehicle_metadata"
OPTION_MQTT_KEEPALIVE = "mqtt_keepalive"
OPTION_DEBUG_LOG = "debug_log"
OPTION_DIAGNOSTIC_INTERVAL = "diagnostic_log_interval"

# Container Management
CONTAINER_REUSE_EXISTING = True  # If True, search for existing containers to reuse (prevents accumulation)
                                  # If False, always create new container (saves 1 API call but may accumulate containers)
                                  # Set to False for testing if you frequently change descriptors

# Door descriptors for binary sensor device class
DOOR_DESCRIPTORS = (
    "vehicle.cabin.door.row1.driver.isOpen",
    "vehicle.cabin.door.row1.passenger.isOpen",
    "vehicle.cabin.door.row2.driver.isOpen",
    "vehicle.cabin.door.row2.passenger.isOpen",
)

DOOR_NON_DOOR_DESCRIPTORS = (
    "vehicle.body.trunk.isOpen",
    "vehicle.body.hood.isOpen",
    "vehicle.body.trunk.door.isOpen",
    "vehicle.body.trunk.left.door.isOpen",
    "vehicle.body.trunk.lower.door.isOpen",
    "vehicle.body.trunk.right.door.isOpen",
    "vehicle.body.trunk.upper.door.isOpen",
)

# GPS / Device Tracker constants
GPS_PAIR_WINDOW = 2.0  # seconds - lat/lon come in separate messages
GPS_MAX_STALE_TIME = 600  # seconds (10 minutes) - discard very old coordinates
GPS_MIN_MOVEMENT_DISTANCE = 3  # meters - minimum movement to trigger update
GPS_COORD_PRECISION = 0.000001  # degrees (~0.1 meter) - ignore smaller changes
EARTH_RADIUS_METERS = 6371000  # Earth radius for Haversine formula

# Coordinator debouncing
COORDINATOR_DEBOUNCE_SECONDS = 5.0  # Update every 5 seconds max
COORDINATOR_MIN_CHANGE_THRESHOLD = 0.01  # Minimum change for numeric values

# MQTT Stream connection constants
MQTT_RECONNECT_BACKOFF_INITIAL = 5  # Initial reconnect backoff in seconds
MQTT_RECONNECT_BACKOFF_MAX = 300  # Maximum reconnect backoff in seconds (5 minutes)
MQTT_RETRY_BACKOFF = 3  # Retry backoff multiplier
MQTT_MIN_RECONNECT_INTERVAL = 10.0  # Minimum seconds between reconnect attempts
MQTT_RECONNECT_DELAY_MIN = 5  # Paho client min reconnect delay
MQTT_RECONNECT_DELAY_MAX = 60  # Paho client max reconnect delay

# MQTT Circuit breaker
MQTT_CIRCUIT_BREAKER_THRESHOLD = 10  # Max failures before circuit opens
MQTT_CIRCUIT_BREAKER_WINDOW = 60  # Failure window in seconds
MQTT_CIRCUIT_BREAKER_DURATION = 300  # Circuit open duration in seconds (5 minutes)

# MQTT Return codes for authentication errors
MQTT_RC_BAD_CREDENTIALS = 4
MQTT_RC_UNAUTHORIZED = 5
MQTT_AUTH_ERROR_CODES = (MQTT_RC_BAD_CREDENTIALS, MQTT_RC_UNAUTHORIZED)
MQTT_UNAUTHORIZED_RETRY_WINDOW = 10  # seconds before retrying after auth error

# Entity configuration
ENTITY_NAME_WAIT_TIMEOUT = 2.0  # seconds to wait for vehicle name before setting entity name

# Auth token refresh
TOKEN_REFRESH_RETRY_WINDOW = 30  # seconds between token refresh attempts

# Platform configuration
PARALLEL_UPDATES = 0  # Disable parallel updates for entity platforms

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
    "vehicle.drivetrain.electricEngine.charging.status"

]