"""Constants for the BMW CarData integration."""

DOMAIN = "cardata"
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
DIAGNOSTIC_LOG_INTERVAL = 30 # How often we print stream logs in seconds
BOOTSTRAP_COMPLETE = "bootstrap_complete"
REQUEST_LOG = "request_log"
REQUEST_LOG_VERSION = 1
REQUEST_LIMIT = 50 # API Quota
REQUEST_WINDOW_SECONDS = 24 * 60 * 60 # How long API Quota is reserved after API Call in seconds
TELEMATIC_POLL_INTERVAL = 40 * 60 # How often to call the Telematic API in seconds
VEHICLE_METADATA = "vehicle_metadata"
OPTION_MQTT_KEEPALIVE = "mqtt_keepalive"
OPTION_DIAGNOSTIC_INTERVAL = "diagnostic_log_interval"

HV_BATTERY_CONTAINER_NAME = "BimmerData HV Battery"
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