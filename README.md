<p align="left">

  <img src="https://img.shields.io/badge/BMW%20CarData-Integration-blue?style=for-the-badge">
  &nbsp;

  <a href="https://github.com/kvanbiesen/bmw-cardata-ha">
    <img src="https://img.shields.io/badge/Maintainer-kvanbiesen-green?style=for-the-badge">
  </a>
  &nbsp;

  <a href="https://github.com/kvanbiesen/bmw-cardata-ha/releases">
    <img src="https://img.shields.io/github/v/release/kvanbiesen/bmw-cardata-ha?style=for-the-badge">
  </a>
  &nbsp;

  <a href="https://github.com/kvanbiesen/bmw-cardata-ha/releases/latest">
    <img src="https://img.shields.io/github/downloads/kvanbiesen/bmw-cardata-ha/latest/total?style=for-the-badge">
  </a>
  &nbsp;

  <a href="https://github.com/kvanbiesen/bmw-cardata-ha/releases">
    <img src="https://img.shields.io/github/downloads/kvanbiesen/bmw-cardata-ha/total?style=for-the-badge">
  </a>
  &nbsp;

  <a href="https://github.com/kvanbiesen/bmw-cardata-ha/issues">
    <img src="https://img.shields.io/github/issues/kvanbiesen/bmw-cardata-ha?style=for-the-badge">
  </a>
  &nbsp;

  <a href="https://github.com/kvanbiesen/bmw-cardata-ha/stargazers">
    <img src="https://img.shields.io/github/stars/kvanbiesen/bmw-cardata-ha?style=for-the-badge">
  </a>
  &nbsp;

  <a href="https://www.buymeacoffee.com/sadisticpandabear">
    <img src="https://img.shields.io/badge/Buy%20Me%20A%20Coffee-Donate-FFDD00?style=for-the-badge&logo=buymeacoffee">
  </a>

</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/kvanbiesen/bmw-cardata-ha/refs/heads/main/images/cardatalogo.png" alt="BMW Cardata logo" width="240" />
</p>

# BMW CarData for Home Assistant

Turn your BMW CarData stream into native Home Assistant entities. This integration subscribes to the BMW CarData MQTT stream (or an optional custom MQTT broker), keeps the token fresh automatically, and creates sensors/binary sensors for every descriptor that emits data.

> **Note:** This entire plugin was generated with the assistance of AI to quickly solve issues with the legacy implementation. The code is intentionally open—to-modify, fork, or build a new integration from it. PRs are welcome unless otherwise noted in the future.

> **Tested Environment:** Home Assistant 2025.3+ is required. Brand logos are included since HA 2026.3 via the `brand/` directory.

<a href="https://www.buymeacoffee.com/sadisticpandabear" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" style="height: 60px !important;width: 217px !important;" ></a>

Not required but appreciated :)

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Issues / Discussion
Please try to post only issues relevant to the integration itself on the [Issues](https://github.com/kvanbiesen/bmw-cardata-ha/issues) and keep all the outside discussion (problems with registration on BMWs side, asking for guidance, etc)

### Configure button actions
On the integration main page, there is now a "Configure" button. You can use it to:
- **Refresh authentication tokens** (will reload integration, might also need HA restart in some problem cases)
- **Start device authorization again** (redo the whole auth flow)
- **MQTT Broker** (switch stream source to a custom broker, including TLS mode and topic prefix)
- **Reset telemetry container** (delete and recreate the BMW telemetry container)
- **Clean up orphaned entities** (remove stale entities that no longer receive data)
- **Settings** (toggle optional features like Magic SOC, Charging History, Tyre Diagnosis, trip-end polling cooldown)

And manual API calls, these should be automatically called when needed, but if it seems that your device names aren't being updated, it might be worth it to run these manually.
- **Initiate vehicles (API)** (fetch all vehicle VINs on your account and create entities)
- **Get basic vehicle information (API)** (fetches vehicle details like model, series, etc. for all known VINs)
- **Get telematics data (API)** (fetches telematics data from the CarData API)

Note that every API call here counts towards your 50/24h quota!

# <u>Installation Instructions</u>


## BMW Portal Setup (DON'T SKIP, DO THIS FIRST - All Steps 1-13 before continuing)

The CarData web portal isn’t available everywhere (e.g., it’s disabled in Finland). You can still enable streaming by logging in by using supported region. It doesn't matter which language you select - all the generated Id and configuration is shared between all of them. 

**DO Steps 1-3 First before installing it in HACS**

### BMW 

- https://www.bmw.co.uk/en-gb/mybmw/vehicle-overview (in English)
- https://www.bmw.de/de-de/mybmw/vehicle-overview (in German)
- https://mybmw.bmwusa.com/ (USA we need testers or temp access)

### Mini

- https://www.mini.co.uk/en-gb/mymini/vehicle-overview (in English)
- https://www.mini.de/de-de/mymini/vehicle-overview (in German)

1. Select the vehicle you want to stream.
2. Choose **BMW CarData** or **Mini CarData**.
3. Generate a client ID as described here: https://bmw-cardata.bmwgroup.com/customer/public/api-documentation/Id-Technical-registration_Step-1
4. Under section CARDATA API, you see **Client ID**. Delete the original one and make a new one. Copy this new one to your clipboard because you will need it during **Configuration Flow** in Home Assistant.
   **Don't press the button Authenticate device (NEVER) **!!!!
5. Request access to **CarData API** first:
   - Click "Request access to CarData API"
   - ⏱️ **Wait 60 seconds** (BMW needs time to propagate permissions)
   Note, BMW portal seems to have some problems with scope selection. If you see an error on the top of the page, reload it, select one scope and wait for 120 seconds, then select another one and wait again.
6. Then request access to **CarData Stream**:
   - Click "Request access to CarData Stream"  
   - ⏱️ **Wait another 60 seconds**
   
   **Why?** BMW's backend needs time to activate permissions. Rushing causes 403 errors.
   
7. Scroll down to **CARDATA STREAMING** and press **Configure data stream** and on that new page, load all descriptors (keep clicking “Load more”).
8. Manually check every descriptor you want to stream or optionally to automate this, open the browser console (F12) and run:
```js
document.querySelectorAll('label.chakra-checkbox:not([data-checked])').forEach(l => l.click());
```
   - If you want the "Predicted SOC" helper sensor to work, make sure your telematics container includes the descriptors `vehicle.drivetrain.batteryManagement.header`, `vehicle.drivetrain.batteryManagement.maxEnergy`, `vehicle.powertrain.electric.battery.charging.power`, and `vehicle.drivetrain.electricEngine.charging.status`. Those fields let the integration reset the predicted state of charge and calculate the charging slope between stream updates. It seems like the `vehicle.drivetrain.batteryManagement.maxEnergy` always gets sent even though it's not explicitly set, but check it anyway.

9. Save the selection.
10. Repeat for all the cars you want to support
11. In Home Assistant, install this integration via HACS (see below under Installation (HACS)) and still in Home Assistant, step trough the Configuration Flow also described here below.
12. During the Home Assistant config flow, paste the client ID, visit the provided verification URL, enter the code (if asked), and approve. **Do not click Continue/Submit in Home Assistant until the BMW page confirms the approval**; submitting early leaves the flow stuck and requires a restart.
13. **If you get Error 500 during setup:**
    
    **Immediate actions:**
    - ❌ Remove the integration from Home Assistant
    - 🔄 Go to BMW portal → Delete current Client ID
    - ⏱️ **Wait 5 minutes** (BMW backend needs to clear old session)
    - ✅ Create new Client ID
    - ⏱️ **Wait another 2 minutes**
    - ✅ Try installation again
    
    **If error persists after 2-3 attempts:**
    - ⏱️ Wait 24 hours (you may have hit daily rate limit)
    - Try during different time of day (BMW servers less loaded)
14. Wait for the car to send data—triggering an action via the MyBMW app (lock/unlock doors) usually produces updates immediately. (older cars might need a drive before sensors start popping up, idrive6)

## Troubleshooting Setup Errors:

### Error 403 (Forbidden)
**Cause**: Authentication credentials incorrect or permissions not activated

**Solutions**:
1. ✅ Verify `clientid` is from BMW portal (NOT your login email)
2. ✅ Ensure both "CarData API" AND "CarData Stream" are enabled
3. ✅ Wait 2-3 minutes after enabling permissions before trying again
4. ✅ Delete and regenerate Client ID if permissions were recently changed
5. ✅ Check that your BMW account has an active ConnectedDrive subscription

### Error 500 (Server Error)
**Cause**: BMW API temporary issue or rate limiting

**Solutions**:
1. ⏱️ Wait 5-10 minutes before retrying
2. 🔄 Delete integration, create new Client ID in BMW portal
3. 🔄 Try setup during off-peak hours (early morning/late evening)
4. ✅ Ensure you didn't click "Authenticate device" in BMW portal (skip this!)
5. 📧 If persistent, contact BMW CarData support - may be account-specific issue -> bmwcardata-b2c-support@bmwgroup.com

### Error: "Stuck on waiting for approval"
**Cause**: Submitted HA config flow before BMW page confirmed approval

**Solution**:
1. 🛑 Wait for BMW page to show: "Device authenticated successfully"
2. ✅ Only then click "Submit" in Home Assistant
3. If already stuck: Restart Home Assistant and start over

## Installation (HACS)

1. Add this repo to HACS as a **custom repository** (type: Integration).
2. Install "Bmw cardata" from the Custom section.
3. Restart Home Assistant.

## Configuration Flow

1. Go to **Settings → Devices & Services → Add Integration** and pick **Bmw cardata**.
2. Enter your CarData **client ID** (created in the BMW portal and seen under section CARDATA API and there copied to your clipboard).
3. The flow displays a `verification_url` and `user_code`. Open the link, enter the code, and approve the device.
4. Once the BMW portal confirms the approval, return to HA and click Submit. If you accidentally submit before finishing the BMW login, the flow will hang until the device-code exchange times out; cancel it and start over after completing the BMW login.
5. If you remove the integration later, you can re-add it with the same client ID—the flow deletes the old entry automatically.
6. Small tip, on newer cars with Idrive7, you can force the sensor creation by opening the BMW/Mini App and press lock doors; on older ones like idrive6, You have to start the car, maybe even drive it a little bit

### Reauthorization
If BMW rejects the token (e.g. because the portal revoked it), please use the Configure > Start Device Authorization Again tool

### Custom MQTT Broker (optional)

You can switch the live stream from BMW's MQTT endpoint to your own broker (for example via [bmw-mqtt-bridge](https://dj0abr.github.io/bmw-mqtt-bridge/)).
BMW authorization is still required; only stream transport changes.
Expected topic format is `<topic_prefix><VIN>` (default prefix `bmw/`) and payload JSON must include `vin` and `data`.
Configure it in Home Assistant via **Settings -> Devices & Services -> BMW CarData -> Configure -> MQTT Broker**.

## Entity Naming & Structure

- Each VIN becomes a device in HA (`VIN` pulled from CarData).
- Sensors/binary sensors are auto-created and named from descriptors (e.g. `Cabin Door Row1 Driver Is Open`).
- Additional attributes include the source timestamp.
- All numeric sensors declare `suggested_display_precision`, so unit conversions (e.g. km to miles) display clean rounded values in Lovelace cards. You can override the display unit per entity via the gear icon in the entity settings, or switch your HA unit system to imperial for a global change.

## Vehicle Dashboard Card (Lovelace)

The integration automatically registers a built-in Lovelace vehicle card — no manual YAML or HACS frontend install needed.

In Home Assistant:
- Go to a dashboard → **Edit dashboard** → **Add card**
- Search for **BMW CarData Vehicle**
- In the card editor, select which vehicle to display

<p align="center">
  <img src="images/card-preview.png" alt="BMW CarData vehicle card preview" width="720" />
</p>

Available configuration options:

| Option | Default | Description |
|--------|---------|-------------|
| `device_id` | *(required)* | Device ID of the BMW vehicle (from the cardata integration) |
| `license_plate` | *(empty)* | License plate number, shown instead of VIN when set |
| `show_indicators` | `true` | Status indicator row (locks, doors, windows, alarm). Windows, tailgate, and hood only show red when the car is parked and locked with the item open (walked-away alert). |
| `show_range` | `true` | Battery / fuel level bar with range |
| `show_image` | `true` | Vehicle image |
| `show_map` | `true` | Inline location map |
| `show_buttons` | `true` | Quick-info tiles (location, mileage, service) |

### YAML Configuration

You can also add the card manually in YAML. The card type is `custom:bmw-cardata-vehicle-card`.

To find the `device_id`, go to **Settings → Devices & Services → BMW CarData**, click on a vehicle device, and copy the device ID from the URL (the hex string after `/config/devices/device/`).

Minimal example:

```yaml
type: custom:bmw-cardata-vehicle-card
device_id: abcdef1234567890abcdef1234567890
```

Full example with all options:

```yaml
type: custom:bmw-cardata-vehicle-card
device_id: abcdef1234567890abcdef1234567890
license_plate: AB 123 CD
show_indicators: true
show_range: true
show_image: true
show_map: true
show_buttons: true
```

To hide the map and quick-info tiles:

```yaml
type: custom:bmw-cardata-vehicle-card
device_id: abcdef1234567890abcdef1234567890
show_map: false
show_buttons: false
```

## Debug Logging
Set `DEBUG_LOG = True` in `custom_components/cardata/const.py` for detailed MQTT/auth logs (disabled by default). To reduce noise, change it to `False` and reload HA.

## Predicted SOC with Learning

The integration includes a predicted SOC (State of Charge) sensor that estimates battery charge during charging sessions. This sensor uses real-time accumulated energy (trapezoidal integration of charging power minus auxiliary consumption) to calculate charging progress more frequently than BMW's native SOC updates. This handles varying power levels naturally (DC taper above 80%, cold-battery ramp-up, grid fluctuations).

### Vehicles Without Power Telemetry

Some older models (e.g. i3s, iDrive 6 cars) do not report charging power or voltage/current via MQTT. For these vehicles, the integration derives an implied charging power from BMW SOC changes: when an API poll delivers a higher SOC than the previous anchor and no real power data exists, the average power is back-calculated from the SOC delta, elapsed time, and default efficiency. The heartbeat then extrapolates between polls using this derived value, giving smooth SOC progression instead of staircase jumps every 30 minutes. Real power data (if it arrives later) overwrites the derived value automatically.

### How Learning Works

The predicted SOC sensor automatically learns your vehicle's charging efficiency:

- **AC charging efficiency**: Starts at 90%, learns per charging condition (phases, voltage, current)
- **DC charging efficiency**: Starts at 93%, learns from actual DC sessions
- Both AC and DC use the same **efficiency matrix** with per-condition outlier detection, history tracking, and trend analysis
- Uses **Exponential Moving Average (EMA)** with adaptive learning rate (converges fast initially, settles to 20%)
- Learning data persists across Home Assistant restarts
- Active charging sessions and pending sessions survive HA restarts (restored sessions skip learning to avoid polluted data from energy gaps)
- After HA restart, stale SOC data from previous sessions is detected and rejected using BMW-provided timestamps, preventing false re-anchoring of the predicted SOC

### Learning Requirements

For learning to occur, a charging session must meet these criteria:
- Minimum 5% SOC gain during the session
- Calculated efficiency between 40% and 98% (outliers are rejected)
- Per-condition outlier detection (2-sigma after 5+ history entries for the same charging condition)
- Valid power data recorded throughout the session

### Session Finalization

Learning happens when a charging session ends:
- **Target reached**: If charging stops within 2% of the target SOC, learning happens immediately
- **Charge interrupted**: If stopped before target, waits for BMW SOC confirmation:
  - DC charging: 5-minute grace period
  - AC charging: 15-minute grace period

### PHEV-Specific Behavior

For Plug-in Hybrid Electric Vehicles (PHEVs), the predicted SOC has special handling:

- **Automatic PHEV detection**: Vehicles with both an HV battery and fuel system are detected as PHEVs, unless metadata (driveTrain/propulsionType) or the model name (e.g. i4, iX, i5) identifies them as a known BEV
- **Sync down on battery depletion**: If the actual BMW SOC is lower than the predicted value, the prediction syncs down immediately. This handles scenarios where the hybrid system depletes the battery (e.g., battery recovery mode, engine-priority driving)
- **Header filtering during charging**: When `charging.level` is available and fresh, `batteryManagement.header` is only allowed through if it would sync UP (header above prediction). Stale header values frozen at the pre-charge level are always below the prediction and get blocked, while legitimate mid-charge header updates above the prediction are allowed through for re-anchoring
- **Charging level ignored for sync-down**: During charging, `charging.level` (BMW's own prediction) is ignored when lower than our energy-based prediction, which tracks the real battery more accurately
- **BEVs**: For pure electric vehicles, the predicted SOC only syncs when not actively charging (standard behavior)

This ensures the predicted SOC stays accurate for PHEVs even when the hybrid system uses battery power in ways that don't register as "discharging" in the BMW API.

**Charging prediction accuracy**: On PHEVs with small batteries, you may notice a small step (typically 2-3 percentage points) at the end of charge when BMW's real SOC arrives and syncs up with the energy-based prediction. Small batteries amplify any efficiency estimation error because each kWh represents a larger percentage of total capacity. This is normal. Charging efficiency varies between sessions due to temperature, charge depth, and grid voltage fluctuations, so the learned EMA efficiency cannot perfectly predict each individual session. The step is largest during the first 5-10 sessions for a given charging configuration (phases, voltage, current) and shrinks as the EMA converges, but some residual variation is expected even after 20+ sessions. On BEVs with larger batteries the same absolute error translates to a much smaller percentage step. The prediction tracks the real battery closely throughout charging and the sync-up at the end ensures the final value is always accurate.

### Reset Buttons

Each EV/PHEV vehicle gets button entities to reset learned data:
- **Reset AC Learning**: Clears all AC entries from the efficiency matrix (resets to default 90%)
- **Reset DC Learning**: Clears all DC entries from the efficiency matrix (resets to default 93%)
- **Reset Magic Learning**: Clears driving consumption learning (resets to model default). Only appears when Magic SOC is enabled.

These buttons appear in the vehicle's device page under Configuration entities.

### Manual Battery Capacity

Each EV/PHEV vehicle gets a **Manual Battery Capacity** number entity (disabled by default) under Configuration entities. When set to a value above 0, it overrides the automatic capacity detection (BMW `maxEnergy` / `batterySizeMax`) for both charging and driving SOC prediction. Useful when BMW reports incorrect capacity values. Set to 0 to return to automatic detection.

### Manual Tank Capacity

Vehicles with fuel data get a **Manual Tank Capacity** number entity (disabled by default) under Configuration entities. When set to a value above 0 (in litres), the vehicle card computes a fuel level percentage from the remaining fuel and displays a progress bar. Useful for vehicles where BMW does not send `fuelSystem.level` (the percentage descriptor) and only sends `remainingFuel` (absolute litres). Set to 0 to disable.

## Magic SOC — Driving Consumption Prediction (Experimental)

Magic SOC predicts battery drain during driving using real-time odometer distance and learned consumption rates. It provides a sub-integer SOC estimate that updates more frequently than BMW's native integer SOC. **Disabled by default** — enable via Settings.

- **Enable via**: Settings → Devices & Services → BMW CarData → Configure → Settings → Enable Magic SOC
- **Sensor**: `vehicle.magic_soc` per vehicle (BEV only; PHEVs get passthrough BMW SOC)
- **How it works**: Anchors on BMW's reported SOC at trip start, then subtracts `distance × learned_consumption / capacity` as the vehicle drives. Re-anchors when BMW sends a fresh SOC mid-drive — if the drift is < 0.5pp, keeps the sub-integer prediction for smoother display.
- **Consumption learning**: Uses EMA (Exponential Moving Average) with adaptive rate. Default 0.21 kWh/km globally, with per-model defaults (e.g. i4 eDrive40 = 0.18 kWh/km). Learns from completed trips where both SOC drop and distance are available. Requires at least 5 trips before the learning rate settles to 20%.
- **Trip detection**: Combines BMW's `isMoving` signal, GPS-derived motion, and odometer changes. Handles MQTT bursts and GPS gaps gracefully.
- **Capacity**: Uses live `maxEnergy` from BMW (reflects real degradation), falls back to `batterySizeMax`, then to per-model defaults.
- **Reset**: A "Reset Magic Learning" button appears under Configuration entities for each BEV.

**Limitations**: This is experimental. Accuracy depends on BMW sending timely SOC and mileage data. Preheating, extended idle with accessories, and firmware glitches can cause temporary inaccuracy. PHEVs are excluded from prediction (hybrid powertrain makes distance-based estimation unreliable).

## Charging History (Optional)

The integration can fetch your BMW charging session history from the past 30 days. This is **disabled by default** to conserve your API quota.

- **Enable via**: Settings → Devices & Services → BMW CarData → Configure → Settings → Enable Charging History
- **API cost**: 1 call per vehicle per day (from your 50-call daily quota)
- **Sensor**: Creates a diagnostic sensor per vehicle showing session count and last charge date
- **Attributes**: Summarised session data (start/end time, start/end SOC, energy consumed, duration, charging type). Full raw data available via the `cardata.fetch_charging_history` service
- **Manual trigger**: Use `cardata.fetch_charging_history` service in Developer Tools

## Tyre Diagnosis (Optional)

The integration can fetch tyre health and wear data from BMW's Smart Maintenance system. This is **disabled by default** to conserve your API quota.

- **Enable via**: Settings → Devices & Services → BMW CarData → Configure → Settings → Enable Tyre Diagnosis
- **API cost**: 1 call per vehicle per day (from your 50-call daily quota)
- **Sensor**: Creates a diagnostic sensor per vehicle showing aggregate tyre status
- **Attributes**: Per-wheel data including dimension, wear, season, manufacturer, defect status, and production date
- **Manual trigger**: Use `cardata.fetch_tyre_diagnosis` service in Developer Tools

## Developer Tools Services

Home Assistant's Developer Tools expose helper services for manual API checks:

- `cardata.fetch_telematic_data` fetches the current contents of the configured telematics container for a VIN and logs the raw payload.
- `cardata.fetch_vehicle_mappings` calls `GET /customers/vehicles/mappings` and logs the mapping details (including PRIMARY or SECONDARY status). Only primary mappings return data; some vehicles do not support secondary users, in which case the mapped user is considered the primary one.
- `cardata.fetch_basic_data` calls `GET /customers/vehicles/{vin}/basicData` to retrieve static metadata (model name, series, etc.) for the specified VIN.
- `cardata.fetch_charging_history` fetches the last 30 days of charging sessions for a VIN. Uses 1 API call per vehicle.
- `cardata.fetch_tyre_diagnosis` fetches tyre health and wear data for a VIN. Uses 1 API call per vehicle.
- `cardata.fetch_vehicle_images` manually fetches vehicle images for all configured vehicles.
- `cardata.clean_hv_containers` lists or deletes high-voltage battery telemetry containers (actions: `list`, `delete`, `delete_all`, `delete_all_matching`).
- `cardata.migrate_entity_ids` migrates entity IDs from old format to new format. Use `dry_run` to preview changes without applying them.

## API Quota and MQTT Streaming

BMW imposes a **50 calls/day** limit on the CarData API. This integration does not enforce the limit client-side — BMW's own 429 response is respected via backoff. API usage is minimized through MQTT freshness gating and rate limiting:

- **MQTT Stream (real-time)**: The MQTT stream is unlimited and provides real-time updates for events like door locks, motion state, charging power, etc. GPS coordinates are paired using BMW payload timestamps (same GPS fix detection) with an arrival-time fallback, so location updates work even when latitude and longitude arrive in separate MQTT messages. In direct BMW mode, token refresh during MQTT reconnection is lock-free to avoid blocking the connection, and the MQTT connection is proactively reconnected with fresh credentials to prevent session expiry (~1 hour).
- **Trip-end polling**: When a vehicle stops moving (trip ends), the integration triggers an immediate API poll to capture post-trip battery state. This ensures SOC is updated even when the MQTT stream only delivers GPS/mileage but not SOC (common on some models). A configurable per-VIN cooldown (default 10 minutes) prevents GPS burst flapping from burning API quota. A 30-second grace period after door unlock prevents brief intermediate stops (e.g. picking up a passenger) from fragmenting a trip and wasting API calls. Trip-end polling can be **disabled entirely** or the cooldown can be **increased** via Settings → Devices & Services → BMW CarData → Configure → Settings. Disabling it is useful for vehicles making many short trips (e.g. delivery drivers, nurses) that would otherwise exhaust the daily API quota.
- **Charge-end polling**: When charging completes or stops, the integration triggers an immediate API poll to get the actual BMW SOC for learning calibration of the predicted SOC sensor, subject to the same per-VIN cooldown.
- **Fallback polling**: The integration polls periodically as a fallback in case MQTT stream fails or after Home Assistant restarts. VINs with fresh MQTT data are skipped individually, so in multi-car setups only stale VINs consume API calls.
- **Daily optional features**: When Charging History and/or Tyre Diagnosis are enabled, each makes exactly 1 API call per vehicle per day regardless of whether the call succeeds or fails (no retries). The polling interval automatically increases to compensate — e.g. with both features on 2 cars, polling stretches from 2h to 2.4h per VIN.
- **Multi-VIN setups**: All vehicles share the same 50 call/day limit. The poll interval scales with VIN count plus any enabled daily features. Each VIN is guaranteed at least 1 poll per day; BMW's 429 backoff handles actual quota enforcement.
- **Rate limiting**: If BMW returns a rate-limited response (HTTP 429 or HTTP 403 with `CU-429` error code), the integration backs off automatically with exponential delay.

## Requirements

- BMW CarData account with streaming access (CarData API + CarData Streaming subscribed in the portal).
- Client ID created in the BMW portal (see "BMW Portal Setup").
- Home Assistant 2025.3+.
- TLS 1.3 capable SSL library (required for direct BMW MQTT mode): OpenSSL 1.1.1+, LibreSSL 3.2.0+, or equivalent.
- Familiarity with BMW's CarData documentation: https://bmw-cardata.bmwgroup.com/customer/public/api-documentation/Id-Introduction

## !! Recommended setup with people on multiple BMWs (not required, it's working as is but you're limiting yourself in accuracy since the hardcoded 50 limits a day) !!
- Car 1 -> email_1
- Car 2 -> mail_2
- .....
- Use those separate accounts in the integration
- Use mail_x+1 which has all the cars merged for the BMW app
- As said, not needed but then you live with outdated data (Hour x amount of cars on single account).

## Translations

The setup wizard, error messages, and options menu are translated into the following languages:

- English (en)
- German (de)
- French (fr)
- Italian (it)
- Dutch (nl)
- Spanish (es)
- Portuguese (pt)

Home Assistant automatically selects the translation matching your configured language. Entity names are not translated as they use BMW descriptor names with values and units.

## Project Architecture

The integration is organized into focused modules:

| Module | Purpose |
|--------|---------|
| `const.py` | Shared constants: descriptor paths, timeouts, domain identifiers |
| `__init__.py` | Thin entry point: delegates to `lifecycle.py` |
| `lifecycle.py` | Setup/unload orchestration, ghost device cleanup |
| `coordinator.py` | Central state management, message dispatch, entity signaling |
| `soc_wiring.py` | SOC/charging/driving prediction wiring between descriptors and prediction engines |
| `device_info.py` | Device metadata building, BEV detection, state restoration |
| `coordinator_housekeeping.py` | Diagnostics, stale VIN cleanup, old descriptor eviction, connection events |
| `soc_prediction.py` | Charging SOC: trapezoidal energy integration, session management |
| `soc_types.py` | Charging data types: `LearnedEfficiency`, `PendingSession`, `ChargingSession` |
| `soc_learning.py` | Charging efficiency EMA learning, session finalization, persistence |
| `magic_soc.py` | Driving SOC: distance-based consumption prediction, adaptive EMA learning |
| `stream.py` | MQTT connection management, credential hot-swap |
| `stream_circuit_breaker.py` | Circuit breaker for reconnection rate limiting |
| `stream_reconnect.py` | Reconnection, unauthorized handling, retry scheduling |
| `geo_utils.py` | Shared geographic utilities (Haversine distance) |
| `motion_detection.py` | GPS centroid movement detection, parking zone logic |
| `sensor_diagnostics.py` | Diagnostic sensors: connection, metadata, efficiency, charging history, tyre diagnosis |
| `sensor.py` / `binary_sensor.py` / `device_tracker.py` | Home Assistant entity platforms |
| `button.py` | Reset learning buttons (AC, DC, consumption) |
| `number.py` | Manual battery capacity and tank capacity input entities |
| `config_flow.py` | Setup and reauthorization UI flows |
| `options_flow.py` | Options menu: settings, MQTT broker, API actions, entity cleanup |
| `bootstrap.py` | VIN discovery, metadata fetch, container creation |
| `auth.py` | Token refresh loop, reauth flow, stream error handling |
| `runtime.py` | Per-entry runtime data, locks, session health |
| `telematics.py` | Scheduled API polling, trip-end/charge-end triggers |
| `container.py` | Telematic container CRUD, signature-based reuse |
| `services.py` | HA service handlers (API calls, migrations, container management) |
| `ratelimit.py` | Rate limiting: 429 backoff, unauthorized loop protection, container rate limiter |
| `http_retry.py` | HTTP retry with exponential backoff and jitter |
| `image.py` | Vehicle image fetching and caching |
| `frontend_cards.py` | Lovelace card backend: websocket API, resource registration |
| `migrations.py` | Entity ID format migration tool |

## Known Limitations

- Direct BMW MQTT has one-stream-per-GCID behavior: make sure no other direct clients are connected simultaneously. To share one upstream stream, use a bridge + custom MQTT broker mode.
- The CarData API is read-only; sending commands remains outside this integration.
- **Premature Continue in auth flow: If you hit Continue before authorizing on BMW's site, the device-code flow gets stuck. Cancel the flow and restart the integration (or Home Assistant) once you've completed the BMW login.**
- **Older models (i3, i3s, iDrive 6 cars, older F-series)**: These vehicles send telemetry very infrequently — typically only when the car is stopped/turned off and when charging reaches 100%. There are no real-time MQTT updates while charging or driving, so most sensors will appear stale between events. The Predicted SOC sensor can help during charging, but accuracy depends on receiving at least an initial SOC value. This is a BMW platform limitation, not a bug in the integration.
- **Neue Klasse (NK/NA5) vehicles**: BMW's CarData API does not currently provide `batteryManagement.header` or `charging.level` for NK vehicles, which means the main SOC source is missing. The integration falls back to `trip...hvSoc` (trip-end battery level) for the Magic SOC sensor. This is a best-effort fallback: the value is accurate right after a trip ends, but becomes stale after charging (it still shows the pre-charge level until the next trip). Charging prediction is not available. BMW may add proper NK support to CarData in the future.

## Stale Issue Policy

Issues that remain inactive for 1 week receive an automated reminder. A second reminder follows after 2 weeks, and a final warning after 3 weeks. Issues with no response after 4 weeks are automatically closed. Any comment from a non-bot user resets the cycle. Issues labeled `pinned` or `security` are exempt.

## License

This project is licensed under the BSD 2-Clause License - see the [LICENSE](LICENSE.md) file for details.

### Attribution

This software was created by [Kris Van Biesen](https://github.com/kvanbiesen). Taken over since no response for original developper (https://github.com/JjyKsi/bmw-cardata-ha). Please keep this notice if you redistribute or modify the code.
