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

Turn your BMW CarData stream into native Home Assistant entities. This integration subscribes directly to the BMW CarData MQTT stream, keeps the token fresh automatically, and creates sensors/binary sensors for every descriptor that emits data.

> **Note:** This entire plugin was generated with the assistance of AI to quickly solve issues with the legacy implementation. The code is intentionally open—to-modify, fork, or build a new integration from it. PRs are welcome unless otherwise noted in the future.

> **Tested Environment:** since I adopted the project, I used the latest ha 2025.12 (2025.3+ is required)

<a href="https://www.buymeacoffee.com/sadisticpandabear" target="_blank"><img src="https://cdn.buymeacoffee.com/buttons/v2/default-yellow.png" alt="Buy Me A Coffee" style="height: 60px !important;width: 217px !important;" ></a>

Not required but appreciated :)

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

## Issues / Discussion
Please try to post only issues relevant to the integration itself on the [Issues](https://github.com/kvanbiesen/bmw-cardata-ha/issues) and keep all the outside discussion (problems with registration on BMWs side, asking for guidance, etc)

### Configure button actions
On the integration main page, there is now "Configure" button. You can use it to:
- Refresh authentication tokens (will reload integration, might also need HA restart in some problem cases)
- Start device authorization again (redo the whole auth flow. Not tested yet but should work ™️)

And manual API calls, these should be automatically called when needed, but if it seems that your device names aren't being updated, it might be worth it to run these manually. 
- Initiate Vehicles API call (Fetch all Vehicle VINS on your account and create entities out of them)
- Get Basic Vehicle Information (Fetches vehicle details like model, etc. for all known VINS)
- Get telematics data (Fetches a telematics data from the CarData API. This is a limited hardcoded subset compared to stream. I can add more if needed)

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
   Note, BMW portal seems to have some problems with scope selection. If you see an error on the top of the page, reload it, select one scope and wait for 120 seconds, then select the another one and wait agin.
6. Then request access to **CarData Stream**:
   - Click "Request access to CarData Stream"  
   - ⏱️ **Wait another 60 seconds**
   
   **Why?** BMW's backend needs time to activate permissions. Rushing causes 403 errors.
   
7. Scroll down to **CARDATA STREAMING** and press **Configure data stream** and on that new page, load all descriptors (keep clicking “Load more”).
8. Manually check every descriptor you want to stream or optionally to automate this, open the browser console (F12) and run:
```js
document.querySelectorAll('label.chakra-checkbox:not([data-checked])').forEach(l => l.click());
```
   - If you want the "Predicted SOC" helper sensor to work, make sure your telematics container includes the descriptors `vehicle.drivetrain.batteryManagement.header`, `vehicle.drivetrain.batteryManagement.maxEnergy`, `vehicle.powertrain.electric.battery.charging.power`, and `vehicle.drivetrain.electricEngine.charging.status`. Those fields let the integration reset the predicted state of charge and calculate the charging slope between stream updates. It seems like the `vehicle.drivetrain.batteryManagement.maxEnergy` always get sended even tho its not explicitly set, but check it anyways.

9. Save the selection.
10. Repeat for all the cars you want to support
11. In Home Assistant, install this integration via HACS (see below under Installation (HACS)) and still in Home Assistant, step trough the Configuration Flow also described here below.
12. During the Home Assistant config flow, paste the client ID, visit the provided verification URL, enter the code (if asked), and approve. **Do not click Continue/Submit in Home Assistant until the BMW page confirms the approval**; submitting early leaves the flow stuck and requires a restart.
13. *If you get Error 500 during setup:**
    
    **Immediate actions:**
    - ❌ Remove the integration from Home Assistant
    - 🔄 Go to BMW portal → Delete current Client ID
    - ⏱️ **Wait 5 minutes** (BMW backend needs to clear old session)
    - ✅ Create new Client ID
    - ⏱️ **Wait another 2 minutes**
    - ✅ Try installation again
    
    **If error persists after 2-3 attempts:**
    - ⏱️ Wait 24 hours (you may have hit daily rate limit)
    - Try during different time of day (BMW servers less loaded
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

## Entity Naming & Structure

- Each VIN becomes a device in HA (`VIN` pulled from CarData).
- Sensors/binary sensors are auto-created and named from descriptors (e.g. `Cabin Door Row1 Driver Is Open`).
- Additional attributes include the source timestamp.

## Debug Logging
Set `DEBUG_LOG = True` in `custom_components/cardata/const.py` for detailed MQTT/auth logs (disabled by default). To reduce noise, change it to `False` and reload HA.

## Predicted SOC with Learning

The integration includes a predicted SOC (State of Charge) sensor that estimates battery charge during charging sessions. This sensor uses real-time accumulated energy (trapezoidal integration of charging power minus auxiliary consumption) to calculate charging progress more frequently than BMW's native SOC updates. This handles varying power levels naturally (DC taper above 80%, cold-battery ramp-up, grid fluctuations).

### How Learning Works

The predicted SOC sensor automatically learns your vehicle's charging efficiency:

- **AC charging efficiency**: Starts at 90%, learns from actual charging sessions
- **DC charging efficiency**: Starts at 93%, learns from actual charging sessions
- Uses **Exponential Moving Average (EMA)** with 20% learning rate
- Separate learning for AC and DC charging
- Learning data persists across Home Assistant restarts
- Active charging sessions and pending sessions survive HA restarts (restored sessions skip learning to avoid polluted data from energy gaps)

### Learning Requirements

For learning to occur, a charging session must meet these criteria:
- Minimum 5% SOC gain during the session
- Calculated efficiency between 82% and 98% (outliers are rejected)
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
- **Stale header filtering during charging**: When `charging.level` is available and fresh, the stale `batteryManagement.header` value is skipped to avoid corrupting the predicted SOC display
- **BEVs**: For pure electric vehicles, the predicted SOC only syncs when not actively charging (standard behavior)

This ensures the predicted SOC stays accurate for PHEVs even when the hybrid system uses battery power in ways that don't register as "discharging" in the BMW API.

### Reset Buttons

Each EV/PHEV vehicle gets two button entities to reset learned efficiency:
- **Reset AC Learning**: Resets AC efficiency to default (90%)
- **Reset DC Learning**: Resets DC efficiency to default (93%)

These buttons appear in the vehicle's device page under Configuration entities.

## Developer Tools Services

Home Assistant's Developer Tools expose helper services for manual API checks:

- `cardata.fetch_telematic_data` fetches the current contents of the configured telematics container for a VIN and logs the raw payload.
- `cardata.fetch_vehicle_mappings` calls `GET /customers/vehicles/mappings` and logs the mapping details (including PRIMARY or SECONDARY status). Only primary mappings return data; some vehicles do not support secondary users, in which case the mapped user is considered the primary one.
- `cardata.fetch_basic_data` calls `GET /customers/vehicles/{vin}/basicData` to retrieve static metadata (model name, series, etc.) for the specified VIN.
- `migrations` call for proper renaming the sensors from old installations

## API Quota and MQTT Streaming

BMW imposes a **50 calls/day** limit on the CarData API. This integration does not enforce the limit client-side — BMW's own 429 response is respected via backoff. API usage is minimized through MQTT freshness gating and rate limiting:

- **MQTT Stream (real-time)**: The MQTT stream is unlimited and provides real-time updates for events like door locks, motion state, charging power, etc. GPS coordinates are paired using BMW payload timestamps (same GPS fix detection) with an arrival-time fallback, so location updates work even when latitude and longitude arrive in separate MQTT messages. Token refresh during MQTT reconnection is lock-free to avoid blocking the connection. After each token refresh, the MQTT connection is proactively reconnected with the fresh credentials to prevent BMW from dropping the session when the old token expires (~1 hour).
- **Trip-end polling**: When a vehicle stops moving (trip ends), the integration triggers an immediate API poll to capture post-trip battery state. This ensures SOC is updated even when the MQTT stream only delivers GPS/mileage but not SOC (common on some models). A per-VIN 10-minute cooldown prevents GPS burst flapping from burning API quota.
- **Charge-end polling**: When charging completes or stops, the integration triggers an immediate API poll to get the actual BMW SOC for learning calibration of the predicted SOC sensor, subject to the same per-VIN cooldown.
- **Fallback polling**: The integration polls every 12 hours as a fallback in case MQTT stream fails or after Home Assistant restarts. VINs with fresh MQTT data are skipped individually, so in multi-car setups only stale VINs consume API calls.
- **Multi-VIN setups**: All vehicles share the same 50 call/day limit.
- **Rate limiting**: If BMW returns a 429 (rate limited) response, the integration backs off automatically with exponential delay.

## Requirements

- BMW CarData account with streaming access (CarData API + CarData Streaming subscribed in the portal).
- Client ID created in the BMW portal (see "BMW Portal Setup").
- Home Assistant 2025.3+.
- TLS 1.3 capable SSL library: OpenSSL 1.1.1+, LibreSSL 3.2.0+, or equivalent (BMW's MQTT server requires TLS 1.3).
- Familiarity with BMW's CarData documentation: https://bmw-cardata.bmwgroup.com/customer/public/api-documentation/Id-Introduction

## !! Recommended setup with people on multiple bmw's (not required, its working as iss but you limiting yourself in accuracy since the harcode 50 limites a day !!
- Car 1 -> email_1
- Car 2 -> mail_2
- .....
- Use those seperate accounts in the the integration
- Use mail_x+1 with has all the cars merged for the bmmw app
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
| `__init__.py` | Entry point: setup/unload, token refresh loop, ghost device cleanup |
| `coordinator.py` | Central state management, message dispatch, entity signaling |
| `soc_wiring.py` | SOC/charging/driving prediction wiring between descriptors and prediction engines |
| `device_info.py` | Device metadata building, BEV detection, state restoration |
| `coordinator_housekeeping.py` | Diagnostics, stale VIN cleanup, old descriptor eviction, connection events |
| `soc_prediction.py` | Charging SOC: trapezoidal energy integration, EMA efficiency learning |
| `magic_soc.py` | Driving SOC: distance-based consumption prediction, adaptive EMA learning |
| `stream.py` | MQTT connection management, circuit breaker, credential hot-swap |
| `motion_detection.py` | GPS centroid movement detection, parking zone logic |
| `sensor.py` / `binary_sensor.py` / `device_tracker.py` | Home Assistant entity platforms |
| `config_flow.py` | Setup, reauthorization, and options UI flows |
| `bootstrap.py` | VIN discovery, metadata fetch, container creation |
| `auth.py` | Token refresh, reauth flow, stream error handling |
| `telematics.py` | Scheduled API polling, trip-end/charge-end triggers |
| `container.py` | Telematic container CRUD, signature-based reuse |

## Known Limitations

- Only one BMW stream per GCID: make sure no other clients are connected simultaneously.
- The CarData API is read-only; sending commands remains outside this integration.
- **Premature Continue in auth flow: If you hit Continue before authorizing on BMW’s site, the device-code flow gets stuck. Cancel the flow and restart the integration (or Home Assistant) once you’ve completed the BMW login.**

## Stale Issue Policy

Issues that remain inactive for 1 week receive an automated reminder. A second reminder follows after 2 weeks, and a final warning after 3 weeks. Issues with no response after 4 weeks are automatically closed. Any comment from a non-bot user resets the cycle. Issues labeled `pinned` or `security` are exempt.

## License

This project is licensed under the BSD 2-Clause License - see the [LICENSE](LICENSE.md) file for details.

### Attribution

This software was created by [Kris Van Biesen](https://github.com/kvanbiesen). Taken over since no response for original developper (https://github.com/JjyKsi/bmw-cardata-ha). Please keep this notice if you redistribute or modify the co
de.
