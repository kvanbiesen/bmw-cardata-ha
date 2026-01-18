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
4. Under section CARDATA API, you see **Client ID**. Copy this to your clipboard because you will need it during **Configuration Flow** in Home Assistant.
5. Now select Request access to CarData API and CarData Stream.
   Note, BMW portal seems to have some problems with scope selection. If you see an error on the top of the page, reload it, select one scope and wait for +30 seconds, then select the another one and wait agin.
6. Don't press the button Authenticate device!!!!
7. Scroll down to **CARDATA STREAMING** and press **Configure data stream** and on that new page, load all descriptors (keep clicking “Load more”).
8. Manually check every descriptor you want to stream or optionally to automate this, open the browser console (F12) and run:
```js
document.querySelectorAll('label.chakra-checkbox:not([data-checked])').forEach(l => l.click());
```
   - If you want the "Extrapolated SOC" helper sensor to work, make sure your telematics container includes the descriptors `vehicle.drivetrain.batteryManagement.header`, `vehicle.drivetrain.batteryManagement.maxEnergy`, `vehicle.powertrain.electric.battery.charging.power`, and `vehicle.drivetrain.electricEngine.charging.status`. Those fields let the integration reset the extrapolated state of charge and calculate the charging slope between stream updates. It seems like the `vehicle.drivetrain.batteryManagement.maxEnergy` always get sended even tho its not explicitly set, but check it anyways.

9. Save the selection.
10. Repeat for all the cars you want to support
11. In Home Assistant, install this integration via HACS (see below under Installation (HACS)) and still in Home Assistant, step trough the Configuration Flow also described here below.
12. During the Home Assistant config flow, paste the client ID, visit the provided verification URL, enter the code (if asked), and approve. **Do not click Continue/Submit in Home Assistant until the BMW page confirms the approval**; submitting early leaves the flow stuck and requires a restart.
13. If Step 12 Fails with error 500, remove the integration, go back to bmw page and create a new id (delete current and make new again) Wait couple of minutes and then try installing the integration again
14. Wait for the car to send data—triggering an action via the MyBMW app (lock/unlock doors) usually produces updates immediately. (older cars might need a drive before sensors start popping up, idrive6)

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

## Developer Tools Services

Home Assistant's Developer Tools expose helper services for manual API checks:

- `cardata.fetch_telematic_data` fetches the current contents of the configured telematics container for a VIN and logs the raw payload.
- `cardata.fetch_vehicle_mappings` calls `GET /customers/vehicles/mappings` and logs the mapping details (including PRIMARY or SECONDARY status). Only primary mappings return data; some vehicles do not support secondary users, in which case the mapped user is considered the primary one.
- `cardata.fetch_basic_data` calls `GET /customers/vehicles/{vin}/basicData` to retrieve static metadata (model name, series, etc.) for the specified VIN.
- `migrations` call for proper renaming the sensors from old installations

## Requirements

- BMW CarData account with streaming access (CarData API + CarData Streaming subscribed in the portal).
- Client ID created in the BMW portal (see "BMW Portal Setup").
- Home Assistant 2025.3+.
- Familiarity with BMW’s CarData documentation: https://bmw-cardata.bmwgroup.com/customer/public/api-documentation/Id-Introduction

## Known Limitations

- Only one BMW stream per GCID: make sure no other clients are connected simultaneously.
- The CarData API is read-only; sending commands remains outside this integration.
- **Premature Continue in auth flow: If you hit Continue before authorizing on BMW’s site, the device-code flow gets stuck. Cancel the flow and restart the integration (or Home Assistant) once you’ve completed the BMW login.**

## License

This project is licensed under the BSD 2-Clause License - see the [LICENSE](LICENSE.md) file for details.

### Attribution

This software was created by [Kris Van Biesen](https://github.com/kvanbiesen). Taken over since no response for original developper (https://github.com/JjyKsi/bmw-cardata-ha). Please keep this notice if you redistribute or modify the co
de.
