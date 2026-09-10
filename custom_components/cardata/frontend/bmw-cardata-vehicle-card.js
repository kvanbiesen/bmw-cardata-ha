/* BMW CarData Lovelace card
 * - Vehicle selector via backend websocket
 * - Visual style inspired by Vehicle Status Card layout
 * - Works for any selected vehicle (not tied to a specific model)
 * - Configurable sections
 */

const WS_TYPE = "cardata/vehicle_cards";
const CARD_TAG = "bmw-cardata-vehicle-card";
const CACHE_MS = 30_000;
const CARD_SIZE_UNIT_PX = 50;
const MAP_HEIGHT_DEFAULT = 120;

const ensureCustomCardsArray = () => {
  window.customCards = window.customCards || [];
  return window.customCards;
};

const boolConfig = (cfg, key, fallback) => {
  const raw = cfg?.[key];
  return typeof raw === "boolean" ? raw : fallback;
};

const normalizeState = (stateObj) => {
  const raw = stateObj?.state;
  if (raw === undefined || raw === null) return "";
  if (raw === "unknown" || raw === "unavailable") return "";
  return String(raw).trim().toLowerCase();
};

const formatState = (stateObj, hass) => {
  if (!stateObj) return "—";
  const state = stateObj.state;
  if (state === "unknown" || state === "unavailable") return "—";
  if (hass?.formatEntityState) return hass.formatEntityState(stateObj);
  const unit = stateObj.attributes?.unit_of_measurement;
  return unit ? `${state} ${unit}` : `${state}`;
};

const toNumberOrZero = (stateObj) => {
  const state = normalizeState(stateObj);
  if (!state) return 0;
  const parsed = Number(state);
  return Number.isFinite(parsed) ? parsed : 0;
};

const clamp = (value, min, max) => Math.min(max, Math.max(min, value));

const mapHeightConfig = (cfg) => {
  // Accept plain numbers as well as strings like "120px" - the GUI editor's
  // number selector only ever stores a bare number, but hand-written YAML
  // can easily include the "px" suffix shown in that field's UI label.
  const value = Number.parseFloat(cfg?.map_height);
  return Number.isFinite(value) && value > 0 ? value : MAP_HEIGHT_DEFAULT;
};

const isOpenState = (stateObj) => {
  const state = normalizeState(stateObj);
  if (!state) return false;
  return state.includes("open") || state === "on" || state === "true";
};

const isOnState = (stateObj) => {
  const state = normalizeState(stateObj);
  if (!state) return false;
  return state === "on" || state === "true" || state.includes("active");
};

const escapeHtml = (input) =>
  String(input ?? "")
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");

const iconBadge = (icon, statusClass = "", entityId = "", title = "") => `
  <button class="indicator ${statusClass}" data-entity-id="${escapeHtml(entityId)}" title="${escapeHtml(title)}">
    <ha-ripple></ha-ripple>
    <ha-icon icon="${icon}"></ha-icon>
  </button>
`;

// One quick-info tile built from the same elements as the native tile card.
// The background button is the tap target for the row, the icon keeps its own.
const tileItem = ({ icon, label, value, entity, cls = "" }) => {
  const entityAttr = escapeHtml(entity || "");
  return `
  <ha-card class="tile-item${cls ? ` ${cls}` : ""}">
    <div class="tile-item-bg" role="button" tabindex="0" data-entity-id="${entityAttr}" title="${entityAttr}"><ha-ripple></ha-ripple></div>
    <div class="tile-item-content">
      <ha-tile-icon interactive data-entity-id="${entityAttr}"><ha-icon slot="icon" icon="${escapeHtml(icon)}"></ha-icon></ha-tile-icon>
      <ha-tile-info primary="${escapeHtml(label)}" secondary="${escapeHtml(value)}"></ha-tile-info>
    </div>
  </ha-card>`;
};

const tileGrid = (items) => `<div class="tile-grid">${items.map(tileItem).join("")}</div>`;


const hasUsableState = (stateObj) => {
  const state = normalizeState(stateObj);
  return state !== "";
};

const compactStateLabel = (stateObj, t) => {
  const state = normalizeState(stateObj);
  if (!state) return "—";
  const key = `state.${state}`;
  const translated = t(key);
  if (translated !== key) return translated;
  return state.replaceAll("_", " ").toUpperCase();
};

const firstDefined = (...values) => values.find((value) => value !== undefined && value !== null && value !== "");

const sanitizePlate = (raw) => {
  if (typeof raw !== "string") return "";
  return raw.trim().replace(/[^\p{L}\p{N}\s-]/gu, "").substring(0, 15).toUpperCase();
};

const DEFAULT_LANG = "en";

const TRANSLATIONS = {
  en: {
    location: "Location",
    location_unavailable: "Location unavailable",
    away: "Away",
    home: "Home",
    range: "Range",
    fuel: "Fuel",
    motion: "Motion",
    moving: "Moving",
    parked: "Parked",
    charging: "Charging",
    level: "Level",
    tires: "Tires",
    tire: "Tire",
    mileage: "Mileage",
    lease_remaining: "Lease remaining",
    monthly_budget: "Remaining/month",
    monthly_average: "Average/month",
    km_balance: "km balance",
    driven: "Driven",
    target_today: "Target today",
    total_distance: "Total allowance",
    lease_start: "Lease start",
    lease_end: "Lease end",
    projected_at_end: "Projected at end",
    cost_refund: "Cost / refund",
    excess_km: "Excess mileage",
    shortfall_km: "Under mileage",
    payment_due: "Additional payment",
    refund: "Refund",
    unit_days_short: "d",
    unit_months_short: "mo",
    doors_overall: "Doors overall",
    lock: "Lock",
    locked: "Locked",
    unlocked: "Unlocked",
    alarm: "Alarm",
    alarm_triggered: "Alarm: TRIGGERED",
    alarm_unavailable: "Alarm status unavailable",
    windows: "Windows",
    windows_closed: "closed",
    windows_open_count: "open",
    charging_active: "active",
    lights: "Lights",
    lights_on: "on",
    lights_off: "off",
    hood_tailgate_open: "Hood and tailgate open",
    hood_open: "Hood open",
    tailgate_open: "Tailgate open",
    hood_tailgate_closed: "Hood and tailgate: closed",
    total_range: "Total Range",
    ev: "EV",
    unknown: "unknown",
    select_vehicle: "Select a vehicle in the card editor.",
    vehicle_not_found: "Vehicle not found yet. Try again in a few seconds.",
    no_tracker: "No vehicle tracker entity available",
    tracker_unavailable: "Tracker entity unavailable",
    map_loading: "Loading map…",
    map_failed: "Unable to load Home Assistant map",
    "state.chargingactive": "Charging Active",
    "state.chargingended": "Charging Ended",
    "state.nocharging": "No Charging",
    "state.chargingpaused": "Charging Paused",
    "state.chargingerror": "Charging Error",
    "state.doorstiltcabin": "Cabin Doors Tilted",
    "state.doorsonly": "Doors Only",
    "state.secured": "Secured",
    "state.open": "Open",
    "state.closed": "Closed",
    "state.locked": "Locked",
    "state.unlocked": "Unlocked",
    "editor.device_id": "Vehicle",
    "editor.license_plate": "License plate (shown instead of VIN)",
    "editor.show_title": "Show vehicle name / card header",
    "editor.soc_source": "Battery level source",
    "editor.soc_soc": "BMW State of Charge (last known)",
    "editor.soc_predicted": "Predicted SOC (charging)",
    "editor.soc_magic": "Magic SOC (driving)",
    "editor.show_indicators": "Show indicator row",
    "editor.show_range": "Show SOC and range bar",
    "editor.show_image": "Show vehicle image",
    "editor.image_crop_top": "Image crop top",
    "editor.image_crop_bottom": "Image crop bottom",
    "editor.image_zoom": "Image zoom",
    "editor.show_map": "Show mini map",
    "editor.map_height": "Mini map height",
    "editor.show_buttons": "Show quick info buttons",
    "editor.leasing_entity": "Leasing sensor (optional)",
    "editor.leasing_tiles": "Leasing tiles (order = display order)",
    "editor.language": "Card language",
    "editor.lang_auto": "Auto (Home Assistant language)",
  },
  de: {
    location: "Standort",
    location_unavailable: "Standort nicht verfügbar",
    away: "Unterwegs",
    home: "Zuhause",
    range: "Reichweite",
    fuel: "Kraftstoff",
    motion: "Bewegung",
    moving: "In Fahrt",
    parked: "Geparkt",
    charging: "Laden",
    level: "Füllstand",
    tires: "Reifen",
    tire: "Reifen",
    mileage: "Kilometerstand",
    lease_remaining: "Leasing-Restlaufzeit",
    monthly_budget: "Restsaldo/Monat",
    monthly_average: "Schnitt/Monat",
    km_balance: "km-Saldo",
    driven: "Gefahren",
    target_today: "Soll heute",
    total_distance: "Gesamtkilometer",
    lease_start: "Leasingbeginn",
    lease_end: "Leasingende",
    projected_at_end: "Prognose Vertragsende",
    cost_refund: "Kosten / Erstattung",
    excess_km: "Mehrkilometer",
    shortfall_km: "Minderkilometer",
    payment_due: "Nachzahlung",
    refund: "Erstattung",
    unit_days_short: "T",
    unit_months_short: "Mon.",
    doors_overall: "Türen gesamt",
    lock: "Schloss",
    locked: "Verriegelt",
    unlocked: "Entriegelt",
    alarm: "Alarm",
    alarm_triggered: "Alarm: AUSGELÖST",
    alarm_unavailable: "Alarmstatus nicht verfügbar",
    windows: "Fenster",
    windows_closed: "geschlossen",
    windows_open_count: "offen",
    charging_active: "aktiv",
    lights: "Licht",
    lights_on: "an",
    lights_off: "aus",
    hood_tailgate_open: "Motorhaube und Heckklappe offen",
    hood_open: "Motorhaube offen",
    tailgate_open: "Heckklappe offen",
    hood_tailgate_closed: "Motorhaube und Heckklappe: geschlossen",
    total_range: "Gesamtreichweite",
    ev: "Elektro",
    unknown: "unbekannt",
    select_vehicle: "Wähle ein Fahrzeug im Karten-Editor.",
    vehicle_not_found: "Fahrzeug noch nicht gefunden. Versuch es in ein paar Sekunden erneut.",
    no_tracker: "Kein Fahrzeug-Tracker verfügbar",
    tracker_unavailable: "Tracker-Entität nicht verfügbar",
    map_loading: "Karte lädt…",
    map_failed: "Home-Assistant-Karte konnte nicht geladen werden",
    "state.chargingactive": "Lädt",
    "state.chargingended": "Laden beendet",
    "state.nocharging": "Lädt nicht",
    "state.chargingpaused": "Laden pausiert",
    "state.chargingerror": "Ladefehler",
    "state.doorstiltcabin": "Türen gekippt",
    "state.doorsonly": "Nur Türen",
    "state.secured": "Gesichert",
    "state.open": "Offen",
    "state.closed": "Geschlossen",
    "state.locked": "Verriegelt",
    "state.unlocked": "Entriegelt",
    "editor.device_id": "Fahrzeug",
    "editor.license_plate": "Kennzeichen (statt VIN angezeigt)",
    "editor.show_title": "Fahrzeugname / Kartentitel anzeigen",
    "editor.soc_source": "Quelle für Batteriestand",
    "editor.soc_soc": "BMW-Ladestand (zuletzt bekannt)",
    "editor.soc_predicted": "Prognostizierter Ladestand (beim Laden)",
    "editor.soc_magic": "Magic SOC (während der Fahrt)",
    "editor.show_indicators": "Statuszeile anzeigen",
    "editor.show_range": "Ladestand- und Reichweitenbalken anzeigen",
    "editor.show_image": "Fahrzeugbild anzeigen",
    "editor.image_crop_top": "Bild oben zuschneiden",
    "editor.image_crop_bottom": "Bild unten zuschneiden",
    "editor.image_zoom": "Bildzoom",
    "editor.show_map": "Mini-Karte anzeigen",
    "editor.map_height": "Höhe der Mini-Karte",
    "editor.show_buttons": "Schnellinfo-Kacheln anzeigen",
    "editor.leasing_entity": "Leasing-Sensor (optional)",
    "editor.leasing_tiles": "Leasing-Kacheln (Reihenfolge = Anzeige)",
    "editor.language": "Kartensprache",
    "editor.lang_auto": "Automatisch (HA-Sprache)",
  },
};

// value = tile key in the render registry, label = translation key for the editor
const LEASE_TILE_OPTIONS = [
  { value: "lease_remaining", labelKey: "lease_remaining" },
  { value: "monthly_budget", labelKey: "monthly_budget" },
  { value: "monthly_average", labelKey: "monthly_average" },
  { value: "km_balance", labelKey: "km_balance" },
  { value: "driven", labelKey: "driven" },
  { value: "target", labelKey: "target_today" },
  { value: "total", labelKey: "total_distance" },
  { value: "lease_start", labelKey: "lease_start" },
  { value: "lease_end", labelKey: "lease_end" },
  { value: "projected", labelKey: "projected_at_end" },
  { value: "cost", labelKey: "cost_refund" },
];
const DEFAULT_LEASE_TILES = ["lease_remaining", "monthly_budget", "projected", "cost"];

const resolveLang = (cfg, hass) => {
  const configured = typeof cfg?.language === "string" && cfg.language !== "auto" ? cfg.language : "";
  const raw = (configured || hass?.locale?.language || hass?.language || DEFAULT_LANG).toLowerCase();
  const short = raw.split("-")[0];
  return TRANSLATIONS[raw] ? raw : TRANSLATIONS[short] ? short : DEFAULT_LANG;
};

const localize = (lang, key) =>
  TRANSLATIONS[lang]?.[key] ?? TRANSLATIONS[DEFAULT_LANG]?.[key] ?? key;

const humanizeLocationState = (rawState, t) => {
  if (rawState === undefined || rawState === null) return t("location_unavailable");
  const normalized = String(rawState).trim().toLowerCase();
  if (!normalized || normalized === "unknown" || normalized === "unavailable") return t("location_unavailable");
  if (normalized === "not_home") return t("away");
  if (normalized === "home") return t("home");
  return String(rawState).replaceAll("_", " ");
};

const humanizeStateValue = (rawState, t) => {
  if (rawState === undefined || rawState === null) return "—";
  const normalized = String(rawState).trim().toLowerCase();
  if (!normalized || normalized === "unknown" || normalized === "unavailable") return "—";
  const key = `state.${normalized}`;
  const translated = t(key);
  if (translated !== key) return translated;

  return normalized
    .replaceAll("_", " ")
    .split(" ")
    .filter(Boolean)
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
};

const attrNumber = (stateObj, key) => {
  const value = Number(stateObj?.attributes?.[key]);
  return Number.isFinite(value) ? value : NaN;
};

const formatSignedDistance = (value, unit) => {
  if (!Number.isFinite(value)) return "—";
  const rounded = Math.round(value);
  return `${rounded > 0 ? "+" : ""}${rounded.toLocaleString()} ${unit || "km"}`;
};

const formatAbsDistance = (value, unit) => {
  if (!Number.isFinite(value)) return "—";
  return `${Math.abs(Math.round(value)).toLocaleString()} ${unit || "km"}`;
};

const formatLeaseDate = (value, lang) => {
  if (!value) return "—";
  const parsed = new Date(value);
  return Number.isNaN(parsed.getTime()) ? String(value) : parsed.toLocaleDateString(lang);
};

// positive = extra distance / extra cost (alert), negative = under distance / refund (good)
const leaseDeltaClass = (value) =>
  !Number.isFinite(value) || Math.round(value) === 0 ? "" : value > 0 ? "alert" : "good";

const formatLeaseRemaining = (days, months, t) => {
  if (Number.isFinite(days) && days < 61) return `${Math.max(0, Math.round(days))} ${t("unit_days_short")}`;
  if (Number.isFinite(months)) return `${months} ${t("unit_months_short")}`;
  return "—";
};

const formatLeaseCost = (value, currency) => {
  if (!Number.isFinite(value)) return "—";
  try {
    return new Intl.NumberFormat(undefined, { style: "currency", currency: currency || "EUR" }).format(value);
  } catch {
    return `${value.toFixed(2)} ${currency || "EUR"}`;
  }
};

class BmwCardataVehicleCard extends HTMLElement {
  setConfig(config) {
    const cfg = config || {};
    this._config = cfg.license_plate
      ? { ...cfg, license_plate: sanitizePlate(cfg.license_plate) }
      : cfg;
    this._initialized = false;
    this._vehicles = null;
    this._vehiclesFetchedAt = 0;
    this._fetchInFlight = null;
  }

  getCardSize() {
    const cfg = this._config || {};
    let size = 5;
    if (boolConfig(cfg, "show_image", true)) size += 2;
    if (boolConfig(cfg, "show_map", true)) size += mapHeightConfig(cfg) / CARD_SIZE_UNIT_PX;
    if (boolConfig(cfg, "show_buttons", true)) size += 2;
    if (cfg.leasing_entity) {
      const tileCount = Array.isArray(cfg.leasing_tiles) && cfg.leasing_tiles.length
        ? cfg.leasing_tiles.length
        : DEFAULT_LEASE_TILES.length;
      size += Math.ceil(tileCount / 2) / 2;
    }
    return size;
  }

  static getConfigForm() {
    // Static context has no hass reference; read the UI language from the
    // home-assistant root element (common custom-card pattern), fall back to en.
    const lang = resolveLang({}, document.querySelector("home-assistant")?.hass);
    const t = (key) => localize(lang, key);
    return {
      schema: [
        {
          name: "device_id",
          required: true,
          selector: {
            device: { integration: "cardata" },
          },
        },
        { name: "license_plate", selector: { text: {} } },
        { name: "show_title", selector: { boolean: {} } },
        {
          name: "soc_source",
          selector: {
            select: {
              options: [
                { value: "soc", label: t("editor.soc_soc") },
                { value: "predicted", label: t("editor.soc_predicted") },
                { value: "magic", label: t("editor.soc_magic") },
              ],
              mode: "dropdown",
            },
          },
        },
        { name: "show_indicators", selector: { boolean: {} } },
        { name: "show_range", selector: { boolean: {} } },
        { name: "show_image", selector: { boolean: {} } },
        {
          name: "image_crop_top",
          selector: {
            number: {
              mode: "box",
              min: 0,
              max: 40,
              step: 1,
              unit_of_measurement: "%",
                    },
          },
        },
        {
          name: "image_crop_bottom",
          selector: {
            number: {
              mode: "box",
              min: 0,
              max: 40,
              step: 1,
              unit_of_measurement: "%",
            },
          },
        },
        {
          name: "image_zoom",
          selector: {
            number: {
              mode: "box",
              min: 50,
              max: 200,
              step: 5,
              unit_of_measurement: "%",
            },
          },
        },		
        { name: "show_map", selector: { boolean: {} } },
        {
          name: "map_height",
          selector: {
            number: {
              mode: "box",
              unit_of_measurement: "px",
            },
          },
        },
        { name: "show_buttons", selector: { boolean: {} } },
        {
          name: "leasing_entity",
          selector: { entity: { domain: "sensor" } },
        },
        {
          name: "leasing_tiles",
          selector: {
            select: {
              multiple: true,
              mode: "dropdown",
              options: LEASE_TILE_OPTIONS.map((opt) => ({ value: opt.value, label: t(opt.labelKey) })),
            },
          },
        },
        {
          name: "language",
          selector: {
            select: {
              options: [
                { value: "auto", label: t("editor.lang_auto") },
                { value: "en", label: "English" },
                { value: "de", label: "Deutsch" },
              ],
              mode: "dropdown",
            },
          },
        },
      ],
      computeLabel: (schema) => {
        const key = `editor.${schema.name}`;
        const label = t(key);
        return label === key ? undefined : label;
      },
    };
  }

  static getStubConfig() {
    return {
      soc_source: "soc",
      show_title: true,
      show_indicators: true,
      show_range: true,
      show_image: true,
      show_map: true,
      map_height: MAP_HEIGHT_DEFAULT,
      show_buttons: true,
    };
  }

  set hass(hass) {
    this._hass = hass;
    if (!this._config) return;

    if (!this._initialized) {
      this._initialized = true;
      this.attachShadow({ mode: "open" });
      this.shadowRoot.innerHTML = `
        <style>
          :host { display: block; }

          /* Sizes and radii mirror the native tile card and its features:
           * 56px tile rows, 42px feature controls, 36px icon circles, 20% tint. */
          .vin {
            color: var(--secondary-text-color);
            font-size: var(--ha-font-size-s, 12px);
            font-weight: var(--ha-font-weight-normal, 400);
            line-height: var(--ha-line-height-condensed, 1.2);
            letter-spacing: 0.4px;
          }
          .vin:empty { display: none; }
          #main-wrapper {
            display: grid;
            gap: var(--ha-space-3, 12px);
          }
          .vin:not(:empty) + #main-wrapper {
            margin-top: var(--ha-space-3, 12px);
          }
          #main-wrapper > :empty { display: none; }

          .indicators {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(60px, 1fr));
            gap: var(--ha-space-3, 12px);
          }
          .indicator {
            --tile-color: var(--state-inactive-color);
            --ha-ripple-color: var(--tile-color);
            --mdc-icon-size: 20px;
            appearance: none;
            position: relative;
            overflow: hidden;
            cursor: pointer;
            display: flex;
            align-items: center;
            justify-content: center;
            width: 100%;
            height: 42px;
            margin: 0;
            padding: 0;
            border: 0;
            border-radius: var(--ha-border-radius-lg, 12px);
            background: none;
            color: var(--tile-color);
            -webkit-tap-highlight-color: transparent;
            transition: color 180ms ease-in-out;
          }
          .indicator::before {
            content: "";
            position: absolute;
            inset: 0;
            background-color: var(--tile-color);
            opacity: 0.2;
            transition: background-color 180ms ease-in-out, opacity 180ms ease-in-out;
            pointer-events: none;
          }
          .indicator ha-icon {
            position: relative;
          }
          .indicator:focus { outline: none; }
          .indicator:focus-visible { box-shadow: 0 0 0 2px var(--tile-color); }
          .indicator.ok { --tile-color: var(--state-icon-color); }
          .indicator.alert { --tile-color: var(--error-color); }
          .indicator.good { --tile-color: var(--success-color); }
          .indicator.charging {
            animation: chargingBadgePulse 1.4s ease-in-out infinite;
          }

          .range-box {
            display: grid;
            gap: var(--ha-space-2, 8px);
          }
          .range-top {
            display: flex;
            align-items: center;
            gap: var(--ha-space-3, 12px);
          }
          .bar-wrap,
          .bar-wrap-unified {
            --tile-color: var(--state-icon-color);
            position: relative;
            flex: 1 1 auto;
            height: 42px;
            border-radius: var(--ha-border-radius-lg, 12px);
            overflow: hidden;
            cursor: pointer;
          }
          .bar-wrap-unified {
            display: flex;
          }
          .bar-wrap::before,
          .bar-wrap-unified::before {
            content: "";
            position: absolute;
            inset: 0;
            background-color: var(--tile-color);
            opacity: 0.2;
          }
          .bar-level {
            position: relative;
            height: 100%;
            background: var(--tile-color);
            transition: width 180ms ease-in-out;
            overflow: hidden;
          }
          .bar-wrap.charging .bar-level {
            animation: chargingBarPulse 1.8s ease-in-out infinite;
          }
          .bar-wrap.charging .bar-level::after,
          .bar-wrap-unified.charging .bar-segment-unified.ev::after {
            content: "";
            position: absolute;
            inset: 0;
            background: linear-gradient(
              110deg,
              transparent 10%,
              rgba(255, 255, 255, 0.35) 45%,
              transparent 80%
            );
            transform: translateX(-120%);
            animation: chargingSweep 2.3s linear infinite;
            pointer-events: none;
          }
          .energy-text {
            position: absolute;
            left: var(--ha-space-3, 12px);
            top: 50%;
            transform: translateY(-50%);
            color: var(--text-primary-color, #fff);
            font-size: var(--ha-font-size-m, 14px);
            font-weight: var(--ha-font-weight-medium, 500);
            text-shadow: 0 1px 2px rgb(0 0 0 / 35%);
          }
          .range-value {
            display: flex;
            align-items: center;
            gap: var(--ha-space-2, 8px);
            color: var(--primary-text-color);
            font-size: var(--ha-font-size-m, 14px);
            font-weight: var(--ha-font-weight-medium, 500);
            white-space: nowrap;
            cursor: pointer;
          }

          /* PHEV unified bar */
          .bar-wrap-unified.charging {
            animation: chargingBarPulse 1.8s ease-in-out infinite;
          }
          .bar-segment-unified {
            position: relative;
            height: 100%;
            overflow: hidden;
            transition: width 180ms ease-in-out;
          }
          .bar-segment-unified.ev {
            background: var(--success-color, #4caf50);
          }
          .bar-segment-unified.fuel {
            background: var(--warning-color, #ff9800);
          }
          .range-split-labels {
            display: flex;
            justify-content: space-between;
            gap: var(--ha-space-3, 12px);
          }
          .range-split-label {
            --mdc-icon-size: 18px;
            display: flex;
            align-items: center;
            gap: var(--ha-space-2, 8px);
            flex: 1;
            cursor: pointer;
            font-size: var(--ha-font-size-s, 12px);
          }
          .range-split-label.ev {
            color: var(--success-color, #4caf50);
          }
          .range-split-label.fuel {
            color: var(--warning-color, #ff9800);
          }

          .image {
            width: 100%;
            border-radius: var(--ha-border-radius-lg, 12px);
            overflow: hidden;
            cursor: pointer;
          }
          .image img {
            width: 100%;
            display: block;
            object-fit: cover;
            object-position: center;
            transform-origin: center center;
            transform: scale(var(--image-zoom, 1));
            margin-top: calc(-1 * var(--image-crop-top, 0%));
            margin-bottom: calc(-1 * var(--image-crop-bottom, 0%));
          }
          .image.charging img {
            animation: chargingImagePulse 2.2s ease-in-out infinite;
          }

          .map {
            border-radius: var(--ha-border-radius-lg, 12px);
            overflow: hidden;
            background: var(--secondary-background-color);
          }
          .map hui-map-card {
            --ha-card-border-width: 0;
            display: block;
            width: 100%;
            height: var(--map-height, ${MAP_HEIGHT_DEFAULT}px);
          }
          .map-mount {
            height: var(--map-height, ${MAP_HEIGHT_DEFAULT}px);
            overflow: hidden;
          }
          .map-mount > * {
            height: 100%;
          }
          .map-fallback {
            height: var(--map-height, ${MAP_HEIGHT_DEFAULT}px);
            display: flex;
            align-items: center;
            justify-content: center;
            color: var(--secondary-text-color);
            font-size: var(--ha-font-size-s, 12px);
          }

          .tile-grid {
            display: grid;
            grid-template-columns: repeat(2, minmax(0, 1fr));
            gap: var(--ha-space-2, 8px);
          }
          .tile-item {
            --tile-color: var(--state-icon-color);
            --ha-ripple-color: var(--tile-color);
            --ha-ripple-hover-opacity: 0.04;
            --ha-ripple-pressed-opacity: 0.12;
            -webkit-tap-highlight-color: transparent;
            transition: box-shadow 180ms ease-in-out, border-color 180ms ease-in-out;
          }
          .tile-item.alert { --tile-color: var(--error-color); }
          .tile-item.good { --tile-color: var(--success-color); }
          .tile-item:has(.tile-item-bg:focus-visible) {
            border-color: var(--tile-color);
            box-shadow: var(--ha-card-box-shadow, 0 0 0 0 transparent), 0 0 0 1px var(--tile-color);
          }
          /* Both layers extend under the card border, like the native tile container. */
          .tile-item-bg {
            position: absolute;
            inset: 0;
            margin: calc(-1 * var(--ha-card-border-width, 1px));
            border-radius: inherit;
            overflow: hidden;
            cursor: pointer;
          }
          .tile-item-bg:focus { outline: none; }
          .tile-item-content {
            position: relative;
            display: flex;
            align-items: center;
            gap: 10px;
            padding: 0 10px;
            margin: calc(-1 * var(--ha-card-border-width, 1px));
            min-height: 56px;
            min-width: 0;
            box-sizing: border-box;
            pointer-events: none;
          }
          .tile-item-content ha-tile-icon {
            --tile-icon-color: var(--tile-color);
            position: relative;
            padding: 6px;
            margin: -6px;
            flex: none;
            pointer-events: auto;
          }
          .tile-item-content ha-tile-info {
            min-width: 0;
          }

          @media (max-width: 520px) {
            .tile-grid {
              grid-template-columns: 1fr;
            }
            .indicators {
              grid-template-columns: repeat(3, minmax(0, 1fr));
            }
          }

          @keyframes chargingBadgePulse {
            0% { filter: brightness(1); }
            50% { filter: brightness(1.45); }
            100% { filter: brightness(1); }
          }

          @keyframes chargingBarPulse {
            0% { filter: brightness(1); }
            50% { filter: brightness(1.18); }
            100% { filter: brightness(1); }
          }

          @keyframes chargingSweep {
            0% { transform: translateX(-120%); }
            100% { transform: translateX(120%); }
          }

          @keyframes chargingImagePulse {
            0% { filter: brightness(1) saturate(1); }
            50% { filter: brightness(1.08) saturate(1.1); }
            100% { filter: brightness(1) saturate(1); }
          }
        </style>
        <ha-card>
          <h1 class="card-header" id="name"></h1>
          <div class="card-content">
            <div class="vin" id="vin"></div>
            <main id="main-wrapper">
              <div id="indicators"></div>
              <div id="images"></div>
              <div id="range_info"></div>
              <div id="mini_map"></div>
              <div id="buttons"></div>
              <div id="leasing"></div>
            </main>
          </div>
        </ha-card>
      `;
      this._nameEl = this.shadowRoot.getElementById("name");
      this._bindInteractions();
    }

    this._maybeFetchVehicles();
    this._render();
  }

  _maybeFetchVehicles() {
    const hass = this._hass;
    if (!hass || typeof hass.callWS !== "function") return;

    const now = Date.now();
    if (this._vehicles && now - this._vehiclesFetchedAt < CACHE_MS) return;
    if (this._fetchInFlight) return;

    this._fetchInFlight = hass
      .callWS({ type: WS_TYPE })
      .then((payload) => {
        this._vehicles = Array.isArray(payload?.vehicles) ? payload.vehicles : [];
        this._vehiclesFetchedAt = Date.now();
      })
      .catch(() => {
        // Keep placeholders; card remains functional.
      })
      .finally(() => {
        this._fetchInFlight = null;
        this._render();
      });
  }

  _bindInteractions() {
    if (!this.shadowRoot || this._interactionsBound) return;
    this._interactionsBound = true;
    const entityIdFrom = (node) => {
      if (!(node instanceof Element)) return "";
      const target = node.closest("[data-entity-id]");
      return target ? target.getAttribute("data-entity-id") || "" : "";
    };
    const insideTileIcon = (node) => node instanceof Element && Boolean(node.closest("ha-tile-icon"));
    // ha-tile-icon runs Home Assistant's action handler, which cancels the
    // synthesized click on touch devices, so its taps arrive as "action" events.
    this.shadowRoot.addEventListener("click", (event) => {
      if (insideTileIcon(event.target)) return;
      this._openMoreInfo(entityIdFrom(event.target));
    });
    this.shadowRoot.addEventListener("action", (event) => {
      if (event.detail?.action !== "tap" || !insideTileIcon(event.target)) return;
      this._openMoreInfo(entityIdFrom(event.target));
    });
    this.shadowRoot.addEventListener("keydown", (event) => {
      if (event.key !== "Enter" && event.key !== " ") return;
      const node = event.target;
      if (!(node instanceof Element) || node.getAttribute("role") !== "button") return;
      const entityId = entityIdFrom(node);
      if (!entityId) return;
      event.preventDefault();
      this._openMoreInfo(entityId);
    });
  }

  // The header must leave the DOM when hidden: ha-card spaces .card-content
  // by sibling position, so a hidden header would still pull the content up.
  _setTitleVisible(visible) {
    const nameEl = this._nameEl;
    if (!nameEl) return;
    if (visible) {
      if (!nameEl.isConnected) this.shadowRoot.querySelector("ha-card").prepend(nameEl);
    } else if (nameEl.isConnected) {
      nameEl.remove();
    }
  }

  _openMoreInfo(entityId) {
    if (!entityId) return;
    this.dispatchEvent(
      new CustomEvent("hass-more-info", {
        bubbles: true,
        composed: true,
        detail: { entityId },
      })
    );
  }

  async _createMapCard(hass, trackerEntityId) {
    try {
      if (!window.loadCardHelpers) return null;
      const helpers = await window.loadCardHelpers();
      if (!helpers?.createCardElement) return null;
      // Picture first (default label_mode), fall back to icon mode only when
      // there's no entity_picture to show - otherwise label_mode:"icon" would
      // suppress a picture that's actually there.
      const hasPicture = !!hass?.states?.[trackerEntityId]?.attributes?.entity_picture;
      const mapCard = helpers.createCardElement({
        type: "map",
        entities: hasPicture ? [trackerEntityId] : [{ entity: trackerEntityId, label_mode: "icon" }],
        default_zoom: 14,
        hours_to_show: 24,
      });
      mapCard.layout = "grid";
      mapCard.hass = hass;
      return mapCard;
    } catch {
      return null;
    }
  }

  _renderMap(target, hass, trackerEntityId, t) {
    if (!target) return;
    // This method writes the DOM directly, so the _setHtml cache is stale.
    target._lastHtml = undefined;

    if (!trackerEntityId) {
      this._cachedMapCard = null;
      this._cachedMapTracker = null;
      target.innerHTML = `
        <div class="map">
          <div class="map-fallback">${escapeHtml(t("no_tracker"))}</div>
        </div>
      `;
      return;
    }

    if (!hass?.states?.[trackerEntityId]) {
      this._cachedMapCard = null;
      this._cachedMapTracker = null;
      target.innerHTML = `
        <div class="map">
          <div class="map-fallback">${escapeHtml(t("tracker_unavailable"))}: ${escapeHtml(trackerEntityId)}</div>
        </div>
      `;
      return;
    }

    // Reuse existing map card — just update hass.
    if (this._cachedMapCard && this._cachedMapTracker === trackerEntityId) {
      this._cachedMapCard.hass = hass;
      return;
    }

    this._cachedMapCard = null;
    this._cachedMapTracker = null;

    const renderToken = (this._mapRenderToken || 0) + 1;
    this._mapRenderToken = renderToken;

    const wrapper = document.createElement("div");
    wrapper.className = "map";

    const mapMount = document.createElement("div");
    mapMount.className = "map-mount";
    mapMount.innerHTML = `<div class="map-fallback">${escapeHtml(t("map_loading"))}</div>`;

    wrapper.appendChild(mapMount);
    target.replaceChildren(wrapper);

    this._createMapCard(hass, trackerEntityId).then((mapCard) => {
      if (!target.isConnected) return;
      if (this._mapRenderToken !== renderToken) return;
      if (!mapCard) {
        mapMount.innerHTML = `<div class="map-fallback">${escapeHtml(t("map_failed"))}</div>`;
        return;
      }
      this._cachedMapCard = mapCard;
      this._cachedMapTracker = trackerEntityId;
      mapMount.replaceChildren(mapCard);
    });
  }

  // Compare against the string written last time, not against the live DOM.
  // The browser serializes attributes differently from the template (for
  // example `interactive` comes back as `interactive=""`), and a false
  // mismatch would rebuild the section on every state update, which drops
  // hover state and swallows clicks in progress.
  _setHtml(el, html) {
    if (!el || el._lastHtml === html) return;
    el._lastHtml = html;
    el.innerHTML = html;
  }

  _render() {
    if (!this.shadowRoot) return;

    const hass = this._hass;
    const cfg = this._config || {};
    const deviceId = cfg.device_id;
    const lang = resolveLang(cfg, hass);
    const t = (key) => localize(lang, key);

    if (!deviceId) {
      this._renderMessage(t("select_vehicle"));
      return;
    }

    const vehicles = this._vehicles || [];
    const vehicle = vehicles.find((v) => v && v.device_id === deviceId);
    if (!vehicle) {
      this._renderMessage(t("vehicle_not_found"));
      return;
    }

    const nameEl = this._nameEl;
    const vinEl = this.shadowRoot.getElementById("vin");
    const indicatorsEl = this.shadowRoot.getElementById("indicators");
    const rangeEl = this.shadowRoot.getElementById("range_info");
    const imageEl = this.shadowRoot.getElementById("images");
    const mapEl = this.shadowRoot.getElementById("mini_map");
    const buttonsEl = this.shadowRoot.getElementById("buttons");

    const vin = vehicle.vin || "";
    const name = vehicle.name || vin || "BMW CarData";
    const entities = vehicle.entities || {};

    const read = (key) => hass?.states?.[entities[key]];

    const showTitle = boolConfig(cfg, "show_title", true);
    this._setTitleVisible(showTitle);
    nameEl.textContent = showTitle ? name : "";
    vinEl.textContent = cfg.license_plate || vin;

    const showIndicators = boolConfig(cfg, "show_indicators", true);
    const showRange = boolConfig(cfg, "show_range", true);
    const showImage = boolConfig(cfg, "show_image", true);
    const showMap = boolConfig(cfg, "show_map", true);
    const showButtons = boolConfig(cfg, "show_buttons", true);
    const mapEntityId = entities.device_tracker;

    const imageCropTop = Number.isFinite(Number(cfg.image_crop_top))
      ? Number(cfg.image_crop_top)
      : 0;

    const imageCropBottom = Number.isFinite(Number(cfg.image_crop_bottom))
      ? Number(cfg.image_crop_bottom)
      : 0;

    const imageZoom = Number.isFinite(Number(cfg.image_zoom))
      ? Number(cfg.image_zoom)
      : 100;

    imageEl.style.setProperty("--image-crop-top", `${imageCropTop}%`);
    imageEl.style.setProperty("--image-crop-bottom", `${imageCropBottom}%`);
    imageEl.style.setProperty("--image-zoom", imageZoom / 100);

    mapEl.style.setProperty("--map-height", `${mapHeightConfig(cfg)}px`);

    const lockState = normalizeState(read("doors_lock"));
    const doorsOverallStateObj = read("doors_overall");
    const doorsOverallState = compactStateLabel(doorsOverallStateObj, t);
    const doorsOverallRaw = normalizeState(doorsOverallStateObj);
    const alarmActiveStateObj = read("alarm_active");
    const alarmArmingStateObj = read("alarm_arming");
    const alarmActiveState = normalizeState(alarmActiveStateObj);
    const alarmArmingLabel = compactStateLabel(alarmArmingStateObj, t);
    const chargingState = normalizeState(read("charging_state"));
    const lockEntity = entities.doors_lock || "";
    const doorsOverallEntity = entities.doors_overall || "";
    const motionEntity = entities.motion_state || "";
    const motionStateObj = read("motion_state");
    const alarmActiveEntity = entities.alarm_active || "";
    const alarmArmingEntity = entities.alarm_arming || "";
    const chargingEntity = entities.charging_state || "";
    const windowEntity =
      entities.window_front_driver ||
      entities.window_front_passenger ||
      entities.window_rear_driver ||
      entities.window_rear_passenger ||
      "";
    const lightsEntity = entities.lights || "";
    const hoodEntity = entities.hood || "";
    const tailgateEntity = entities.tailgate || "";
    const socSourceKey = cfg.soc_source === "predicted" ? "soc_predicted" : cfg.soc_source === "magic" ? "soc_magic" : "soc";
    const socEntity = entities[socSourceKey] || entities.soc || "";
    const socState = socEntity ? hass?.states?.[socEntity] : undefined;
    const fuelLevelEntity = entities.fuel_level || "";
    const rangeEntity = entities.range_total || "";
    const remainingFuelEntity = entities.remaining_fuel || "";
    const rangeElectricEntity = entities.range_electric || "";
    const rangeFuelEntity = entities.range_fuel || "";

    const openWindows = [
      read("window_front_driver"),
      read("window_front_passenger"),
      read("window_rear_driver"),
      read("window_rear_passenger"),
    ].filter((stateObj) => isOpenState(stateObj)).length;

    const hoodOpen = isOpenState(read("hood"));
    const tailgateOpen = isOpenState(read("tailgate"));
    const lightsOn = isOnState(read("lights"));
    const hasAlarm =
      Boolean(alarmActiveEntity && hasUsableState(alarmActiveStateObj)) ||
      Boolean(alarmArmingEntity && hasUsableState(alarmArmingStateObj));
    const alarmIsActive = alarmActiveState === "on" || alarmActiveState === "true";
    const alarmArmingState = normalizeState(alarmArmingStateObj);
    const alarmIsArmed = alarmArmingState !== "" && alarmArmingState !== "unarmed";
    const motionState = normalizeState(motionStateObj);
    const motionKnown = motionState !== "";
    const isMoving = motionState === "on" || motionState === "true" || motionState.includes("moving");
    const hasCharging = Boolean(chargingEntity && hasUsableState(read("charging_state")));
    const hasFuelLevel = Boolean(fuelLevelEntity && hasUsableState(read("fuel_level")));
    const hasSoc = Boolean(socEntity && hasUsableState(socState));
    const hasFuelRemaining = Boolean(remainingFuelEntity && hasUsableState(read("remaining_fuel")));
    const hasRange = Boolean(rangeEntity && hasUsableState(read("range_total")));
    const hasRangeElectric = Boolean(rangeElectricEntity && hasUsableState(read("range_electric")));
    const hasRangeFuel = Boolean(rangeFuelEntity && hasUsableState(read("range_fuel")));
    const isPHEV = hasRangeElectric && hasRangeFuel;
    const tankCapValue = toNumberOrZero(read("manual_tank_capacity"));
    const fuelLitres = toNumberOrZero(read("remaining_fuel"));
    const hasFuelWithCap = hasFuelRemaining && tankCapValue > 0;
    const primaryLevelState = hasSoc ? socState : hasFuelLevel ? read("fuel_level") : (hasFuelWithCap || hasFuelRemaining) ? read("remaining_fuel") : null;
    const primaryLevelEntity = hasSoc ? socEntity : hasFuelLevel ? fuelLevelEntity : hasFuelRemaining ? remainingFuelEntity : "";
    const primaryLevelValue = hasSoc ? clamp(Math.round(toNumberOrZero(socState)), 0, 100) : hasFuelLevel ? clamp(Math.round(toNumberOrZero(read("fuel_level"))), 0, 100) : hasFuelWithCap ? clamp(Math.round(fuelLitres / tankCapValue * 100), 0, 100) : 0;
    const primaryLevelLabel = hasSoc ? `${primaryLevelValue}%` : hasFuelLevel ? `${primaryLevelValue}%` : hasFuelWithCap ? `${primaryLevelValue}%` : hasFuelRemaining ? formatState(read("remaining_fuel"), hass) : "—";
    const primaryLevelHasBar = hasSoc || hasFuelLevel || hasFuelWithCap;
    const primaryRangeState = hasRange ? read("range_total") : hasFuelRemaining ? read("remaining_fuel") : null;
    const primaryRangeEntity = hasRange ? rangeEntity : hasFuelRemaining ? remainingFuelEntity : "";
    const primaryRangeIcon = hasRange ? "mdi:arrow-left-right" : "mdi:gas-station";
    const primaryRangeText = primaryRangeState ? formatState(primaryRangeState, hass) : "—";

    const isLocked = lockState.includes("lock") && !lockState.includes("unlock");
    const doorsOverallKnown = doorsOverallRaw !== "";
    const doorsOverallOpen = doorsOverallRaw.includes("open");
    const doorsOverallSecured =
      doorsOverallRaw.includes("closed") ||
      doorsOverallRaw.includes("locked") ||
      doorsOverallRaw.includes("secured");
    const chargingActive =
      hasCharging &&
      chargingState !== "nocharging" &&
      chargingState !== "chargingended" &&
      chargingState !== "chargingerror" &&
      (
        chargingState.includes("charging") ||
        chargingState.includes("vehicle2grid") ||
        chargingState === "v2g"
      );

    const indicatorItems = [
      {
        icon: doorsOverallKnown ? "mdi:car-door" : "mdi:car-door-lock",
        stateClass: doorsOverallKnown
          ? (doorsOverallOpen ? "alert" : doorsOverallSecured ? "ok" : "")
          : isLocked
            ? "ok"
            : "alert",
        entity: doorsOverallEntity || lockEntity,
        title: doorsOverallKnown
          ? `${t("doors_overall")}: ${doorsOverallState}`
          : `${t("lock")}: ${isLocked ? t("locked") : t("unlocked")}`,
      },
      hasAlarm
        ? {
            icon: "mdi:shield-lock",
            stateClass: alarmIsActive ? "alert" : alarmIsArmed ? "good" : "ok",
            entity: alarmArmingEntity || alarmActiveEntity,
            title: alarmIsActive
              ? t("alarm_triggered")
              : `${t("alarm")}: ${alarmArmingLabel || t("unknown")}`,
          }
        : null,
      {
        icon: openWindows > 0 ? "mdi:car-windshield-outline" : "mdi:car-windshield",
        stateClass: openWindows > 0 && !isMoving && isLocked ? "alert" : openWindows > 0 ? "" : "ok",
        entity: windowEntity,
        title: `${t("windows")}: ${openWindows > 0 ? `${openWindows} ${t("windows_open_count")}` : t("windows_closed")}`,
      },
      hasCharging
        ? {
            icon: "mdi:ev-station",
            stateClass: chargingActive ? "ok charging" : "",
            entity: chargingEntity,
            title: `${t("charging")}: ${
              chargingActive
                ? t("charging_active")
                : compactStateLabel(read("charging_state"), t)
            }`,
          }
        : lightsEntity
          ? {
              icon: "mdi:car-light-high",
              stateClass: lightsOn ? "ok" : "",
              entity: lightsEntity,
              title: `${t("lights")}: ${lightsOn ? t("lights_on") : t("lights_off")}`,
            }
          : null,	
      {
        icon: hoodOpen && tailgateOpen ? "mdi:car" : hoodOpen ? "mdi:engine-outline" : tailgateOpen ? "mdi:car-back" : "mdi:car",
        stateClass: (hoodOpen || tailgateOpen) && !isMoving && isLocked ? "alert" : (hoodOpen || tailgateOpen) ? "" : "ok",
        entity: hoodOpen ? (hoodEntity || tailgateEntity) : tailgateOpen ? (tailgateEntity || hoodEntity) : (hoodEntity || tailgateEntity),
        title: hoodOpen && tailgateOpen ? t("hood_tailgate_open") : hoodOpen ? t("hood_open") : tailgateOpen ? t("tailgate_open") : t("hood_tailgate_closed"),
      },
    ].filter(Boolean);

    if (showIndicators) {
      this._setHtml(indicatorsEl, `
        <div class="indicators">
          ${indicatorItems
            .map((item) => iconBadge(item.icon, item.stateClass, item.entity, item.title))
            .join("")}
        </div>
      `);
    } else {
      this._setHtml(indicatorsEl, "");
    }

    if (showRange && (primaryLevelState || primaryRangeState)) {
      // PHEV support: Show split range bar if both electric and fuel ranges are available
      if (isPHEV) {
        const socValue = clamp(Math.round(toNumberOrZero(socState)), 0, 100);
        const fuelLevelValue = clamp(Math.round(toNumberOrZero(read("fuel_level"))), 0, 100);
        const evRangeCurrent = toNumberOrZero(read("range_electric"));
        const fuelRangeCurrent = toNumberOrZero(read("range_fuel"));
        
        // Calculate max possible ranges (when at 100%)
        const evRangeMax = socValue > 0 ? (evRangeCurrent / socValue) * 100 : 0;
        const fuelRangeMax = fuelLevelValue > 0 ? (fuelRangeCurrent / fuelLevelValue) * 100 : 0;
        const totalMaxRange = evRangeMax + fuelRangeMax;
        
        // Calculate percentage of total max range that each current range represents
        const evRangePercent = totalMaxRange > 0 ? (evRangeCurrent / totalMaxRange) * 100 : 0;
        const fuelRangePercent = totalMaxRange > 0 ? (fuelRangeCurrent / totalMaxRange) * 100 : 0;
        
        const evRangeText = formatState(read("range_electric"), hass);
        const fuelRangeText = formatState(read("range_fuel"), hass);
        const totalRangeValue = Math.round(evRangeCurrent + fuelRangeCurrent);
        const totalRangeUnit = read("range_electric")?.attributes?.unit_of_measurement || "km";
        const totalRangeText = totalRangeValue > 0 ? `${totalRangeValue} ${totalRangeUnit}` : "—";
        
        this._setHtml(rangeEl, `
          <div class="range-box phev">
            <div class="range-top">
              <div class="bar-wrap-unified ${chargingActive ? "charging" : ""}" data-entity-id="${escapeHtml(rangeEntity)}" title="${escapeHtml(t("total_range"))}: ${escapeHtml(totalRangeText)}">
                <div class="bar-segment-unified ev" style="width:${evRangePercent}%;" data-entity-id="${escapeHtml(rangeElectricEntity)}" title="${escapeHtml(t("ev"))}: ${escapeHtml(evRangeText)} (${socValue}%)"></div>
                <div class="bar-segment-unified fuel" style="width:${fuelRangePercent}%;" data-entity-id="${escapeHtml(rangeFuelEntity)}" title="${escapeHtml(t("fuel"))}: ${escapeHtml(fuelRangeText)} (${fuelLevelValue}%)"></div>
              </div>
              <div class="range-value" data-entity-id="${escapeHtml(rangeEntity)}" title="${escapeHtml(t("total_range"))}">
                <ha-icon icon="mdi:arrow-left-right"></ha-icon>
                <span>${escapeHtml(totalRangeText)}</span>
              </div>
            </div>
            <div class="range-split-labels">
              <div class="range-split-label ev" data-entity-id="${escapeHtml(rangeElectricEntity)}" title="${escapeHtml(rangeElectricEntity)}">
                <ha-icon icon="mdi:lightning-bolt"></ha-icon>
                <span>${escapeHtml(evRangeText)} (${socValue}%)</span>
              </div>
              <div class="range-split-label fuel" data-entity-id="${escapeHtml(rangeFuelEntity)}" title="${escapeHtml(rangeFuelEntity)}">
                <ha-icon icon="mdi:gas-station"></ha-icon>
                <span>${escapeHtml(fuelRangeText)} (${fuelLevelValue}%)</span>
              </div>
            </div>
          </div>
        `);
      } else if (primaryLevelHasBar) {
        // Standard display with progress bar (SOC, fuel %, or fuel litres with manual tank capacity)
        this._setHtml(rangeEl, `
          <div class="range-box">
            <div class="range-top">
              <div class="bar-wrap ${chargingActive ? "charging" : ""}" data-entity-id="${escapeHtml(primaryLevelEntity)}" title="${escapeHtml(primaryLevelEntity)}">
                <div class="bar-level" style="width:${primaryLevelValue}%;"></div>
                <div class="energy-text">${primaryLevelLabel}</div>
              </div>
              <div class="range-value" data-entity-id="${escapeHtml(primaryRangeEntity)}" title="${escapeHtml(primaryRangeEntity)}">
                <ha-icon icon="${primaryRangeIcon}"></ha-icon>
                <span>${escapeHtml(primaryRangeText)}</span>
              </div>
            </div>
          </div>
        `);
      } else {
        // No percentage available — show range value only (no progress bar)
        this._setHtml(rangeEl, `
          <div class="range-box">
            <div class="range-top">
              <div class="range-value" data-entity-id="${escapeHtml(primaryRangeEntity)}" title="${escapeHtml(primaryRangeEntity)}">
                <ha-icon icon="${primaryRangeIcon}"></ha-icon>
                <span>${escapeHtml(primaryRangeText)}</span>
              </div>
            </div>
          </div>
        `);
      }
    } else {
      this._setHtml(rangeEl, "");
    }

    if (showImage && entities.image && hass?.states) {
      const imgState = hass.states[entities.image];
      const pic = imgState?.attributes?.entity_picture;
      this._setHtml(imageEl, pic
        ? `<div class="image ${chargingActive ? "charging" : ""}" data-entity-id="${escapeHtml(entities.image)}" title="${escapeHtml(entities.image)}"><img alt="${escapeHtml(vin)}" src="${escapeHtml(pic)}"></div>`
        : "");
    } else {
      this._setHtml(imageEl, "");
    }

    if (showMap) {
      this._renderMap(mapEl, hass, mapEntityId, t);
    } else {
      this._setHtml(mapEl, "");
    }

    if (showButtons) {
      const tireKeys = ["tire_fl", "tire_fr", "tire_rl", "tire_rr"];
      const tireLabels = { tire_fl: "FL", tire_fr: "FR", tire_rl: "RL", tire_rr: "RR" };
      const pressureToKpa = (v, u) => {
        const ul = (u || "").toLowerCase().trim();
        if (ul === "bar") return v * 100;
        if (ul === "psi") return v * 6.895;
        return v;
      };
      const kpaTo = (kpa, u) => {
        const ul = (u || "").toLowerCase().trim();
        if (ul === "bar") return kpa / 100;
        if (ul === "psi") return kpa / 6.895;
        return kpa;
      };
      const tireEntries = tireKeys
        .map((key) => {
          const obj = read(key);
          const value = toNumberOrZero(obj);
          const unit = obj?.attributes?.unit_of_measurement || "";
          return { key, value, unit, kpa: pressureToKpa(value, unit) };
        })
        .filter((t) => t.value > 0);
      const displayUnit = tireEntries.length ? tireEntries[0].unit : "";
      const tireAvgKpa = tireEntries.length
        ? tireEntries.reduce((a, b) => a + b.kpa, 0) / tireEntries.length
        : 0;
      const lowTire = tireEntries.length >= 2
        ? tireEntries.find((t) => t.kpa < tireAvgKpa * 0.8)
        : null;
      const tireAlert = lowTire !== null && lowTire !== undefined;
      const formatPressure = (v) => v >= 100 ? v.toFixed(0) : v >= 10 ? v.toFixed(1) : v.toFixed(2);
      const tireAvgDisplay = kpaTo(tireAvgKpa, displayUnit);
      const tireValue = tireAlert
        ? `${formatPressure(lowTire.value)} ${lowTire.unit}`.trim()
        : tireAvgDisplay > 0
          ? `${formatPressure(tireAvgDisplay)} ${displayUnit}`.trim()
          : "—";
      const tireEntity = tireAlert
        ? (entities[lowTire.key] || "")
        : (entities.tire_fl || entities.tire_fr || entities.tire_rl || entities.tire_rr || "");

      const quickItems = [
        {
          icon: "mdi:map-marker",
          label: t("location"),
          value: humanizeLocationState(read("device_tracker")?.state, t),
          entity: entities.device_tracker || "",
        },
        {
          icon: hasRange ? "mdi:arrow-left-right" : "mdi:gas-station",
          label: hasRange ? t("range") : t("fuel"),
          value: primaryRangeText,
          entity: primaryRangeEntity,
        },
        {
          icon: "mdi:motion-sensor",
          label: t("motion"),
          value: motionKnown ? (isMoving ? t("moving") : t("parked")) : "—",
          entity: motionEntity || "",
        },
        hasCharging
          ? {
              icon: "mdi:ev-station",
              label: t("charging"),
              value: humanizeStateValue(chargingState, t),
              entity: entities.charging_state || "",
            }
          : {
              icon: "mdi:fuel",
              label: t("level"),
              value: primaryLevelLabel,
              entity: primaryLevelEntity,
            },
        tireEntries.length
          ? {
              icon: "mdi:car-tire-alert",
              label: tireAlert ? `${t("tire")} ${tireLabels[lowTire.key]}` : t("tires"),
              value: tireValue,
              entity: tireEntity,
              alert: tireAlert,
            }
          : null,
        {
          icon: "mdi:counter",
          label: t("mileage"),
          value: formatState(read("mileage"), hass),
          entity: entities.mileage || "",
        },
      ].filter((item) => item && firstDefined(item.entity, item.value) !== "");

      this._setHtml(
        buttonsEl,
        tileGrid(quickItems.map((item) => ({ ...item, cls: item.alert ? "alert" : "" })))
      );
    } else {
      this._setHtml(buttonsEl, "");
    }

    const leasingEl = this.shadowRoot.getElementById("leasing");
    const leasingEntityId = typeof cfg.leasing_entity === "string" ? cfg.leasing_entity : "";
    if (leasingEntityId) {
      const leaseState = hass?.states?.[leasingEntityId];
      const leaseAvailable = hasUsableState(leaseState);
      const attr = (key) => (leaseAvailable ? attrNumber(leaseState, key) : NaN);
      const daysRemaining = attr("days_remaining");
      const monthsRemaining = attr("months_remaining");
      const monthlyRemaining = attr("monthly_remaining");
      const monthlyAverage = attr("monthly_average");
      const deviation = attr("deviation");
      const drivenKm = attr("actual");
      const targetKm = attr("target");
      const projectedDelta = leaseAvailable ? Number(leaseState.state) : NaN;
      const projectedCost = attr("projected_cost");
      const distanceUnit = leaseState?.attributes?.unit_of_measurement || "km";
      const currency = hass?.config?.currency || "EUR";
      const projectedDirection = leaseDeltaClass(projectedDelta);
      const costDirection = leaseDeltaClass(projectedCost);
      const plainDistance = (value) =>
        Number.isFinite(value) ? `${Math.round(value).toLocaleString()} ${distanceUnit}` : "—";
      const tileDefs = {
        lease_remaining: {
          icon: "mdi:calendar-clock",
          label: t("lease_remaining"),
          value: formatLeaseRemaining(daysRemaining, monthsRemaining, t),
          cls: "",
        },
        monthly_budget: {
          icon: "mdi:speedometer",
          label: t("monthly_budget"),
          value: plainDistance(monthlyRemaining),
          cls: Number.isFinite(monthlyRemaining) && monthlyRemaining < 0 ? "alert" : "",
        },
        monthly_average: {
          icon: "mdi:chart-timeline-variant",
          label: t("monthly_average"),
          value: plainDistance(monthlyAverage),
          cls: "",
        },
        km_balance: {
          icon: "mdi:map-marker-distance",
          label: t("km_balance"),
          value: formatSignedDistance(deviation, distanceUnit),
          cls: leaseDeltaClass(deviation),
        },
        driven: {
          icon: "mdi:counter",
          label: t("driven"),
          value: plainDistance(drivenKm),
          cls: "",
        },
        target: {
          icon: "mdi:bullseye-arrow",
          label: t("target_today"),
          value: plainDistance(targetKm),
          cls: "",
        },
        total: {
          icon: "mdi:road-variant",
          label: t("total_distance"),
          value: plainDistance(attr("total_distance")),
          cls: "",
        },
        lease_start: {
          icon: "mdi:calendar-start",
          label: t("lease_start"),
          value: formatLeaseDate(leaseState?.attributes?.lease_start, lang),
          cls: "",
        },
        lease_end: {
          icon: "mdi:calendar-end",
          label: t("lease_end"),
          value: formatLeaseDate(leaseState?.attributes?.lease_end, lang),
          cls: "",
        },
        projected: {
          icon: "mdi:chart-line",
          // Directional label ("excess"/"under") carries the sign, so the value goes unsigned;
          // neutral/unknown keeps the generic label with the signed value.
          label: projectedDirection === "alert" ? t("excess_km") : projectedDirection === "good" ? t("shortfall_km") : t("projected_at_end"),
          value: projectedDirection ? formatAbsDistance(projectedDelta, distanceUnit) : formatSignedDistance(projectedDelta, distanceUnit),
          cls: projectedDirection,
        },
        cost: {
          icon: "mdi:cash",
          label: costDirection === "alert" ? t("payment_due") : costDirection === "good" ? t("refund") : t("cost_refund"),
          value: costDirection ? formatLeaseCost(Math.abs(projectedCost), currency) : formatLeaseCost(projectedCost, currency),
          cls: costDirection,
        },
      };
      const selectedTiles = Array.isArray(cfg.leasing_tiles) && cfg.leasing_tiles.length
        ? cfg.leasing_tiles.filter((key) => tileDefs[key])
        : DEFAULT_LEASE_TILES;
      const leasingItems = selectedTiles.map((key) => tileDefs[key]);
      this._setHtml(
        leasingEl,
        tileGrid(leasingItems.map((item) => ({ ...item, entity: leasingEntityId })))
      );
    } else {
      this._setHtml(leasingEl, "");
    }
  }

  _renderMessage(message) {
    if (!this.shadowRoot) return;

    const nameEl = this._nameEl;
    const vinEl = this.shadowRoot.getElementById("vin");
    const indicatorsEl = this.shadowRoot.getElementById("indicators");
    const rangeEl = this.shadowRoot.getElementById("range_info");
    const imageEl = this.shadowRoot.getElementById("images");
    const mapEl = this.shadowRoot.getElementById("mini_map");
    const buttonsEl = this.shadowRoot.getElementById("buttons");
    const leasingEl = this.shadowRoot.getElementById("leasing");

    this._setTitleVisible(true);
    nameEl.textContent = "BMW CarData";
    vinEl.textContent = message;
    this._setHtml(indicatorsEl, "");
    this._setHtml(rangeEl, "");
    this._setHtml(imageEl, "");
    this._setHtml(mapEl, "");
    this._setHtml(buttonsEl, "");
    this._setHtml(leasingEl, "");
  }
}

if (!customElements.get(CARD_TAG)) {
  customElements.define(CARD_TAG, BmwCardataVehicleCard);
}

const cards = ensureCustomCardsArray();
if (!cards.some((c) => c && c.type === CARD_TAG)) {
  cards.push({
    type: CARD_TAG,
    name: "BMW CarData Vehicle",
    description: "BMW-style vehicle card with indicators, SOC/range, map and quick info",
    preview: true,
  });
}
