---
name: Bug Report
about: Report a problem with the BMW CarData Home Assistant integration
title: "[Bug]: "
labels: bug
---

# 🐞 Bug Report – BMW CarData Home Assistant Integration

Thanks for taking the time to report an issue! 🙌

This template helps us understand what’s going wrong so we can fix it faster. Please fill out **as much as possible** — the more details, the better.

---

## 📋 Summary

**Briefly describe the problem:**
(One or two sentences explaining what’s not working)

---

## 🚗 Environment

Please provide the following details:

* **Home Assistant version:**
  (e.g. 2024.12.1)

* **BMW CarData HA integration version:**
  (e.g. v1.2.0)

* **Installation method:**

  * [ ] HACS
  * [ ] Manual

* **BMW vehicle model & year (optional but helpful):**
  (e.g. BMW i4 2023)

---

## ❓ What happened?

**Describe the issue in detail:**
Please explain what you expected to happen and what actually happened instead.

---

## 🔁 Steps to Reproduce

Steps to reproduce the behavior:

1.
2.
3.
4.

---

## 📸 Screenshots (if applicable)

If you have screenshots or screen recordings, please add them here.

---

## 🧪 Logs & Debug Information

⚠️ **This is very important** ⚠️

Please include relevant logs so we can diagnose the issue.

### Enable Debug Logging

If possible, enable debug logging for the integration and reproduce the issue.

```yaml
logger:
  default: info
  logs:
    custom_components.bmw_cardata: debug
```

### Paste Logs Below

Please **remove or redact personal data** (VIN, location, tokens, etc.).

```
Paste logs here
```

---

## 🧠 Additional Context

Add any other information that might help (recent Home Assistant updates, BMW ConnectedDrive changes, API issues, etc.).

---

## ✅ Checklist

Before submitting, please confirm:

* [ ] I’m using the latest version of the integration
* [ ] I’ve searched existing issues to avoid duplicates
* [ ] I’ve included logs and version information

---

❤️ Thanks for helping improve the BMW CarData Home Assistant integration!
