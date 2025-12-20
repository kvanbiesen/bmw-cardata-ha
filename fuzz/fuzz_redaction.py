import os
import sys

import atheris

CARDATA_PATH = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "custom_components", "cardata")
)
sys.path.insert(0, CARDATA_PATH)

with atheris.instrument_imports():
    import utils


def _consume_text(fdp: atheris.FuzzedDataProvider, max_len: int) -> str:
    return fdp.ConsumeUnicodeNoSurrogates(max_len)


def _consume_payload(fdp: atheris.FuzzedDataProvider, depth: int = 0) -> object:
    if depth >= 3:
        return _consume_text(fdp, 40)

    choice = fdp.ConsumeIntInRange(0, 6)
    if choice == 0:
        return _consume_text(fdp, 120)
    if choice == 1:
        return fdp.ConsumeIntInRange(-1_000_000, 1_000_000)
    if choice == 2:
        return fdp.ConsumeBool()
    if choice == 3:
        return [
            _consume_payload(fdp, depth + 1)
            for _ in range(fdp.ConsumeIntInRange(0, 4))
        ]
    if choice == 4:
        payload = {}
        for _ in range(fdp.ConsumeIntInRange(0, 4)):
            key = _consume_text(fdp, 20)
            payload[key] = _consume_payload(fdp, depth + 1)
        return payload
    if choice == 5:
        return tuple(
            _consume_payload(fdp, depth + 1)
            for _ in range(fdp.ConsumeIntInRange(0, 4))
        )
    return {_consume_text(fdp, 20) for _ in range(fdp.ConsumeIntInRange(0, 4))}


def TestOneInput(data: bytes) -> None:
    fdp = atheris.FuzzedDataProvider(data)
    text = _consume_text(fdp, 240)
    vin = _consume_text(fdp, 20)
    payload = _consume_payload(fdp)

    utils.redact_sensitive_data(text)
    utils.redact_vin_in_text(text)
    utils.is_valid_vin(vin)
    utils.redact_vin(vin)
    utils.redact_vins([vin, text])
    utils.redact_vin_payload(payload)


def main() -> None:
    atheris.Setup(sys.argv, TestOneInput)
    atheris.Fuzz()


if __name__ == "__main__":
    main()
