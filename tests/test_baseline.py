import qrcode

from tap_charger_scan.baseline import OpenCVQRDecoder


def test_opencv_baseline_decodes_a_clean_qr_code() -> None:
    payload = "tap://charger/NL-TAP-E123"
    image = qrcode.make(payload).convert("RGB")

    result = OpenCVQRDecoder().decode(image)

    assert result.payload == payload
    assert result.preprocessing == "original"
