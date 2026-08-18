"""ONVIF Camera Discovery — find RTSP stream URLs from a Hikvision NVR.

Run this on-site to discover available camera streams and their RTSP URLs.
The output URL can be passed directly to live_pipeline.py --url <url>.

Usage:
  python discover_cameras.py --host 192.168.1.100 --user admin --pass mypassword
  python discover_cameras.py --host 192.168.1.100 --user admin --pass mypassword --port 80
"""
import argparse
import sys

try:
    from onvif import ONVIFCamera
except ImportError:
    print("ERROR: onvif-zeep not installed. Run: pip install onvif-zeep")
    sys.exit(1)


def discover(host, port, user, password):
    print("=" * 60)
    print("  ONVIF Camera Discovery")
    print("  Host: %s:%d" % (host, port))
    print("=" * 60)

    try:
        cam = ONVIFCamera(host, port, user, password)
    except Exception as e:
        print("\nERROR: Failed to connect to ONVIF service at %s:%d" % (host, port))
        print("  %s" % e)
        print("\nTroubleshooting:")
        print("  - Verify the IP is reachable (ping %s)" % host)
        print("  - ONVIF port is usually 80 or 8000 for Hikvision NVRs")
        print("  - Check credentials with the site admin")
        return

    print("\n  Connected successfully.\n")

    # Device info
    try:
        device_service = cam.create_devicemgmt_service()
        info = device_service.GetDeviceInformation()
        print("  Device: %s %s" % (info.Manufacturer, info.Model))
        print("  Firmware: %s" % info.FirmwareVersion)
        print("  Serial: %s" % info.SerialNumber)
    except Exception as e:
        print("  [WARN] Could not fetch device info: %s" % e)

    # Media service — get profiles and stream URIs
    try:
        media_service = cam.create_media_service()
        profiles = media_service.GetProfiles()
    except Exception as e:
        print("\nERROR: Could not access media service: %s" % e)
        print("  The device may not support ONVIF media queries.")
        return

    if not profiles:
        print("\n  No media profiles found.")
        return

    print("\n  Found %d media profile(s):\n" % len(profiles))
    print("-" * 60)

    stream_urls = []
    for i, profile in enumerate(profiles):
        token = profile.token
        name = profile.Name if hasattr(profile, "Name") else "unnamed"

        # Resolution
        res_str = "unknown"
        try:
            vec = profile.VideoEncoderConfiguration
            if vec:
                w = vec.Resolution.Width
                h = vec.Resolution.Height
                enc = vec.Encoding if hasattr(vec, "Encoding") else "?"
                fps = vec.RateControl.FrameRateLimit if hasattr(vec, "RateControl") else "?"
                res_str = "%dx%d %s @ %s fps" % (w, h, enc, fps)
        except Exception:
            pass

        # Stream URI
        uri = None
        try:
            stream_setup = {
                "Stream": "RTP-Unicast",
                "Transport": {"Protocol": "RTSP"},
            }
            uri_response = media_service.GetStreamUri(
                {"StreamSetup": stream_setup, "ProfileToken": token})
            uri = uri_response.Uri
        except Exception as e:
            uri = "ERROR: %s" % e

        print("  [%d] Profile: %s (token: %s)" % (i + 1, name, token))
        print("      Resolution: %s" % res_str)
        print("      RTSP URL:   %s" % uri)
        print()

        if uri and not uri.startswith("ERROR"):
            stream_urls.append((name, uri, res_str))

    print("-" * 60)

    if stream_urls:
        print("\n  Ready-to-use commands:\n")
        for name, url, res in stream_urls:
            # Ensure credentials are in the URL
            if "@" not in url:
                url = url.replace("rtsp://", "rtsp://%s:%s@" % (user, password))
            print("  # %s (%s)" % (name, res))
            print("  python live_pipeline.py --url \"%s\" \\"
                  % url)
            print("    --config config/belt_config_top.json --speed 2.0 "
                  "--record --stream --db output/live.db")
            print()
    else:
        print("\n  No RTSP URLs discovered.")
        print("  Try direct Hikvision RTSP format:")
        print("    rtsp://%s:%s@%s:554/Streaming/Channels/101"
              % (user, password, host))
        print("    (101 = channel 1 main, 102 = channel 1 sub, "
              "201 = channel 2 main, etc.)")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Discover RTSP stream URLs from an ONVIF-compatible NVR/camera")
    parser.add_argument("--host", required=True,
                        help="NVR/camera IP address")
    parser.add_argument("--port", type=int, default=80,
                        help="ONVIF service port (default: 80, Hikvision often uses 80 or 8000)")
    parser.add_argument("--user", required=True,
                        help="ONVIF username")
    parser.add_argument("--pass", dest="password", required=True,
                        help="ONVIF password")
    args = parser.parse_args()

    discover(args.host, args.port, args.user, args.password)
