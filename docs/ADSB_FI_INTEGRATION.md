# Authorized ADSB.fi integration

The project owner confirmed on August 9, 2026 that ADSB.fi granted written permission for this
internal Monterey Jet Center operational-awareness use case. Keep the original email and complete
headers outside this repository as the authoritative permission evidence. Do not commit personal
information or email contents here.

The application uses only the documented ADSB.fi open-data API at
`https://opendata.adsb.fi/api/`, not the interactive Globe webpage. Nearby traffic is requested
through the recommended v3 latitude/longitude/distance endpoint. The configured ten-second polling
interval is below the published public limit of one request per second.

ADS-B data is provided courtesy of [ADSB.fi](https://adsb.fi/). ADSB.fi data is advisory and is
provided without warranty. The adapter retains the application's bounded cache and exponential
backoff so a temporary provider failure does not erase a previously observed aircraft immediately.

If ADSB.fi revokes permission or materially changes its terms or API, set `adsb.provider` back to
`adsb_lol` with `adsb.base_url: https://api.adsb.lol` until the integration is reviewed.
