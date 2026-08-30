from __future__ import annotations

from dataclasses import dataclass, field

from mry_alert.models import NearbyAircraft

DIGITS = {
    "0": "Zero",
    "1": "One",
    "2": "Two",
    "3": "Three",
    "4": "Four",
    "5": "Five",
    "6": "Six",
    "7": "Seven",
    "8": "Eight",
    "9": "Niner",
}
PHONETIC = {
    "A": "Alpha",
    "B": "Bravo",
    "C": "Charlie",
    "D": "Delta",
    "E": "Echo",
    "F": "Foxtrot",
    "G": "Golf",
    "H": "Hotel",
    "I": "India",
    "J": "Juliet",
    "K": "Kilo",
    "L": "Lima",
    "M": "Mike",
    "N": "November",
    "O": "Oscar",
    "P": "Papa",
    "Q": "Quebec",
    "R": "Romeo",
    "S": "Sierra",
    "T": "Tango",
    "U": "Uniform",
    "V": "Victor",
    "W": "Whiskey",
    "X": "X-ray",
    "Y": "Yankee",
    "Z": "Zulu",
}


def registration_spoken_form(registration: str) -> str | None:
    value = registration.upper().replace("-", "").strip()
    if not value.startswith("N") or len(value) < 2 or not value[1:].isalnum():
        return None
    words = [PHONETIC["N"]]
    for character in value[1:]:
        word = DIGITS.get(character) or PHONETIC.get(character)
        if word is None:
            return None
        words.append(word)
    return " ".join(words)


def build_adsb_prompt(aircraft: list[NearbyAircraft], maximum: int = 5) -> str:
    plausible = [
        item
        for item in aircraft
        if item.registration and (item.seconds_since_seen is None or item.seconds_since_seen <= 20)
    ]
    plausible.sort(
        key=lambda item: (
            item.distance_nm if item.distance_nm is not None else 999.0,
            item.seconds_since_seen if item.seconds_since_seen is not None else 999.0,
        )
    )
    selected = plausible[:maximum]
    forms = [
        form
        for item in selected
        if (form := registration_spoken_form(item.registration or "")) is not None
    ]
    registrations = [item.registration for item in selected if item.registration]
    if not registrations or not forms:
        return ""
    return (
        "Nearby aircraft registrations:\n"
        + ", ".join(registrations)
        + ".\nPossible spoken forms:\n"
        + ".\n".join(forms)
        + "."
    )


@dataclass
class AdsbPromptCache:
    prompt: str = ""
    registrations: list[str] = field(default_factory=list)

    def update(self, aircraft: list[NearbyAircraft], maximum: int) -> None:
        self.prompt = build_adsb_prompt(aircraft, maximum)
        self.registrations = [
            item.registration for item in aircraft[:maximum] if item.registration is not None
        ]
