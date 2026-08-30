"""Generate packaged PNG icons from the extension's existing blue/white branding."""

from pathlib import Path

from PIL import Image, ImageDraw

OUTPUT_DIRECTORY = Path(__file__).resolve().parents[1] / "extension" / "icons"
BRAND_BLUE = "#174a72"
AIRCRAFT_POINTS = [
    (18, 72),
    (60, 62),
    (84, 27),
    (94, 29),
    (82, 63),
    (109, 71),
    (109, 80),
    (80, 78),
    (67, 102),
    (59, 101),
    (63, 76),
    (18, 81),
]


def render_icon(size: int) -> None:
    scale = 4
    canvas_size = 128 * scale
    image = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(image)
    draw.rounded_rectangle(
        (0, 0, canvas_size - 1, canvas_size - 1),
        radius=24 * scale,
        fill=BRAND_BLUE,
    )
    draw.polygon([(x * scale, y * scale) for x, y in AIRCRAFT_POINTS], fill="white")
    resized = image.resize((size, size), Image.Resampling.LANCZOS)
    resized.save(OUTPUT_DIRECTORY / f"icon{size}.png", format="PNG", optimize=True)


def main() -> None:
    OUTPUT_DIRECTORY.mkdir(parents=True, exist_ok=True)
    for size in (16, 32, 48, 128):
        render_icon(size)


if __name__ == "__main__":
    main()
