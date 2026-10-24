COLORS = [
    "red", "blue", "green", "yellow", "black",
    "white", "brown", "gray", "orange", "pink", "purple",
]
REFUSAL = "unknown"
CANDIDATES = COLORS + [REFUSAL]

COLOR_RGB = {
    "red": (200, 30, 30),
    "blue": (30, 60, 200),
    "green": (30, 140, 50),
    "yellow": (220, 200, 30),
    "black": (25, 25, 25),
    "white": (240, 240, 240),
    "brown": (120, 75, 40),
    "gray": (128, 128, 128),
    "orange": (240, 140, 20),
    "pink": (230, 140, 170),
    "purple": (120, 60, 160),
}

# wrong hint used in the "text conflict" condition; deliberately far in hue
# from the gold color so the image clearly contradicts the hint
DISTRACTOR = {
    "red": "green", "green": "red", "blue": "yellow", "yellow": "blue",
    "black": "white", "white": "black", "brown": "blue", "gray": "red",
    "orange": "blue", "pink": "green", "purple": "yellow",
}

# chromatic color used for the targeted counterfactual repaint
CF_COLOR = {
    "red": "green", "blue": "green", "green": "yellow", "yellow": "red",
    "black": "red", "white": "blue", "brown": "yellow", "gray": "blue",
    "orange": "blue", "pink": "purple", "purple": "pink",
}

CLASSES = [
    "car", "bus", "truck", "motorcycle", "bicycle", "boat", "train",
    "airplane", "dog", "cat", "horse", "sheep", "cow", "bird", "elephant",
    "bear", "zebra", "giraffe", "bench", "chair", "couch", "bed",
    "backpack", "handbag", "suitcase", "umbrella", "tie", "bottle", "cup",
    "bowl", "book", "clock", "vase", "teddy bear", "potted plant",
]

HINTS = ["correct", "wrong", "none"]
EVIDENCE = ["sufficient", "insufficient"]
