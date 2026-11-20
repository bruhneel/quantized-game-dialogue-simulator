"""Compose original dialogue for the fictional town of Fallowglass."""
from pathlib import Path
import random

roles = {
    "Shopkeeper": (
        ["Saltwick lantern", "riverglass compass", "waxed map", "mothsilk cloak", "ironwood peg",
         "brass whistle", "copper kettle", "flint tin", "moonseed loaf", "reed basket"],
        ["The {a} costs two moons. Bring back the {b} and I'll call it even.",
         "Don't buy a {a} from the ferry crew. Mine comes with a {b}.",
         "My aunt hid a {a} under every floorboard. All I inherited was a {b}.",
         "This {a} has survived three floods. The {b} has survived my apprentice.",
         "If anyone asks, the {a} was always yours. The {b} is on the house.",
         "I can trade you the {a} now or wrap it with a {b} for tomorrow."]),
    "Quest Giver": (
        ["old spillway", "miller's attic", "bell orchard", "drowned gate", "south kiln",
         "mirror well", "tallow caves", "lost turnstile", "brick observatory", "eel ferry"],
        ["Meet my sister at the {a}. If she isn't there, try the {b}.",
         "The trail from the {a} to the {b} glows at dusk. Follow it without a torch.",
         "I heard three bells at the {a}, then nothing from the {b}. Find out why.",
         "Take the ledger from the {a} and read it aloud at the {b}.",
         "Before you go to the {a}, ask who still tends the {b}.",
         "Don't promise to save the {a}. Just get the children to the {b}."]),
    "Guard": (
        ["north bridge", "watch cellar", "outer wall", "candle stair", "gatehouse",
         "harbor steps", "ash market", "west parapet", "lower garden", "ferry dock"],
        ["The {a} is closed. Show your pass at the {b}.",
         "I heard you coming from the {a}. The whole {b} probably did.",
         "Nobody leaves the {a} until we hear the all-clear from the {b}.",
         "Keep your blade wrapped between the {a} and the {b}.",
         "Stand watch at the {a}. I'll send tea from the {b} if we have any.",
         "The {a} can wait. There are people trapped near the {b}."]),
    "Companion": (
        ["paper crown", "frozen turnip", "broken fiddle", "striped scarf", "leaky boot",
         "sparrow charm", "muddy satchel", "tin spoon", "rope bracelet", "pocket bell"],
        ["I brought a {a}. I have no idea where I put the {b}.",
         "When this is over, remind me to trade the {a} for a {b}.",
         "If we get separated, look for my {a}. Leave the {b} behind.",
         "My brother would have laughed at that {a}. He kept a {b} just like yours.",
         "You can carry the {a}. I'll take the {b} and make us something warm.",
         "I said I was fine. Then I tried to put my {a} inside my {b}."]),
    "Villain": (
        ["moth court", "silver lock", "winter seal", "cinder throne", "hollow bell",
         "ash decree", "empty crown", "night ferry", "white lantern", "silent choir"],
        ["When I ring the {a}, even the {b} will answer.",
         "You can keep the {a}. By dawn I will own the {b}.",
         "I offered them the {a}. They chose to hide behind the {b}.",
         "A city remembers whoever holds the {a}. Nobody remembers the {b}.",
         "Set the {a} on the table. The {b} will tell me whether you lied.",
         "I never wanted the {a}. I wanted to see you surrender the {b}."]),
}

rng = random.Random(298)
lines = []
for role, (objects, templates) in roles.items():
    for a in objects:
        others = [b for b in objects if b != a]
        for template, b in zip(templates, rng.sample(others, len(templates)), strict=True):
            lines.append(f"{role}: " + template.format(a=a, b=b))
rng.shuffle(lines)
Path(__file__).with_name("dialogue.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
