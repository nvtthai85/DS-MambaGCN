"""Official NTU RGB+D 60 action names (A1-A60). Index 0 = A1.

Use these strings in every figure legend (t-SNE, confusion matrix, alpha plot)
so that labels match the dataset exactly.
"""
NTU60_CLASSES = [
    "drink water", "eat meal/snack", "brushing teeth", "brushing hair", "drop",
    "pickup", "throw", "sitting down", "standing up (from sitting position)", "clapping",
    "reading", "writing", "tear up paper", "wear jacket", "take off jacket",
    "wear a shoe", "take off a shoe", "wear on glasses", "take off glasses", "put on a hat/cap",
    "take off a hat/cap", "cheer up", "hand waving", "kicking something", "reach into pocket",
    "hopping (one foot jumping)", "jump up", "make a phone call/answer phone",
    "playing with phone/tablet", "typing on a keyboard",
    "pointing to something with finger", "taking a selfie", "check time (from watch)",
    "rub two hands together", "nod head/bow",
    "shake head", "wipe face", "salute", "put the palms together", "cross hands in front (say stop)",
    "sneeze/cough", "staggering", "falling", "touch head (headache)",
    "touch chest (stomachache/heart pain)",
    "touch back (backache)", "touch neck (neckache)", "nausea or vomiting condition",
    "use a fan (with hand or paper)/feeling warm", "punching/slapping other person",
    "kicking other person", "pushing other person", "pat on back of other person",
    "point finger at the other person", "hugging other person",
    "giving something to other person", "touch other person's pocket", "handshaking",
    "walking towards each other", "walking apart from each other",
]
assert len(NTU60_CLASSES) == 60

# A50-A60 are the two-person interaction classes (0-based 49..59).
NTU60_INTERACTION = list(range(49, 60))


def short_name(k: int) -> str:
    return f"A{k + 1:02d} {NTU60_CLASSES[k]}"
