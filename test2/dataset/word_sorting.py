with open("names.txt", "r") as f:
    NAMES = f.read().strip().split()

WORDS = list(map(lambda name: name.lower(), NAMES))
