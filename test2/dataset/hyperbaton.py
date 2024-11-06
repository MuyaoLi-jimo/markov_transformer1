# 1 | opinion  | unusual, lovely, beautiful

# 2 | size  | big, small, tall, midsize

# 3 | physical quality  | thin, rough, untidy

# 4 | shape  | round, square, rectangular

# 5 | age  | young, old, youthful

# 6 | colour  | blue, red, pink, grey, tan

# 7 | origin  | Dutch, Japanese, Turkish, Brazilian

# 8 | material  | metal, wood, plastic

# 9 | type  | general-purpose, four-sided, U-shaped

# 10 | purpose  | walking

# 11 | noun | cat sweater
class Hyperbaton:
    def __init__(self) -> None:

        self.opinion = ["unusual", "mysterious", "silly", "repulsive", "good", "terrible", "lovely", "wonderful", "ridiculous", "obnoxious", "adorable", "nice"]
        self.size = ["big", "midsize", "enormous", "small", "tiny", "normal-size", "medium-size"]
        # self.physical = ["thin"]
        self.shape = ["round", "rectangular", "pyramidal", "spherical", "square", "triangular"]
        self.age = ["old", "new", "brand-new"]
        self.color = ["red", "grey", "gray", "tan", "purple", "green", "brown", "blue", "yellow", "black", "white", "pink", "orange"]
        self.origin = ["Chinese", "Brazilian", "Indian", "American", "Egyptian", "Pakistani", "Filipino", "German", "Korean", "Japanese", "Russian", "Vietnamese", "Nigerian", "Turkish"]
        self.material = ["metal", "rubber", "cloth", "glass", "wool", "leather", "steel"]
        # self.type = ["general-purposed", "exercise"]
        self.purpose = ["walking", "snorkeling", "exercise", "driving", "whittling", "drinking", "hiking"]
        self.noun = ["cat", "chair", "dog", "surfboard", "knife", "motorcycle", "car", "sock", "computer", "money", "shoe", "ship"]

        self.classes = {
            ('opinion', 0) : self.opinion,
            ('size', 1): self.size,
            # ('physical quality', 2): self.physical,
            ('shape', 3): self.shape,
            ('age', 4): self.age,
            ('color', 5): self.color,
            ('origin', 6): self.origin,
            ('material', 7): self.material,
            # ('type', 8): self.type,
            ('purpose', 9): self.purpose,
            ('noun', -1): self.noun,
        }
