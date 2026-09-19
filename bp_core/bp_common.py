from bp_core.names import PAD_TOKEN, UNK_TOKEN, normalize_name


class MonsterVocab:
    def __init__(self, slug_to_id=None):
        if slug_to_id is None:
            slug_to_id = {
                PAD_TOKEN: 0,
                UNK_TOKEN: 1,
            }
        self.slug_to_id = {str(slug): int(index) for slug, index in slug_to_id.items()}
        self.id_to_slug = {v: k for k, v in self.slug_to_id.items()}
        self._normalized_to_id = {normalize_name(slug): index for slug, index in self.slug_to_id.items()}
        if len(self._normalized_to_id) != len(self.slug_to_id):
            raise ValueError("vocabulary contains duplicate normalized monster names")
        if set(self.id_to_slug) != set(range(len(self.slug_to_id))):
            raise ValueError("vocabulary IDs must be contiguous from zero")
        for token in (PAD_TOKEN, UNK_TOKEN):
            if normalize_name(token) not in self._normalized_to_id:
                raise ValueError(f"vocabulary must contain {token}")

    def __len__(self):
        return len(self.slug_to_id)

    def add_token(self, slug):
        slug = normalize_name(slug)
        if slug and slug not in self.slug_to_id:
            idx = len(self.slug_to_id)
            self.slug_to_id[slug] = idx
            self.id_to_slug[idx] = slug
            self._normalized_to_id[slug] = idx

    def get_id(self, slug):
        normalized_slug = normalize_name(slug)
        if not normalized_slug or normalized_slug not in self._normalized_to_id:
            raise KeyError(f"monster is outside the vocabulary: {slug!r}")
        return self._normalized_to_id[normalized_slug]

    def get_slug(self, idx):
        index = int(idx)
        if index not in self.id_to_slug:
            raise KeyError(f"vocabulary id does not exist: {index}")
        return self.id_to_slug[index]

    def to_dict(self):
        return dict(self.slug_to_id)


def side_for_order(order, first_pick_side):
    if order < 1 or order > 10:
        return None
    first = normalize_name(first_pick_side)
    if first not in {"me", "enemy"}:
        return None
    other = "enemy" if first == "me" else "me"
    pattern = [first, other, other, first, first, other, other, first, first, other]
    return pattern[order - 1]


def infer_pick_count(my_picks, enemy_picks, first_pick_side):
    return infer_pick_count_for_side(
        my_picks,
        enemy_picks,
        first_pick_side,
        "me",
    )


def infer_pick_count_for_side(my_picks, enemy_picks, first_pick_side, side):
    """Return how many consecutive picks belong to ``side`` at the current turn."""
    next_order = len(my_picks or []) + len(enemy_picks or []) + 1
    side = normalize_name(side)
    if side not in {"me", "enemy"}:
        return 0
    count = 0
    while (
        next_order + count <= 10
        and side_for_order(next_order + count, first_pick_side) == side
    ):
        count += 1
    return count


def valid_vocab_slugs(vocab):
    blocked = {normalize_name(PAD_TOKEN), normalize_name(UNK_TOKEN)}
    return [vocab.get_slug(i) for i in range(len(vocab)) if normalize_name(vocab.get_slug(i)) not in blocked]
