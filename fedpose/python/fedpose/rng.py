"""SplitMix64 PRNG. Implemented identically in Kotlin so that shuffling matches bit for bit."""

MASK = (1 << 64) - 1


class SplitMix64:
    def __init__(self, seed: int):
        self.s = seed & MASK

    def next_u64(self) -> int:
        self.s = (self.s + 0x9E3779B97F4A7C15) & MASK
        z = self.s
        z = ((z ^ (z >> 30)) * 0xBF58476D1CE4E5B9) & MASK
        z = ((z ^ (z >> 27)) * 0x94D049BB133111EB) & MASK
        return z ^ (z >> 31)

    def next_int(self, bound: int) -> int:
        return (self.next_u64() >> 1) % bound

    def next_double(self) -> float:
        """Uniform in [0, 1) with 53 bits."""
        return (self.next_u64() >> 11) / float(1 << 53)

    def shuffle(self, a: list) -> None:
        """Fisher-Yates, from the end."""
        for i in range(len(a) - 1, 0, -1):
            j = self.next_int(i + 1)
            a[i], a[j] = a[j], a[i]
