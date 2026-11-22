//! A small seeded pseudo-random generator ([`Random`]) for the randomized solvers (MAPF-LNS2,
//! PIBT*_T), so that a run is reproducible from its seed without a dependency and in
//! `no_std`.

/// `SplitMix64` (Steele, Lea and Flood 2014): a 64-bit state advanced by a constant and
/// mixed on output. Statistically sound for picking agents and moves, not for cryptography.
#[derive(Debug, Clone)]
pub(crate) struct Random {
    /// The state; any value is valid.
    state: u64,
}

impl Random {
    /// The generator started from `seed`.
    pub(crate) fn new(seed: u64) -> Self {
        Self { state: seed }
    }

    /// The next 64 random bits.
    pub(crate) fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);

        z ^ (z >> 31)
    }

    /// A uniform index below `bound`, which must be positive (Lemire's multiply-shift; the
    /// bias is below `bound / 2^64`).
    pub(crate) fn below(&mut self, bound: usize) -> usize {
        debug_assert!(bound > 0, "an empty range has no index");
        let bound = u64::try_from(bound).expect("usize fits in u64");
        let wide = u128::from(self.next_u64()) * u128::from(bound);

        usize::try_from(wide >> 64).expect("below the bound")
    }

    /// A uniform number in `[0, 1)`.
    pub(crate) fn unit(&mut self) -> f64 {
        #[expect(
            clippy::cast_precision_loss,
            reason = "53 random bits fit an f64 mantissa exactly"
        )]
        let mantissa = (self.next_u64() >> 11) as f64;

        mantissa / 9_007_199_254_740_992.0
    }

    /// An index into `weights` with probability proportional to its weight; `None` if every
    /// weight is zero. Weights must be finite and not negative.
    pub(crate) fn weighted(&mut self, weights: &[f64]) -> Option<usize> {
        let total: f64 = weights.iter().sum();
        if total <= 0.0 {
            return None;
        }

        let mut target = self.unit() * total;
        for (index, &weight) in weights.iter().enumerate() {
            if target < weight {
                return Some(index);
            }

            target -= weight;
        }

        weights.iter().rposition(|&weight| weight > 0.0)
    }

    /// Shuffles `items` in place (Fisher-Yates).
    pub(crate) fn shuffle<T>(&mut self, items: &mut [T]) {
        for last in (1..items.len()).rev() {
            items.swap(last, self.below(last + 1));
        }
    }
}

#[cfg(test)]
mod tests {
    //! Range and determinism checks of the generator.

    use super::*;

    /// The same seed gives the same sequence; indices stay below their bound.
    #[test]
    fn is_deterministic_and_in_range() {
        let (mut a, mut b) = (Random::new(7), Random::new(7));
        for bound in 1..100 {
            let index = a.below(bound);
            assert_eq!(index, b.below(bound));
            assert!(index < bound);
        }
    }

    /// Zero weights are never picked, and all-zero weights pick nothing.
    #[test]
    fn weighted_skips_zero_weights() {
        let mut random = Random::new(1);
        for _ in 0..100 {
            assert_eq!(random.weighted(&[0.0, 2.0, 0.0]), Some(1));
        }

        assert_eq!(random.weighted(&[0.0, 0.0]), None);
    }
}
