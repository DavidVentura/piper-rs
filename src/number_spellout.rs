//! Spell integers as words using CLDR RBNF (rule-based number format) rulesets.
//!
//! MMS voices tokenize raw graphemes and have no digits in their vocab, so any
//! number in the input is silently dropped. Spelling numbers into the target
//! language's words before tokenization lets them be pronounced.
//!
//! The rule files under `rbnf/` are the spellout-cardinal rulesets (plus the
//! helper rulesets they reference) extracted from the Unicode CLDR
//! `common/rbnf/<lang>.xml` data.

use std::collections::HashMap;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Lang {
    En,
    He,
    Ta,
    Th,
    Bn,
    Gu,
    Kn,
    Mr,
    Ms,
    Az,
}

impl Lang {
    /// Map a BCP-47-ish code (`he`, `he_IL`, `he-IL`) to a supported language.
    pub fn from_code(code: &str) -> Option<Lang> {
        let primary = code
            .split(|c| c == '-' || c == '_')
            .next()
            .unwrap_or(code)
            .to_ascii_lowercase();
        Some(match primary.as_str() {
            "en" => Lang::En,
            "he" | "iw" => Lang::He,
            "ta" => Lang::Ta,
            "th" => Lang::Th,
            "bn" => Lang::Bn,
            "gu" => Lang::Gu,
            "kn" => Lang::Kn,
            "mr" => Lang::Mr,
            "ms" => Lang::Ms,
            "az" => Lang::Az,
            _ => return None,
        })
    }

    fn rules(self) -> &'static str {
        match self {
            Lang::En => include_str!("rbnf/en.txt"),
            Lang::He => include_str!("rbnf/he.txt"),
            Lang::Ta => include_str!("rbnf/ta.txt"),
            Lang::Th => include_str!("rbnf/th.txt"),
            Lang::Bn => include_str!("rbnf/bn.txt"),
            Lang::Gu => include_str!("rbnf/gu.txt"),
            Lang::Kn => include_str!("rbnf/kn.txt"),
            Lang::Mr => include_str!("rbnf/mr.txt"),
            Lang::Ms => include_str!("rbnf/ms.txt"),
            Lang::Az => include_str!("rbnf/az.txt"),
        }
    }

    /// The ruleset used to read a bare number aloud. Hebrew uses its
    /// gender-neutral counting form; everything else the plain cardinal.
    fn entry_ruleset(self) -> &'static str {
        match self {
            Lang::He => "spellout-numbering",
            _ => "spellout-cardinal",
        }
    }

    /// Codepoint of digit zero for the language's native decimal digits, or
    /// `None` when it's written with ASCII digits. The other nine digits follow
    /// contiguously (a Unicode invariant for every `Nd` script), so the zero is
    /// all we need. The match is exhaustive on purpose: a new `Lang` won't
    /// compile until its digit script is classified here.
    fn native_digit_zero(self) -> Option<char> {
        match self {
            Lang::En | Lang::He | Lang::Ms | Lang::Az => None,
            Lang::Mr => Some('\u{0966}'), // Devanagari
            Lang::Bn => Some('\u{09E6}'),
            Lang::Gu => Some('\u{0AE6}'),
            Lang::Ta => Some('\u{0BE6}'),
            Lang::Kn => Some('\u{0CE6}'),
            Lang::Th => Some('\u{0E50}'),
        }
    }
}

#[derive(Debug)]
enum Tok {
    Lit(String),
    /// Quotient (n / divisor), via the named ruleset or the current one.
    Quotient(Option<String>),
    /// Remainder (n % divisor), via the named ruleset or the current one.
    Remainder(Option<String>),
    /// `[nonzero|zero]`: render `nonzero` when the remainder is non-zero, else
    /// `zero` (which is empty for a plain `[...]` with no `|`).
    Optional { nonzero: Vec<Tok>, zero: Vec<Tok> },
    /// Format the same value through another ruleset.
    Sub(String),
    /// `=#,##0=` and friends: emit the value as plain digits.
    DecimalFallback,
}

#[derive(Debug)]
struct Rule {
    base: i128,
    divisor: i128,
    tokens: Vec<Tok>,
}

#[derive(Debug, Default)]
struct Ruleset {
    rules: Vec<Rule>,
    negative: Option<Vec<Tok>>,
}

pub struct Speller {
    rulesets: HashMap<String, Ruleset>,
    entry: &'static str,
}

impl Speller {
    pub fn new(lang: Lang) -> Self {
        Speller {
            rulesets: parse_rulesets(lang.rules()),
            entry: lang.entry_ruleset(),
        }
    }

    pub fn spell(&self, n: i128) -> String {
        self.format(n, self.entry)
    }

    // Total over all of i128: any input the embedded rules can't reach falls
    // back to plain digits rather than panicking, so arbitrary runtime text is
    // always safe to spell. (The rules themselves are validated by the unit
    // tests; only `Speller::new` parsing is allowed to panic.)
    fn format(&self, n: i128, ruleset_name: &str) -> String {
        let Some(rs) = self.rulesets.get(ruleset_name) else {
            return n.to_string();
        };

        if n < 0 {
            // `n.checked_neg()` is None only for i128::MIN, which we can't spell.
            let (Some(neg), Some(abs)) = (rs.negative.as_ref(), n.checked_neg()) else {
                return n.to_string();
            };
            return self.render_negative(neg, abs, ruleset_name);
        }

        match pick_rule(&rs.rules, n) {
            Some(rule) => self.render(&rule.tokens, n, rule.divisor, ruleset_name),
            None => n.to_string(),
        }
    }

    // In a `-x: minus >>;` rule the substitution stands for the whole absolute
    // value rather than a quotient/remainder, so it's rendered directly without
    // the divisor arithmetic (which would overflow near i128::MAX).
    fn render_negative(&self, tokens: &[Tok], abs: i128, cur: &str) -> String {
        let mut out = String::new();
        for tok in tokens {
            match tok {
                Tok::Lit(s) => out.push_str(s),
                Tok::Quotient(name) | Tok::Remainder(name) => {
                    out.push_str(&self.format(abs, name.as_deref().unwrap_or(cur)))
                }
                Tok::Sub(name) => out.push_str(&self.format(abs, name)),
                Tok::Optional { .. } | Tok::DecimalFallback => {}
            }
        }
        out
    }

    fn render(&self, tokens: &[Tok], n: i128, divisor: i128, cur: &str) -> String {
        let mut out = String::new();
        for tok in tokens {
            match tok {
                Tok::Lit(s) => out.push_str(s),
                Tok::Quotient(name) => {
                    out.push_str(&self.format(n / divisor, name.as_deref().unwrap_or(cur)))
                }
                Tok::Remainder(name) => {
                    out.push_str(&self.format(n % divisor, name.as_deref().unwrap_or(cur)))
                }
                Tok::Sub(name) => out.push_str(&self.format(n, name)),
                Tok::Optional { nonzero, zero } => {
                    let branch = if n % divisor != 0 { nonzero } else { zero };
                    out.push_str(&self.render(branch, n, divisor, cur));
                }
                Tok::DecimalFallback => out.push_str(&n.to_string()),
            }
        }
        out
    }
}

/// Replace every run of decimal digits in `text` (ASCII or a supported native
/// script) with its spelled-out form.
pub fn normalize_numbers(lang: Lang, text: &str) -> String {
    let speller = Speller::new(lang);
    let mut out = String::with_capacity(text.len());
    // Accumulate a digit run as both its original text (to re-emit on overflow)
    // and its parsed value (None once it exceeds i128).
    let mut run = String::new();
    let mut value: Option<i128> = Some(0);

    let flush = |run: &mut String, value: &mut Option<i128>, out: &mut String| {
        if run.is_empty() {
            return;
        }
        match *value {
            Some(n) => out.push_str(&speller.spell(n)),
            None => out.push_str(run),
        }
        run.clear();
        *value = Some(0);
    };

    for ch in text.chars() {
        if let Some(d) = digit_value(lang, ch) {
            run.push(ch);
            value = value
                .and_then(|v| v.checked_mul(10))
                .and_then(|v| v.checked_add(d as i128));
            continue;
        }
        flush(&mut run, &mut value, &mut out);
        out.push(ch);
    }
    flush(&mut run, &mut value, &mut out);
    out
}

/// Decimal-digit value for ASCII plus, if the language uses one, its native
/// digit block. Recognition is scoped to the language so e.g. a Hebrew voice
/// doesn't try to read Thai digits.
fn digit_value(lang: Lang, c: char) -> Option<u32> {
    if let Some(d) = c.to_digit(10) {
        return Some(d);
    }
    let zero = lang.native_digit_zero()? as u32;
    let cp = c as u32;
    (zero..zero + 10).contains(&cp).then(|| cp - zero)
}

fn pick_rule(rules: &[Rule], n: i128) -> Option<&Rule> {
    rules.iter().take_while(|r| r.base <= n).last()
}

fn divisor_for(base: i128) -> i128 {
    let mut d = 1i128;
    while d * 10 <= base {
        d *= 10;
    }
    d
}

fn parse_rulesets(src: &str) -> HashMap<String, Ruleset> {
    let mut rulesets: HashMap<String, Ruleset> = HashMap::new();
    let mut current: Option<String> = None;

    for raw in src.lines() {
        let line = raw.trim();
        if line.is_empty() {
            continue;
        }
        if let Some(name) = ruleset_header(line) {
            rulesets.entry(name.clone()).or_default();
            current = Some(name);
            continue;
        }

        let name = current
            .as_ref()
            .unwrap_or_else(|| panic!("rule before any ruleset header: `{line}`"));
        let (desc, body) = line
            .split_once(':')
            .unwrap_or_else(|| panic!("malformed rule `{line}`"));
        let body = body.trim().trim_end_matches(';');
        let rs = rulesets.get_mut(name).unwrap();

        if desc.trim() == "-x" {
            rs.negative = Some(tokenize(body));
            continue;
        }
        // Decimal / infinity / NaN special rules are irrelevant to integers.
        if !desc.bytes().next().is_some_and(|b| b.is_ascii_digit()) {
            continue;
        }

        let (base, divisor) = parse_descriptor(desc.trim());
        rs.rules.push(Rule {
            base,
            divisor,
            tokens: tokenize(body),
        });
    }

    for rs in rulesets.values_mut() {
        rs.rules.sort_by_key(|r| r.base);
    }
    rulesets
}

fn ruleset_header(line: &str) -> Option<String> {
    let name = line.strip_prefix('%')?;
    let name = name.strip_suffix(':')?;
    let bare = name.trim_start_matches('%');
    if bare.is_empty() || !bare.bytes().all(|b| b.is_ascii_alphanumeric() || b == b'-') {
        return None;
    }
    Some(bare.to_owned())
}

fn parse_descriptor(desc: &str) -> (i128, i128) {
    if let Some((base, div)) = desc.split_once('/') {
        let base = base.parse().unwrap_or_else(|_| panic!("bad base `{desc}`"));
        let divisor = div.parse().unwrap_or_else(|_| panic!("bad divisor `{desc}`"));
        return (base, divisor);
    }
    let base = desc.parse().unwrap_or_else(|_| panic!("bad base `{desc}`"));
    (base, divisor_for(base))
}

fn normalize_ruleset_name(s: &str) -> String {
    s.trim_start_matches('%').to_owned()
}

fn tokenize(body: &str) -> Vec<Tok> {
    let chars: Vec<char> = body.chars().collect();
    let mut idx = 0;
    let (tokens, _) = tokenize_until(&chars, &mut idx, &[]);
    tokens
}

/// Tokenize until one of `stops` (or end of input) is reached. Returns the
/// tokens and the stop character that ended the run (consumed), if any.
fn tokenize_until(chars: &[char], idx: &mut usize, stops: &[char]) -> (Vec<Tok>, Option<char>) {
    let mut tokens = Vec::new();
    let mut lit = String::new();
    let flush = |lit: &mut String, tokens: &mut Vec<Tok>| {
        if !lit.is_empty() {
            tokens.push(Tok::Lit(std::mem::take(lit)));
        }
    };

    while *idx < chars.len() {
        let c = chars[*idx];
        if stops.contains(&c) {
            *idx += 1;
            flush(&mut lit, &mut tokens);
            return (tokens, Some(c));
        }
        match c {
            '[' => {
                flush(&mut lit, &mut tokens);
                *idx += 1;
                let (nonzero, stop) = tokenize_until(chars, idx, &['|', ']']);
                let zero = if stop == Some('|') {
                    tokenize_until(chars, idx, &[']']).0
                } else {
                    Vec::new()
                };
                tokens.push(Tok::Optional { nonzero, zero });
            }
            '\'' => *idx += 1, // apostrophe only protects adjacent spaces; drop it
            '<' if peek(chars, *idx + 1) == Some('<') => {
                flush(&mut lit, &mut tokens);
                tokens.push(Tok::Quotient(None));
                *idx += 2;
            }
            '>' if peek(chars, *idx + 1) == Some('>') => {
                flush(&mut lit, &mut tokens);
                tokens.push(Tok::Remainder(None));
                *idx += 2;
            }
            '<' if peek(chars, *idx + 1) == Some('%') => {
                flush(&mut lit, &mut tokens);
                let name = read_name(chars, idx, '<');
                tokens.push(Tok::Quotient(Some(normalize_ruleset_name(&name))));
            }
            '>' if peek(chars, *idx + 1) == Some('%') => {
                flush(&mut lit, &mut tokens);
                let name = read_name(chars, idx, '>');
                tokens.push(Tok::Remainder(Some(normalize_ruleset_name(&name))));
            }
            '=' => {
                flush(&mut lit, &mut tokens);
                let inner = read_name(chars, idx, '=');
                if inner.starts_with('%') {
                    tokens.push(Tok::Sub(normalize_ruleset_name(&inner)));
                } else {
                    tokens.push(Tok::DecimalFallback);
                }
            }
            _ => {
                lit.push(c);
                *idx += 1;
            }
        }
    }
    flush(&mut lit, &mut tokens);
    (tokens, None)
}

fn peek(chars: &[char], i: usize) -> Option<char> {
    chars.get(i).copied()
}

/// Read a `<%name<`, `>%name>`, or `=...=` payload. `*idx` points at the
/// opening delimiter; returns the raw text between the delimiters with any `%`
/// markers preserved so the caller can tell a ruleset ref from a number format.
fn read_name(chars: &[char], idx: &mut usize, close: char) -> String {
    *idx += 1; // skip opening delimiter
    let mut name = String::new();
    while *idx < chars.len() && chars[*idx] != close {
        name.push(chars[*idx]);
        *idx += 1;
    }
    *idx += 1; // skip closing delimiter
    name
}

#[cfg(test)]
mod tests {
    use super::*;

    const ALL: [Lang; 10] = [
        Lang::En,
        Lang::He,
        Lang::Ta,
        Lang::Th,
        Lang::Bn,
        Lang::Gu,
        Lang::Kn,
        Lang::Mr,
        Lang::Ms,
        Lang::Az,
    ];

    fn lang_from_tag(tag: &str) -> Lang {
        Lang::from_code(tag).unwrap_or_else(|| panic!("unknown vector lang `{tag}`"))
    }

    /// Every vector is `lang<TAB>n<TAB>expected`, generated from ICU's RBNF
    /// against the same CLDR rulesets we embed (see `rbnf/test_vectors.tsv`).
    #[test]
    fn matches_icu_ground_truth() {
        let vectors = include_str!("rbnf/test_vectors.tsv");
        let mut spellers: HashMap<&str, Speller> = HashMap::new();
        for line in vectors.lines().filter(|l| !l.trim().is_empty()) {
            let mut cols = line.split('\t');
            let tag = cols.next().unwrap();
            let n: i128 = cols.next().unwrap().parse().unwrap();
            let expected = cols.next().unwrap();
            let speller = spellers
                .entry(tag)
                .or_insert_with(|| Speller::new(lang_from_tag(tag)));
            assert_eq!(speller.spell(n), expected, "{tag} {n}");
        }
    }

    #[test]
    fn english_and_hebrew_spot_checks() {
        let en = Speller::new(Lang::En);
        assert_eq!(en.spell(305), "three hundred five");
        assert_eq!(en.spell(1234), "one thousand two hundred thirty-four");
        let he = Speller::new(Lang::He);
        assert_eq!(he.spell(123), "מאה עשרים ושלוש");
        assert_eq!(he.spell(200), "מאתיים");
    }

    // ICU 74.2 predates the `[a|b]` compound construct in CLDR, so Tamil and
    // Kannada can't be auto-checked against it; these are verified by hand.
    #[test]
    fn tamil_and_kannada_hand_checks() {
        let ta = Speller::new(Lang::Ta);
        assert_eq!(ta.spell(0), "பூஜ்யம்");
        assert_eq!(ta.spell(10), "பத்து");
        assert_eq!(ta.spell(20), "இருபது");
        assert_eq!(ta.spell(100), "நூறு");
        assert_eq!(ta.spell(1000), "ஆயிரம்");
        let kn = Speller::new(Lang::Kn);
        assert_eq!(kn.spell(0), "ಸೊನ್ನೆ");
        assert_eq!(kn.spell(10), "ಹತ್ತು");
        assert_eq!(kn.spell(100), "ನೂರು");
        assert_eq!(kn.spell(1000), "ಸಾವಿರ");
    }

    #[test]
    fn never_panics_and_never_leaks() {
        for lang in ALL {
            let s = Speller::new(lang);
            for n in [
                i128::MIN,
                i128::MIN + 1,
                i128::MAX,
                -1,
                0,
                999_999_999_999_999_999,
                1_000_000_000_000_000_000,
            ] {
                let _ = s.spell(n);
            }
            for n in -2000..2000 {
                let words = s.spell(n);
                assert!(!words.contains('|'), "{lang:?} {n} leaked `|`: {words}");
            }
            for k in 0..64 {
                let _ = s.spell(1i128 << k);
            }
        }
    }

    #[test]
    fn text_pass_spells_and_keeps_surrounding_text() {
        assert_eq!(
            normalize_numbers(Lang::He, "שלום איך אתה? 123 הוא מספר"),
            "שלום איך אתה? מאה עשרים ושלוש הוא מספר"
        );
        assert_eq!(
            normalize_numbers(Lang::En, "I have 23 cats"),
            "I have twenty-three cats"
        );
    }

    #[test]
    fn text_pass_handles_native_digits() {
        // Tamil digits ௧௨௩ = 123
        let out = normalize_numbers(Lang::Ta, "\u{0BE7}\u{0BE8}\u{0BE9}");
        assert!(!out.contains('\u{0BE7}'), "native digit survived: {out}");
        assert_eq!(out, Speller::new(Lang::Ta).spell(123));
    }

    #[test]
    fn unknown_language_code_is_none() {
        assert!(Lang::from_code("xx").is_none());
        assert_eq!(Lang::from_code("he_IL"), Some(Lang::He));
    }
}
