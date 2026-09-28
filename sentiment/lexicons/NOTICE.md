# Vendored: VADER lexicon

`vader_lexicon.txt` is the `vader_lexicon` resource distributed with
[nltk_data](https://github.com/nltk/nltk_data) (`packages/sentiment/vader_lexicon.zip`),
originally from Hutto & Gilbert's VADER project
(https://github.com/cjhutto/vaderSentiment), released under the MIT License.

It is vendored here, rather than fetched with `nltk.download()` at run time,
so scoring and tests work offline against the committed copy - the same
fixtures-first rule every other data source in this repo follows. It was
pulled once, by hand, from the public `nltk_data` GitHub Pages mirror
(free, no key, no account) during Day 2's work.
