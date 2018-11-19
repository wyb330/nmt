def format_bpe_text(symbols, end_syms, delimiter=b"@@"):
    """Convert a sequence of bpe words into sentence."""
    words = []
    word = b""
    if isinstance(symbols, str):
        symbols = symbols.encode()
    delimiter_len = len(delimiter)
    for symbol in symbols:
        if symbol in end_syms:
            break
        if len(symbol) >= delimiter_len and symbol[-delimiter_len:] == delimiter:
            word += symbol[:-delimiter_len]
        else:  # end of a word
            word += symbol
            words.append(word.decode('utf8'))
            word = b""
    return " ".join(words)


punctuation = """"$%&()*+,-/:;<=>[\]^_`{|}~"""
puncuation_table = str.maketrans({key: None for key in punctuation})


def remove_puncuations(text):
    return text.translate(puncuation_table)


def tokenize_sent(sent):
    # words = nltk.word_tokenize(sent.lower())
    if sent[-1] in ['.', '?', '!']:
        sent = sent[:-1] + ' ' + sent[-1]
    sent = remove_puncuations(sent)
    words = sent.lower().split()
    return words


def read_text(filename, max_lines=0, encoding="utf8"):
    with open(filename, 'r', encoding=encoding) as f:
        lines = []
        count = 0
        for line in f:
            count += 1
            if not line:
                break
            lines.append(line.strip())
            if (max_lines > 0) and (count > max_lines):
                break
        return lines
