
# from spacy.lang.en.tokenizer_exceptions import TOKENIZER_EXCEPTIONS as SENT_END_EXCEPTIONS
SENT_END_EXCEPTIONS = ["a.", "b.", "c.", "d.", "e.", "f.", "g.", "h.", "i.", "j.", "k.", "l.", "m.", "n.", "o.", "p.", "q.", "r.", "s.", "t.", "u.", "v.", "w.", "x.", "y.", "z.", "ä.", "ö.", "ü.", "._.", "°c.", "°f.", "°k.", "1a.m.", "1p.m.", "2a.m.", "2p.m.", "3a.m.", "3p.m.", "4a.m.", "4p.m.", "5a.m.", "5p.m.", "6a.m.", "6p.m.", "7a.m.", "7p.m.", "8a.m.", "8p.m.", "9a.m.", "9p.m.", "10a.m.", "10p.m.", "11a.m.", "11p.m.", "12a.m.", "12p.m.", "mt.", "ak.", "ala.", "apr.", "ariz.", "ark.", "aug.", "calif.", "colo.", "conn.", "dec.", "del.", "feb.", "fla.", "ga.", "ia.", "id.", "ill.", "ind.", "jan.", "jul.", "jun.", "kan.", "kans.", "ky.", "la.", "mar.", "mass.", "mich.", "minn.", "miss.", "n.c.", "n.d.", "n.h.", "n.j.", "n.m.", "n.y.", "neb.", "nebr.", "nev.", "nov.", "oct.", "okla.", "ore.", "pa.", "s.c.", "sep.", "sept.", "tenn.", "va.", "wash.", "wis.", "a.m.", "adm.", "bros.", "co.", "corp.", "d.c.", "dr.", "e.g.", "gen.", "gov.", "i.e.", "inc.", "jr.", "ltd.", "md.", "messrs.", "mo.", "mont.", "mr.", "mrs.", "ms.", "p.m.", "ph.d.", "prof.", "rep.", "rev.", "sen.", "st.", "vs.", "v.s."]

def is_sent_end(text: str) -> bool:
    if not text or text[-1].isalnum():
        return False
    last_word = text.split()[-1]
    return last_word.lower() not in SENT_END_EXCEPTIONS


def make_script(text: str, norm: bool = False) -> str:
    text = "".join(e if e.isalnum() else " " for e in text)
    if norm:
        text = text.lower()
    return " ".join(text.split())


def make_ref(text: str) -> str:
    return "".join(e for e in text.lower() if e.isalnum())
