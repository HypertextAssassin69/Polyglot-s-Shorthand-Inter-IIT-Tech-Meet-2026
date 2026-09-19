from tokenizers import Tokenizer
from tokenizers.models import BPE


def create_bpe_tokenizer():
    tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
    return tokenizer