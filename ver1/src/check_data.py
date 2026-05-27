from tokenizers import Tokenizer
from tokenizers.models import BPE
from tokenizers.pre_tokenizers import ByteLevel

tokenizer = Tokenizer(BPE(
    vocab="C:\\Job\\Depression Detect\\vie dataset\\tokenizer\\vocab.json",
    merges="C:\\Job\\Depression Detect\\vie dataset\\tokenizer\\merges.txt",
))
tokenizer.pre_tokenizer = ByteLevel()

tokenizer.save("C:\\Job\\Depression Detect\\vie dataset\\tokenizer\\tokenizer.json")
print("Done!")