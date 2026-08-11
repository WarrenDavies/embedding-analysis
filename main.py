from transformers import AutoModelForCausalLM, AutoTokenizer


model_name = "meta-llama/Llama-3.2-3B"

concepts = ["Self(I)", "Number(2)", "Question(Who/which)", "Quantity(More)", "Quality(Good)", "Negation(Not)", "Preposition(For)"]
test_tokens_by_language = {
    "english": ["I", "two", "who", "more", "good", "not", "for"],
    "spanish": ["yo", "dos","cual", "más", "bien", "no", "por"],
    "german": ["ich", "zwei","wer", "mehr", "gut", "nicht", "für"],
    "french": ["je", "deux", "qui", "plus", "bon", "pas", "pour"],
    "italian": ["io", "due"," chi", "più", "bene", "non", "per"],
    "portuguese": ["eu", "dois","quem", "mais", "bom", "não", "para"],
}

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
)
vocab = tokenizer.get_vocab()

for lang, words in test_tokens_by_language.items():
    for word in words:
        tokens_no_space = tokenizer.encode(word, add_special_tokens=False)
        tokens_with_space = tokenizer.encode(" " + word, add_special_tokens=False)
        if len(tokens_no_space) == 1:
            print(f"{word}: 1 token as-is")
        elif len(tokens_with_space) == 1:
            print(f"'{word}' needs a leading space to be 1 token")
        else:
            print(f"{word} is more than 1 token")
