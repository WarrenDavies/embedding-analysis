def print_if_words_are_single_token(words, tokenizer):
    for word in words:
        tokens_no_space = tokenizer.encode(word, add_special_tokens=False)
        tokens_with_space = tokenizer.encode(" " + word, add_special_tokens=False)
        if len(tokens_no_space) == 1:
            print(f"{word}: 1 token as-is")
        elif len(tokens_with_space) == 1:
            print(f"'{word}' needs a leading space to be 1 token")
        else:
            print(f"{word} is more than 1 token")