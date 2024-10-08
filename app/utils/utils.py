def post_process(text: str) -> str:
    # Replace \n with actual newlines and \t with actual tabs
    return text.replace('\\n', '\n').replace('\\t', '\t')