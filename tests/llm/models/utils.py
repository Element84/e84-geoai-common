from uuid import uuid4


def build_long_system_prompt():
    """Builds a long system prompt for testing caching."""
    cache_breaking_string = f"Cache breaking string: {uuid4()}\n"
    long_system_prompt = f"{cache_breaking_string}\nPrompt caching test." + (
        "\nTest system prompt. Please ignore." * 1000
    )
    return long_system_prompt
