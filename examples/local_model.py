"""Example of using PORF with local models (LM Studio, Ollama)."""

from porf import research


def main():
    # Using LM Studio (OpenAI-compatible server on port 1234)
    report = research(
        topic="Introduction to Large Language Models",
        profile="quick",              # use quick profile for faster local inference
        model="openai/local-model",   # prefix with openai/ for LM Studio
        api_base="http://localhost:1234/v1",
        search_languages="auto",
        output_language="auto",
        on_progress=print,
    )

    print("\n" + "=" * 60)
    print(report.markdown)

    # Save to file
    with open("local_report.md", "w") as f:
        f.write(report.markdown)

    print(f"\nSaved to local_report.md")


if __name__ == "__main__":
    main()
