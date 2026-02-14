"""Example of thorough deep research."""

from porf import research


def main():
    report = research(
        topic="The Impact of Generative AI on Creative Industries",
        profile="deep",               # thorough research with all consistency checks
        search_languages=["en"],      # can add more: ["en", "ru", "de"]
        output_language="en",         # explicitly set output language
        on_progress=print,
    )

    print("\n" + "=" * 60)
    print(report.markdown)

    # Save to file
    with open("deep_report.md", "w") as f:
        f.write(report.markdown)

    print(f"\nSaved to deep_report.md")
    print(f"Total sources: {len(report.sources)}")
    print(f"Sections: {len(report.sections)}")


if __name__ == "__main__":
    main()
