"""Example of multi-language research."""

from porf import research


def main():
    # Search in multiple languages, write in Russian
    report = research(
        topic="Влияние искусственного интеллекта на рынок труда",
        profile="balanced",
        search_languages=["ru", "en"],  # search in both Russian and English
        output_language="ru",            # write the article in Russian
        on_progress=print,
    )

    print("\n" + "=" * 60)
    print(report.markdown)

    # Save to file
    with open("multilingual_report.md", "w") as f:
        f.write(report.markdown)

    print(f"\nSaved to multilingual_report.md")
    print(f"Total sources: {len(report.sources)}")


if __name__ == "__main__":
    main()
