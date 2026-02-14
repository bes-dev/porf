"""Basic example of using PORF."""

from porf import research


def main():
    report = research(
        topic="The Evolution of Cyberpunk: From Literary Movement to Cultural Phenomenon",
        profile="balanced",           # "quick", "balanced", "deep", "academic"
        search_languages="auto",      # auto-detect from topic
        output_language="auto",       # write in same language as topic
        on_progress=print,
    )

    print("\n" + "=" * 60)
    print(report.markdown)

    # Save to file
    with open("report.md", "w") as f:
        f.write(report.markdown)

    print(f"\nSaved to report.md")
    print(f"Total sources: {len(report.sources)}")
    print(f"Sections: {len(report.sections)}")


if __name__ == "__main__":
    main()
