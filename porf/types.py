"""Data types for deep_research."""

from dataclasses import dataclass, field


@dataclass
class Source:
    """Information source."""
    url: str
    title: str
    content: str
    query: str  # search query that led to this source


@dataclass
class Section:
    """Report section."""
    title: str
    content: str
    sources: list[int] = field(default_factory=list)  # source indices
    level: int = 2


@dataclass
class Report:
    """Research result."""
    topic: str
    sections: list[Section]
    sources: list[Source]
    trace: list[dict] = field(default_factory=list)

    @property
    def markdown(self) -> str:
        """Report as Markdown with citations."""
        lines = [f"# {self.topic}\n"]

        for section in self.sections:
            if section.title:
                lines.append(f"{'#' * section.level} {section.title}\n")
            if section.content:
                lines.append(section.content)
                lines.append("")

        # Sources
        if self.sources:
            lines.append("## Sources\n")
            for i, src in enumerate(self.sources, 1):
                lines.append(f"{i}. [{src.title}]({src.url})")

        return "\n".join(lines)

    def to_dict(self) -> dict:
        """Serialize to dict."""
        return {
            "topic": self.topic,
            "sections": [
                {"title": s.title, "content": s.content, "sources": s.sources, "level": s.level}
                for s in self.sections
            ],
            "sources": [
                {"url": s.url, "title": s.title, "content": s.content, "query": s.query}
                for s in self.sources
            ]
        }
