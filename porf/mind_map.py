"""Mind map knowledge structure for Co-STORM."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from urllib.parse import urlparse, parse_qs, urlencode

from .types import Source


def normalize_url(url: str) -> str:
    """Normalize URL for deduplication: strip tracking params, dates, www, CMS prefixes."""
    p = urlparse(url.lower().rstrip('/'))
    host = (p.hostname or '').removeprefix('www.')
    path = re.sub(r'/\d{4}/\d{2}/\d{2}/', '/', p.path)
    path = re.sub(r'/\d{4}/\d{2}/', '/', path)
    # Strip common CMS path prefixes
    segments = [s for s in path.split('/') if s]
    _cms = {'blog', 'blogs', 'post', 'posts', 'articles', 'article', 'news',
            'entry', 'p', 'resources'}
    segments = [s for s in segments if s not in _cms]
    path = '/' + '/'.join(segments) if segments else '/'
    path = path.rstrip('/')
    strip = {'utm_source', 'utm_medium', 'utm_campaign', 'utm_content',
             'utm_term', 'ref', 'fbclid', 'gclid'}
    if p.query:
        params = {k: v for k, v in parse_qs(p.query).items() if k not in strip}
        query = urlencode(params, doseq=True)
    else:
        query = ''
    return f"{host}{path}{'?' + query if query else ''}"


@dataclass
class Snippet:
    """Atomic unit of information in the mind map."""
    source: Source
    question: str
    query: str
    uuid: int = 0


@dataclass(eq=False)
class MindMapNode:
    """Node in the mind map tree."""
    name: str
    brief: str = ""
    children: list[MindMapNode] = field(default_factory=list)
    parent: MindMapNode | None = field(default=None, repr=False)
    snippet_uuids: set[int] = field(default_factory=set)
    claim: str = ""
    draft: str = ""
    open_questions: list[str] = field(default_factory=list)
    contributors: set[str] = field(default_factory=set)

    def leaves(self) -> list[MindMapNode]:
        if not self.children:
            return [self]
        return [n for c in self.children for n in c.leaves()]

    def all_nodes(self) -> list[MindMapNode]:
        result = [self]
        for c in self.children:
            result.extend(c.all_nodes())
        return result

    def path(self) -> str:
        parts = []
        node = self
        while node and node.parent is not None:
            parts.append(node.name)
            node = node.parent
        return " > ".join(reversed(parts))


class MindMap:
    """Hierarchical knowledge structure with per-node snippet accumulation."""

    def __init__(self, topic: str):
        self.root = MindMapNode(name=topic)
        self.snippets: dict[int, Snippet] = {}
        self._next_uuid = 1
        self._hashes: set[int] = set()
        self.reservoir: list[Source] = []
        self._reservoir_urls: set[str] = set()

    # ── Snippet management ────────────────────────────────────────────

    def add_snippet(self, snippet: Snippet) -> int | None:
        """Add snippet to global store. Returns uuid, or None if duplicate."""
        h = hash((normalize_url(snippet.source.url),
                  snippet.source.content[:200], snippet.question[:100]))
        if h in self._hashes:
            return None
        self._hashes.add(h)
        uid = self._next_uuid
        self._next_uuid += 1
        snippet.uuid = uid
        self.snippets[uid] = snippet
        return uid

    def attach(self, node: MindMapNode, uid: int):
        """Attach a snippet to a node."""
        node.snippet_uuids.add(uid)

    def add_to_reservoir(self, sources: list[Source]):
        """Store sources for later gap-filling (uncited results go here)."""
        for src in sources:
            norm = normalize_url(src.url)
            if norm not in self._reservoir_urls:
                self._reservoir_urls.add(norm)
                self.reservoir.append(src)

    # ── Queries ───────────────────────────────────────────────────────

    def least_explored_leaf(self) -> MindMapNode:
        leaves = self.root.leaves()
        return min(leaves, key=lambda n: len(n.snippet_uuids)) if leaves else self.root

    def structure_paths(self) -> list[tuple[MindMapNode, str]]:
        """All (node, path_string) pairs for non-root nodes."""
        result: list[tuple[MindMapNode, str]] = []
        def _walk(node: MindMapNode, parts: list[str]):
            for child in node.children:
                p = parts + [child.name]
                result.append((child, " > ".join(p)))
                _walk(child, p)
        _walk(self.root, [self.root.name])
        return result

    def summary(self) -> str:
        """Text representation of the tree with snippet counts."""
        lines: list[str] = []
        def _walk(node: MindMapNode, indent: int):
            n = len(node.snippet_uuids)
            suffix = f" ({n} sources)" if n else ""
            lines.append(f"{'  ' * indent}{node.name}{suffix}")
            for c in node.children:
                _walk(c, indent + 1)
        _walk(self.root, 0)
        return "\n".join(lines)

    def claim_summary(self) -> str:
        """Text representation of the tree with claims and draft status."""
        lines: list[str] = []
        def _walk(node: MindMapNode, indent: int):
            n = len(node.snippet_uuids)
            claim = f" — {node.claim}" if node.claim else ""
            draft = " [draft]" if node.draft else ""
            lines.append(f"{'  ' * indent}{node.name} ({n} src{draft}){claim}")
            for c in node.children:
                _walk(c, indent + 1)
        _walk(self.root, 0)
        return "\n".join(lines)

    def numbered_sections(self) -> tuple[dict[int, MindMapNode], str]:
        """Numbered section list for prompts. Returns (index→node map, text)."""
        index_map: dict[int, MindMapNode] = {}
        lines: list[str] = []
        counter = 0
        def _walk(node: MindMapNode, indent: int):
            nonlocal counter
            for child in node.children:
                counter += 1
                index_map[counter] = child
                n = len(child.snippet_uuids)
                claim = f" — {child.claim}" if child.claim else ""
                draft = " [draft]" if child.draft else ""
                lines.append(f"{'  ' * indent}{counter}. {child.name} ({n} src{draft}){claim}")
                _walk(child, indent + 1)
        _walk(self.root, 0)
        return index_map, "\n".join(lines)

    def format_node_snippets(self, node: MindMapNode) -> str:
        """Format all snippets in node's subtree, numbered by UUID."""
        uuids: set[int] = set()
        def _collect(n: MindMapNode):
            uuids.update(n.snippet_uuids)
            for c in n.children:
                _collect(c)
        _collect(node)

        ordered = [self.snippets[u] for u in sorted(uuids) if u in self.snippets]
        return "\n\n".join(
            f"[{s.uuid}] {s.source.title}\n{s.source.content}" for s in ordered
        )

    # ── Mutations ─────────────────────────────────────────────────────

    def add_child(self, parent: MindMapNode, name: str,
                  brief: str = "") -> MindMapNode:
        child = MindMapNode(name=name, brief=brief, parent=parent)
        parent.children.append(child)
        return child

    def expand_node(self, node: MindMapNode, sub_names: list[str]):
        """Split a leaf node into sub-nodes."""
        for name in sub_names:
            self.add_child(node, name)

    def flatten_to(self, max_leaves: int):
        """Collapse deepest leaf groups until <= max_leaves."""
        while len(self.root.leaves()) > max_leaves:
            # Find deepest node whose children are ALL leaves
            best, best_depth = None, -1
            for node in self.root.all_nodes():
                if not node.children or node is self.root:
                    continue
                if not all(not c.children for c in node.children):
                    continue
                d = 0
                n = node
                while n.parent:
                    d += 1
                    n = n.parent
                if d > best_depth:
                    best, best_depth = node, d
            if not best:
                break
            for child in best.children:
                best.snippet_uuids |= child.snippet_uuids
            best.children.clear()

    def rebuild_from_plan(self, plan: list[dict]):
        """Restructure tree to match article plan.

        Each plan section specifies source node names from the research tree.
        Snippets are redistributed from old nodes to new plan-aligned nodes.
        """
        if not plan:
            return

        # Index all non-root nodes by lowercase name
        old_nodes: dict[str, MindMapNode] = {}
        for node in self.root.all_nodes():
            if node is not self.root:
                old_nodes[node.name.lower().strip()] = node

        # Collect all snippet uuids from tree
        all_uuids: set[int] = set()
        for node in self.root.all_nodes():
            all_uuids.update(node.snippet_uuids)

        def match(name: str) -> MindMapNode | None:
            low = name.lower().strip()
            if low in old_nodes:
                return old_nodes[low]
            for node_name, node in old_nodes.items():
                if low in node_name or node_name in low:
                    return node
            return None

        claimed: set[int] = set()
        new_children: list[MindMapNode] = []

        def collect_snippets(node, source_names):
            for src_name in source_names:
                matched = match(src_name)
                if matched:
                    for n in matched.all_nodes():
                        for uid in n.snippet_uuids:
                            if uid not in claimed:
                                node.snippet_uuids.add(uid)
                                claimed.add(uid)

        for section in plan:
            name = section.get("name", "")
            if not name:
                continue
            new_node = MindMapNode(name=name, parent=self.root)
            subs = section.get("subsections", [])
            if subs:
                for sub in subs:
                    sub_node = MindMapNode(name=sub.get("name", ""), parent=new_node)
                    collect_snippets(sub_node, sub.get("sources", []))
                    new_node.children.append(sub_node)
            else:
                collect_snippets(new_node, section.get("sources", []))
            new_children.append(new_node)

        # Distribute unclaimed snippets to leaf nodes with fewest
        all_leaves = [leaf for nc in new_children for leaf in nc.leaves()]
        for uid in all_uuids - claimed:
            if all_leaves:
                target = min(all_leaves, key=lambda n: len(n.snippet_uuids))
                target.snippet_uuids.add(uid)

        self.root.children = new_children

    def trim(self):
        """Remove empty leaves; merge single-child nodes."""
        changed = True
        while changed:
            changed = False
            for node in self.root.all_nodes():
                before = len(node.children)
                node.children = [c for c in node.children
                                 if c.children or c.snippet_uuids]
                if len(node.children) < before:
                    changed = True
                if len(node.children) == 1 and node is not self.root:
                    child = node.children[0]
                    node.snippet_uuids |= child.snippet_uuids
                    node.children = child.children
                    for gc in node.children:
                        gc.parent = node
                    changed = True

