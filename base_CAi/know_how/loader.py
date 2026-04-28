"""Loader for know-how documents."""

import glob
import os
from pathlib import Path


class KnowHowLoader:
    """Load and manage know-how documents for the agent."""

    def __init__(self, know_how_dir: str | None = None):
        """Initialize the know-how loader.

        Args:
            know_how_dir: Directory containing know-how documents.
                         If None, uses the default know_how directory.

        """
        if know_how_dir is None:
            # Default to the know_how directory in the package
            current_dir = Path(__file__).parent
            know_how_dir = str(current_dir)

        self.know_how_dir = know_how_dir
        self.documents = {}
        self._load_documents()

    def _load_documents(self):
        """Load all markdown documents from the know-how directory."""
        pattern = os.path.join(self.know_how_dir, "*.md")
        md_files = glob.glob(pattern)

        for filepath in md_files:
            filename = os.path.basename(filepath)
            filename_without_ext = os.path.splitext(filename)[0]

            # Skip README, QUICK_START, and other meta documentation (all caps filenames)
            if filename.upper() in ["README.MD", "QUICK_START.MD"] or filename_without_ext.isupper():
                continue

            # Read the document
            with open(filepath) as f:
                content = f.read()

            # Extract title, description, and metadata from the document
            title, description, metadata = self._extract_metadata(content, filename)

            # Use short_description from metadata if available, otherwise fall back to extracted description
            if "short_description" in metadata and metadata["short_description"]:
                description = metadata["short_description"]

            # Store document info
            doc_id = os.path.splitext(filename)[0]
            self.documents[doc_id] = {
                "id": doc_id,
                "name": title,
                "description": description,
                "content": content,
                "content_without_metadata": self._strip_metadata(content),
                "filepath": filepath,
                "metadata": metadata,
            }

    def _extract_metadata(self, content: str, filename: str) -> tuple[str, str, dict]:
        """Extract title, description, and metadata from markdown content.

        Args:
            content: Markdown content
            filename: Filename (used as fallback for title)

        Returns:
            Tuple of (title, description, metadata_dict)

        """
        lines = content.split("\n")

        # Extract title (first h1)
        title = None
        for line in lines:
            if line.startswith("# "):
                title = line[2:].strip()
                break

        if title is None:
            # Fallback to filename
            title = filename.replace("_", " ").replace(".md", "").title()

        # Extract metadata section
        metadata = {}
        in_metadata = False
        current_field = None

        for _i, line in enumerate(lines):
            if line.startswith("## Metadata"):
                in_metadata = True
                continue
            elif in_metadata:
                if line.startswith("##") and "Metadata" not in line:
                    # End of metadata section
                    break
                elif line.startswith("**") and "**:" in line:
                    # New metadata field (e.g., **Authors**:)
                    field_match = line.split("**")[1]
                    current_field = field_match.lower().replace(" ", "_")
                    # Get the value after the colon if it exists on the same line
                    colon_idx = line.find("**:")
                    if colon_idx != -1:
                        value_part = line[colon_idx + 3 :].strip()
                        if value_part:
                            metadata[current_field] = value_part
                        else:
                            metadata[current_field] = ""
                elif current_field and line.strip() and not line.startswith("---"):
                    # Continuation of current field
                    if current_field not in metadata:
                        metadata[current_field] = ""
                    if line.startswith("- "):
                        # List item
                        if metadata[current_field]:
                            metadata[current_field] += ", " + line[2:].strip()
                        else:
                            metadata[current_field] = line[2:].strip()
                    elif not line.startswith("```"):
                        # Regular text
                        if metadata[current_field]:
                            metadata[current_field] += " " + line.strip()
                        else:
                            metadata[current_field] = line.strip()

        # Extract description (content under ## Overview or first paragraph)
        description = ""
        in_overview = False
        overview_lines = []

        for _i, line in enumerate(lines):
            if line.startswith("## Overview"):
                in_overview = True
                continue
            elif in_overview:
                if line.startswith("##"):
                    # End of overview section
                    break
                elif line.strip():
                    overview_lines.append(line.strip())

        if overview_lines:
            description = " ".join(overview_lines)
        else:
            # Fallback: get first non-empty paragraph after title
            found_title = False
            for line in lines:
                if line.startswith("# "):
                    found_title = True
                    continue
                if found_title and line.strip() and not line.startswith("#"):
                    description = line.strip()
                    break

        # Limit description length
        if len(description) > 200:
            description = description[:197] + "..."

        return title, description, metadata

    def _strip_metadata(self, content: str) -> str:
        """Strip the metadata section from document content.

        Args:
            content: Full document content with metadata

        Returns:
            Content without metadata section

        """
        lines = content.split("\n")
        result_lines = []
        in_metadata = False
        skip_until_separator = False
        found_first_h1 = False

        for line in lines:
            # Track first H1 (title)
            if line.startswith("# ") and not found_first_h1:
                result_lines.append(line)
                found_first_h1 = True
                continue

            # Detect metadata section start
            if line.startswith("## Metadata"):
                in_metadata = True
                continue

            # Skip separator lines before and after metadata
            if line.strip() == "---":
                if not in_metadata:
                    # This might be the separator before metadata
                    skip_until_separator = True
                    continue
                else:
                    # This is the separator after metadata
                    in_metadata = False
                    skip_until_separator = False
                    continue

            # Skip lines in metadata section
            if in_metadata or skip_until_separator:
                # Check if we hit another H2 (end of metadata)
                if line.startswith("##") and "Metadata" not in line:
                    in_metadata = False
                    skip_until_separator = False
                    result_lines.append(line)
                continue

            # Keep all other lines
            result_lines.append(line)

        # Join and clean up extra blank lines
        result = "\n".join(result_lines)

        # Remove excessive blank lines (more than 2 consecutive)
        while "\n\n\n\n" in result:
            result = result.replace("\n\n\n\n", "\n\n\n")

        return result.strip()

    def get_all_documents(self) -> list[dict]:
        """Get all know-how documents as a list.

        Returns:
            List of document dictionaries with keys: id, name, description, content

        """
        return list(self.documents.values())

    def get_document_by_id(self, doc_id: str) -> dict | None:
        """Get a specific know-how document by ID.

        Args:
            doc_id: Document identifier

        Returns:
            Document dictionary or None if not found

        """
        return self.documents.get(doc_id)

    def get_document_summaries(self) -> list[dict]:
        """Get summaries of all documents (without full content).

        Returns:
            List of document summaries with keys: id, name, description

        """
        return [
            {"id": doc["id"], "name": doc["name"], "description": doc["description"]} for doc in self.documents.values()
        ]

    def add_custom_document(self, doc_id: str, name: str, description: str, content: str, metadata: dict | None = None):
        """Add a custom know-how document programmatically.

        Args:
            doc_id: Unique identifier for the document
            name: Document title
            description: Brief description
            content: Full document content
            metadata: Optional metadata dictionary (authors, license, etc.)

        """
        if metadata is None:
            metadata = {}

        self.documents[doc_id] = {
            "id": doc_id,
            "name": name,
            "description": description,
            "content": content,
            "filepath": None,
            "metadata": metadata,
        }

    def get_document_metadata(self, doc_id: str) -> dict | None:
        """Get metadata for a specific document.

        Args:
            doc_id: Document identifier

        Returns:
            Metadata dictionary or None if not found

        """
        doc = self.documents.get(doc_id)
        return doc.get("metadata", {}) if doc else None

    def print_document_info(self, doc_id: str):
        """Print formatted information about a document including metadata.

        Args:
            doc_id: Document identifier

        """
        doc = self.documents.get(doc_id)
        if not doc:
            print(f"Document '{doc_id}' not found")
            return

        print("=" * 70)
        print(f"📚 {doc['name']}")
        print("=" * 70)
        print(f"\nDescription: {doc['description']}")

        metadata = doc.get("metadata", {})
        if metadata:
            print("\n" + "-" * 70)
            print("METADATA")
            print("-" * 70)

            # Display key metadata fields
            if "authors" in metadata:
                print(f"Authors: {metadata['authors']}")
            if "affiliations" in metadata:
                print(f"Affiliations: {metadata['affiliations']}")
            if "version" in metadata:
                print(f"Version: {metadata['version']}")
            if "last_updated" in metadata:
                print(f"Last Updated: {metadata['last_updated']}")
            if "license" in metadata:
                print(f"License: {metadata['license']}")
            if "commercial_use" in metadata:
                print(f"Commercial Use: {metadata['commercial_use']}")
            if "status" in metadata:
                print(f"Status: {metadata['status']}")

        print("=" * 70)

    def remove_document(self, doc_id: str):
        """Remove a know-how document.

        Args:
            doc_id: Document identifier

        """
        if doc_id in self.documents:
            del self.documents[doc_id]

    def reload(self):
        """Reload all documents from disk."""
        self.documents = {}
        self._load_documents()
