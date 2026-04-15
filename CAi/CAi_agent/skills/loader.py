"""Loader for skill documents."""

import glob
import os
from pathlib import Path


class SkillLoader:
    """Load and manage skill documents for the agent."""

    def __init__(self, skills_dir: str | None = None):
        """Initialize the skill loader.

        Args:
            skills_dir: Directory containing skill documents.
                       If None, uses the default skills directory.

        """
        if skills_dir is None:
            # Default to the skills directory in the package
            current_dir = Path(__file__).parent
            skills_dir = str(current_dir)

        self.skills_dir = skills_dir
        self.skills = {}
        self._load_skills()

    def _load_skills(self):
        """Load all markdown documents from the skills directory."""
        pattern = os.path.join(self.skills_dir, "*.md")
        md_files = glob.glob(pattern)

        for filepath in md_files:
            filename = os.path.basename(filepath)
            filename_without_ext = os.path.splitext(filename)[0]

            # Skip README and other meta documentation
            if filename.upper() in ["README.MD"] or filename_without_ext.isupper():
                continue

            # Read the document
            with open(filepath, encoding="utf-8") as f:
                content = f.read()

            # Extract metadata from the document
            name, description, metadata = self._extract_metadata(content, filename)

            # Store skill info
            skill_id = os.path.splitext(filename)[0]
            self.skills[skill_id] = {
                "id": skill_id,
                "name": name,
                "description": description,
                "content": content,
                "content_without_metadata": self._strip_metadata(content),
                "filepath": filepath,
                "metadata": metadata,
            }

    def _extract_metadata(self, content: str, filename: str) -> tuple[str, str, dict]:
        """Extract name, description, and metadata from markdown content.

        Expected format:
        # Skill Name

        ## Description
        Brief description of what this skill does

        ## Metadata
        **Category**: category_name
        **Required Tools**: tool1, tool2, tool3
        **Difficulty**: Easy/Medium/Hard
        **Use Cases**: use case 1, use case 2

        ## Workflow
        ...

        Args:
            content: Markdown content
            filename: Filename (used as fallback for name)

        Returns:
            Tuple of (name, description, metadata_dict)

        """
        lines = content.split("\n")

        # Extract name (first h1)
        name = None
        for line in lines:
            if line.startswith("# "):
                name = line[2:].strip()
                break

        if name is None:
            # Fallback to filename
            name = filename.replace("_", " ").replace(".md", "").title()

        # Extract metadata section
        metadata = {}
        in_metadata = False
        current_field = None

        for line in lines:
            if line.startswith("## Metadata"):
                in_metadata = True
                continue
            elif in_metadata:
                if line.startswith("##") and "Metadata" not in line:
                    # End of metadata section
                    break
                elif line.startswith("**") and "**:" in line:
                    # New metadata field (e.g., **Category**:)
                    field_match = line.split("**")[1]
                    current_field = field_match.lower().replace(" ", "_")
                    # Get the value after the colon
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

        # Extract description (content under ## Description)
        description = ""
        in_description = False
        description_lines = []

        for line in lines:
            if line.startswith("## Description"):
                in_description = True
                continue
            elif in_description:
                if line.startswith("##"):
                    # End of description section
                    break
                elif line.strip():
                    description_lines.append(line.strip())

        if description_lines:
            description = " ".join(description_lines)
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

        # Limit description length for summary
        if len(description) > 200:
            description = description[:197] + "..."

        return name, description, metadata

    def _strip_metadata(self, content: str) -> str:
        """Strip the metadata section from skill content.

        Args:
            content: Full skill content with metadata

        Returns:
            Content without metadata section

        """
        lines = content.split("\n")
        result_lines = []
        in_metadata = False
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

            # Skip lines in metadata section
            if in_metadata:
                # Check if we hit another H2 (end of metadata)
                if line.startswith("##") and "Metadata" not in line:
                    in_metadata = False
                    result_lines.append(line)
                continue

            # Keep all other lines
            result_lines.append(line)

        # Join and clean up extra blank lines
        result = "\n".join(result_lines)

        # Remove excessive blank lines
        while "\n\n\n\n" in result:
            result = result.replace("\n\n\n\n", "\n\n\n")

        return result.strip()

    def get_all_skills(self) -> list[dict]:
        """Get all skill documents as a list.

        Returns:
            List of skill dictionaries with keys: id, name, description, content

        """
        return list(self.skills.values())

    def get_skill_by_id(self, skill_id: str) -> dict | None:
        """Get a specific skill document by ID.

        Args:
            skill_id: Skill identifier

        Returns:
            Skill dictionary or None if not found

        """
        return self.skills.get(skill_id)

    def get_skill_summaries(self) -> list[dict]:
        """Get summaries of all skills (without full content).

        Returns:
            List of skill summaries with keys: id, name, description, metadata

        """
        return [
            {
                "id": skill["id"],
                "name": skill["name"],
                "description": skill["description"],
                "metadata": skill.get("metadata", {}),
            }
            for skill in self.skills.values()
        ]

    def add_custom_skill(self, skill_id: str, name: str, description: str, content: str, metadata: dict | None = None):
        """Add a custom skill document programmatically.

        Args:
            skill_id: Unique identifier for the skill
            name: Skill title
            description: Brief description
            content: Full skill content
            metadata: Optional metadata dictionary

        """
        if metadata is None:
            metadata = {}

        self.skills[skill_id] = {
            "id": skill_id,
            "name": name,
            "description": description,
            "content": content,
            "content_without_metadata": content,
            "filepath": None,
            "metadata": metadata,
        }

    def remove_skill(self, skill_id: str):
        """Remove a skill document.

        Args:
            skill_id: Skill identifier

        """
        if skill_id in self.skills:
            del self.skills[skill_id]

    def reload(self):
        """Reload all skills from disk."""
        self.skills = {}
        self._load_skills()

    def print_skill_info(self, skill_id: str):
        """Print formatted information about a skill.

        Args:
            skill_id: Skill identifier

        """
        skill = self.skills.get(skill_id)
        if not skill:
            print(f"Skill '{skill_id}' not found")
            return

        print("=" * 70)
        print(f"🎯 {skill['name']}")
        print("=" * 70)
        print(f"\nDescription: {skill['description']}")

        metadata = skill.get("metadata", {})
        if metadata:
            print("\n" + "-" * 70)
            print("METADATA")
            print("-" * 70)

            if "category" in metadata:
                print(f"Category: {metadata['category']}")
            if "required_tools" in metadata:
                print(f"Required Tools: {metadata['required_tools']}")
            if "difficulty" in metadata:
                print(f"Difficulty: {metadata['difficulty']}")
            if "use_cases" in metadata:
                print(f"Use Cases: {metadata['use_cases']}")

        print("=" * 70)
