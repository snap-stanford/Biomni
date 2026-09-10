from biomni.know_how.loader import KnowHowLoader


def test_strip_metadata_preserves_non_metadata_horizontal_rules():
    content = "# Protocol\n\nIntro\n\n---\n\n## Overview\n\nKeep this section.\n"

    assert KnowHowLoader._strip_metadata(object.__new__(KnowHowLoader), content) == content.strip()


def test_strip_metadata_removes_metadata_with_its_leading_separator():
    content = (
        "# Protocol\n\n---\n\n## Metadata\n\n**Authors**: Example\n\n---\n\n## Overview\n\nKeep this section.\n"
    )

    assert KnowHowLoader._strip_metadata(object.__new__(KnowHowLoader), content) == "# Protocol\n\n## Overview\n\nKeep this section."
