"""Tests for the skill discovery module."""

import json
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock
import pytest
import shutil

from skill_discovery import (
    parse_skill_yaml,
    discover_skills,
    rank_skills,
    select_top_skills,
    get_embedding,
    cosine_similarity,
    SkillMetadata,
)


@pytest.fixture
def sample_skill_content():
    """Sample SKILL.md content for testing."""
    return """---
name: web-scraper
description: A skill for scraping web pages
category: web
tags: [scraping, http, parsing]
---
# Web Scraper Skill

This skill helps scrape web pages using requests and BeautifulSoup.

## Capabilities
- Fetch HTML from URLs
- Parse DOM structure
- Extract text and links
"""


@pytest.fixture
def skill_dir_with_skills(tmp_path):
    """Create a temp directory with sample skill directories."""
    # Create skill directories
    skills = {
        "web-scraper": {
            "name": "web-scraper",
            "description": "Scrapes web pages",
            "category": "web",
        },
        "db-query": {
            "name": "db-query",
            "description": "Queries databases",
            "category": "data",
        },
        "file-manager": {
            "name": "file-manager",
            "description": "Manages files",
            "category": "system",
        },
    }
    
    for skill_name, meta in skills.items():
        skill_path = tmp_path / skill_name
        skill_path.mkdir()
        
        skill_md = f"""---
name: {meta['name']}
description: {meta['description']}
category: {meta['category']}
---
# {meta['name'].replace('-', ' ').title()}

This is a test skill for {meta['description']}.
"""
        (skill_path / "SKILL.md").write_text(skill_md)
    
    return tmp_path


def test_parse_skill_yaml(sample_skill_content):
    """Test parsing YAML frontmatter from skill content."""
    metadata, content = parse_skill_yaml(sample_skill_content)
    
    assert metadata["name"] == "web-scraper"
    assert metadata["description"] == "A skill for scraping web pages"
    assert metadata["category"] == "web"
    assert "web-scraper" in content.lower() or "web scraper" in content.lower()


def test_parse_skill_yaml_no_frontmatter():
    """Test parsing skill content without YAML frontmatter."""
    content = "# Simple Skill\n\nThis is a simple skill without frontmatter."
    metadata, parsed_content = parse_skill_yaml(content)
    
    assert metadata == {}
    assert parsed_content == content


def test_discover_skills(skill_dir_with_skills):
    """Test discovering skills in a directory."""
    skills = discover_skills(skill_dir_with_skills)
    
    assert len(skills) == 3
    
    skill_names = [s.name for s in skills]
    assert "web-scraper" in skill_names
    assert "db-query" in skill_names
    assert "file-manager" in skill_names
    
    # Check that metadata is populated
    for skill in skills:
        assert skill.name
        assert skill.description
        assert skill.path.exists()


def test_discover_skills_empty_dir(tmp_path):
    """Test discovering skills in an empty directory."""
    skills = discover_skills(tmp_path)
    assert len(skills) == 0


def test_discover_skills_nonexistent_dir():
    """Test discovering skills in a nonexistent directory."""
    with pytest.raises(FileNotFoundError):
        discover_skills(Path("/nonexistent/path"))


def test_cosine_similarity_identical():
    """Test cosine similarity with identical vectors."""
    vec = [1.0, 0.0, 0.0]
    sim = cosine_similarity(vec, vec)
    assert abs(sim - 1.0) < 1e-6


def test_cosine_similarity_orthogonal():
    """Test cosine similarity with orthogonal vectors."""
    vec1 = [1.0, 0.0, 0.0]
    vec2 = [0.0, 1.0, 0.0]
    sim = cosine_similarity(vec1, vec2)
    assert abs(sim) < 1e-6


def test_cosine_similarity_opposite():
    """Test cosine similarity with opposite vectors."""
    vec1 = [1.0, 0.0, 0.0]
    vec2 = [-1.0, 0.0, 0.0]
    sim = cosine_similarity(vec1, vec2)
    assert abs(sim + 1.0) < 1e-6


@patch('skill_discovery.get_embedding')
def test_rank_skills(mock_get_embedding):
    """Test ranking skills based on relevance to a task."""
    # Mock embeddings - make web-scraper more relevant
    def mock_embed(text):
        if "scraping" in text.lower() or "web" in text.lower():
            return [1.0, 0.0, 0.0]  # Web-related
        elif "database" in text.lower() or "query" in text.lower():
            return [0.0, 1.0, 0.0]  # Database-related
        else:
            return [0.0, 0.0, 1.0]  # Other
    
    mock_get_embedding.side_effect = mock_embed
    
    skills = [
        SkillMetadata(name="web-scraper", description="Scrapes web pages", path=Path("/fake")),
        SkillMetadata(name="db-query", description="Queries databases", path=Path("/fake")),
    ]
    
    ranked = rank_skills(skills, "scrape a website for data")
    
    # Web scraper should rank higher for a scraping task
    assert ranked[0].name == "web-scraper"


def test_select_top_skills():
    """Test selecting top N skills."""
    skills = [
        SkillMetadata(name=f"skill-{i}", description=f"Skill {i}", path=Path("/fake"))
        for i in range(10)
    ]
    
    top = select_top_skills(skills, n=3)
    assert len(top) == 3


def test_select_top_skills_fewer_than_n():
    """Test selecting top N when fewer skills available."""
    skills = [
        SkillMetadata(name="skill-1", description="Skill 1", path=Path("/fake")),
        SkillMetadata(name="skill-2", description="Skill 2", path=Path("/fake")),
    ]
    
    top = select_top_skills(skills, n=5)
    assert len(top) == 2


def test_skill_metadata_from_dict():
    """Test creating SkillMetadata from dictionary."""
    data = {
        "name": "test-skill",
        "description": "A test skill",
        "path": "/path/to/skill",
        "category": "testing",
        "tags": ["test", "example"],
    }
    
    metadata = SkillMetadata.from_dict(data)
    assert metadata.name == "test-skill"
    assert metadata.description == "A test skill"
    assert metadata.path == Path("/path/to/skill")


def test_skill_metadata_to_dict():
    """Test converting SkillMetadata to dictionary."""
    metadata = SkillMetadata(
        name="test-skill",
        description="A test skill",
        path=Path("/path/to/skill"),
    )
    
    data = metadata.to_dict()
    assert data["name"] == "test-skill"
    assert data["description"] == "A test skill"
    assert data["path"] == str(Path("/path/to/skill"))
