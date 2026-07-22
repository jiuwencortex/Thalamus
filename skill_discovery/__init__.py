"""Dynamic skill discovery and retrieval.

This module provides functionality to discover skills in directories,
parse their metadata, and rank them based on relevance to a given task.
"""

from pathlib import Path
from typing import Any, Optional
import yaml


class SkillMetadata:
    """Metadata for a discovered skill."""
    
    def __init__(
        self,
        name: str,
        description: str,
        path: Path,
        category: str = "",
        tags: list[str] | None = None,
        embedding: list[float] | None = None,
    ):
        self.name = name
        self.description = description
        self.path = path
        self.category = category
        self.tags = tags or []
        self.embedding = embedding
    
    def to_dict(self) -> dict[str, Any]:
        """Convert to dictionary."""
        return {
            "name": self.name,
            "description": self.description,
            "path": str(self.path),
            "category": self.category,
            "tags": self.tags,
        }
    
    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "SkillMetadata":
        """Create from dictionary."""
        return cls(
            name=data["name"],
            description=data["description"],
            path=Path(data["path"]),
            category=data.get("category", ""),
            tags=data.get("tags", []),
        )


def parse_skill_yaml(content: str) -> tuple[dict[str, Any], str]:
    """Parse YAML frontmatter from skill content.
    
    Args:
        content: Raw skill content with optional YAML frontmatter
        
    Returns:
        Tuple of (metadata dict, content body)
    """
    if content.startswith("---"):
        parts = content.split("---", 2)
        if len(parts) >= 3:
            try:
                metadata = yaml.safe_load(parts[1]) or {}
                body = parts[2].lstrip()
                return metadata, body
            except yaml.YAMLError:
                pass
    return {}, content


def discover_skills(skills_dir: Path) -> list[SkillMetadata]:
    """Discover all skills in a directory.
    
    Args:
        skills_dir: Directory containing skill subdirectories
        
    Returns:
        List of SkillMetadata objects
        
    Raises:
        FileNotFoundError: If skills_dir doesn't exist
    """
    if not skills_dir.exists():
        raise FileNotFoundError(f"Skills directory not found: {skills_dir}")
    
    skills = []
    
    for skill_path in skills_dir.iterdir():
        if not skill_path.is_dir():
            continue
        
        skill_md = skill_path / "SKILL.md"
        if not skill_md.exists():
            continue
        
        content = skill_md.read_text(encoding="utf-8")
        metadata, body = parse_skill_yaml(content)
        
        skill = SkillMetadata(
            name=metadata.get("name", skill_path.name),
            description=metadata.get("description", ""),
            path=skill_path,
            category=metadata.get("category", ""),
            tags=metadata.get("tags", []),
        )
        skills.append(skill)
    
    return skills


def cosine_similarity(vec1: list[float], vec2: list[float]) -> float:
    """Compute cosine similarity between two vectors.
    
    Args:
        vec1: First vector
        vec2: Second vector
        
    Returns:
        Similarity score between -1 and 1
    """
    if len(vec1) != len(vec2):
        raise ValueError("Vectors must have same length")
    
    dot_product = sum(a * b for a, b in zip(vec1, vec2))
    norm1 = sum(a * a for a in vec1) ** 0.5
    norm2 = sum(b * b for b in vec2) ** 0.5
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    return dot_product / (norm1 * norm2)


def get_embedding(text: str) -> list[float]:
    """Get embedding for text (placeholder implementation).
    
    In production, this would use a real embedding model.
    For now, it returns a simple hash-based embedding.
    """
    # Simple hash-based embedding for testing
    hash_val = hash(text)
    return [float((hash_val >> i) & 1) for i in range(32)]


def rank_skills(
    skills: list[SkillMetadata],
    task_description: str,
    top_n: int = 5,
) -> list[SkillMetadata]:
    """Rank skills by relevance to a task description.
    
    Args:
        skills: List of available skills
        task_description: Description of the task to match
        top_n: Number of top skills to return
        
    Returns:
        Ranked list of skills, most relevant first
    """
    if not skills:
        return []
    
    task_embedding = get_embedding(task_description)
    
    scored_skills = []
    for skill in skills:
        skill_text = f"{skill.name} {skill.description}"
        skill_embedding = get_embedding(skill_text)
        score = cosine_similarity(task_embedding, skill_embedding)
        scored_skills.append((score, skill))
    
    # Sort by score descending
    scored_skills.sort(key=lambda x: x[0], reverse=True)
    
    return [skill for _, skill in scored_skills[:top_n]]


def select_top_skills(skills: list[SkillMetadata], n: int = 5) -> list[SkillMetadata]:
    """Select top N skills from a list.
    
    Args:
        skills: List of skills to select from
        n: Number of skills to select
        
    Returns:
        List of up to n skills
    """
    return skills[:n]
