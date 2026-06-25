
def format_source_reference(doc, index: int) -> str:
    """Format a source document for display in the UI."""
    # Find any [MM:SS] timestamps in the content to extract a cleaner reference
    return f"**Segment {index + 1}**\n\n{doc.page_content}"

def get_video_thumbnail(video_id: str) -> str:
    return f"https://img.youtube.com/vi/{video_id}/maxresdefault.jpg"


def format_duration(seconds: int) -> str:
    if seconds == 0:
        return "Unknown"
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h}h {m}m"
    if m > 0:
        return f"{m}m {s}s"
    return f"{s}s"

def format_views(views: int) -> str:
    if views == 0:
        return "Unknown"
    if views >= 1_000_000_000:
        return f"{views / 1_000_000_000:.1f}B"
    if views >= 1_000_000:
        return f"{views / 1_000_000:.1f}M"
    if views >= 1_000:
        return f"{views / 1_000:.1f}K"
    return str(views)
