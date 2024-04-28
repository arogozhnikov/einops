"""
Converts readme from github repo page to mkdocs-friendly
"""

from pathlib import Path

original_text = Path(__file__).parent.parent.joinpath("README.md").read_text(encoding="utf-8")


def replace_with_video_tag(line: str):
    if line.startswith("https://") and line.endswith(".mp4") and " " not in line:
        # treating as link to mp4 file.
        return ""
        # return f"""
        # <video width="800" controls><source src="{line}" type="video/mp4">
        # Your browser does not support the video </video>\n\n<br />\n\n<br />
        # """.strip()
    else:
        # other lines are not touched
        return line


new_content = "\n".join([replace_with_video_tag(line) for line in original_text.splitlines()])
# save contents
docs_index = Path(__file__).parent.parent.joinpath("docs_src", "index.md")
assert docs_index.parent.exists()
docs_index.write_bytes(new_content.encode("utf-8"))
print("Converted README.md")
