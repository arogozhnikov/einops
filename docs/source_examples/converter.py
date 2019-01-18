"""
just run this script with python converter.py .
It will convert pytorch.ipynb to html page docs/pytorch-examples.html

"""
import nbformat
import markdown

from pygments import highlight
from pygments.lexers import PythonLexer
from pygments.formatters import HtmlFormatter

notebook = nbformat.read('Pytorch.ipynb', as_version=nbformat.NO_CONVERT)

content = ''
cache = ''

for cell in notebook['cells']:
    if cell['cell_type'] == 'code':
        source = cell['source']
        if source.startswith('#left') or source.startswith('#right'):
            trimmed_source = source[source.index('\n') + 1:]
            cache += "<div>{}</div>".format(highlight(trimmed_source, PythonLexer(), HtmlFormatter()))
        if source.startswith('#right'):
            content += "<div class='leftright-wrapper'><div class='leftright-cells'>{}</div></div> ".format(cache)
            cache = ''

    elif cell['cell_type'] == 'markdown':
        content += "<div class='markdown-cell'>{}</div>".format(markdown.markdown(cell['source']))
    else:
        raise RuntimeError('not expected type of cell' + cell['cell_type'])

styles = HtmlFormatter().get_style_defs('.highlight')

styles += '''
    body {
        padding: 50px 10px;
    }
    .leftright-wrapper {
        text-align: center;
        overflow-x: auto;
    }
    .leftright-cells {
        display: inline-flex;
        text-align: left;
    }
    .leftright-cells > div {
        padding: 0px 10px;
        min-width: 350px;
    }
    .markdown-cell{
        max-width: 700px;
        margin: 0px auto;
    }
    h1 {
        text-align: center;
        padding: 10px 0px 0px;
    }
'''

meta_tags = '''
<meta property="og:title" content="Writing better code with pytorch and einops">
<meta property="og:description" content="Learning by example: rewriting and fixing popular code fragments">
<meta property="og:image" content="http://arogozhnikov.github.io/images/einops/einops_video.gif">
<meta property="og:video" content="http://arogozhnikov.github.io/images/einops/einops_video.mp4" />
<meta property="og:url" content="https://arogozhnikov.github.io/einops/pytorch-examples.html">
<meta name="twitter:card" content="summary_large_image">

<!--  Non-Essential, But Recommended -->

<meta property="og:site_name" content="Writing better code with pytorch and einops">
<meta name="twitter:image:alt" content="Learning by example: rewriting and fixing popular code fragments">
'''

result = f'''
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    {meta_tags}
    <title>Writing better code with pytorch+einops</title>
    <style>{styles}</style>
  </head>
  <body>
    {content}
  </body>
</html>
'''

with open('../pytorch-examples.html', 'w') as f:
    f.write(result)
