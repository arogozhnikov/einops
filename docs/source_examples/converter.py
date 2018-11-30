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
            content += "<div class='leftright-cells'>{}</div> ".format(cache)
            cache = ''
        
    elif cell['cell_type'] == 'markdown':
        content += "<div class='markdown-cell'>{}</div>".format(markdown.markdown(cell['source']))
    else:
        raise RuntimeError('not expected type of cell' + cell['cell_type'])

styles = HtmlFormatter().get_style_defs('.highlight')

styles += '''
    .leftright-cells {
        display: flex;
        width: 1200px;
        margin: 10px auto;
        justify-content: center;
    }
    .leftright-cells > div{
        padding: 0px 10px;
        flex-grow: 1;
        flex-basis: 0;
    }
    .markdown-cell{
        width: 700px;
        margin: 0px auto;
    }
    h1 {
        text-align: center;
        padding: 10px 0px 0px;
    }
    body {
        padding: 50px;
    }
'''

result = f'''
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8">
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