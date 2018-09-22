import torch
import itertools
import numpy


def get_identifiers(string):
    identifiers = set('')
    parsed = []
    if '.' in string:
        assert ('...' in string) and (str.count(string, '.') == 3)
        string = string.replace('...', '.')
    left_brackets = str.count(string, '(')
    rght_brackets = str.count(string, ')')
    bracket_group = None
    def add(x):
        if x is not None:
            assert x not in identifiers
            identifiers.add(x)
            if bracket_group is None:
                parsed.append([x])
            else:
                bracket_group.append(x)
    current_identifier = None
    for char in string:
        if char in '().,':
            add(current_identifier)
            current_identifier = None
            if char == '.':
                assert 0 == 1 # TODO
                parsed.append(char)
            elif char == '(':
                assert bracket_group is None
                bracket_group = []
            elif char == ')':
                assert bracket_group is not None
                parsed.append(bracket_group)
                bracket_group = None
        elif '0' <= char <= '9':
            assert current_identifier is not None
            add(current_identifier + char)
            current_identifier = None
        elif 'a' <= char <= 'z':
            add(current_identifier)
            current_identifier = char
        else:
            raise RuntimeError()
            
        # other sybmbols are ignored
    assert bracket_group is None
    add(current_identifier)
    return identifiers, parsed


def get_lemmas(name):
    lemmas = []
    name = list(name)
    while len(name) > 0:
        letter = name.pop()
        if 'a' <= letter <= 'z' or letter == '_':
            lemmas.append(letter)
        else:
            assert '0' <= letter <= '9'
            prev_letter = name.pop()
            assert 'a' <= prev_letter <= 'z'
            lemmas.append(prev_letter + letter)
    return lemmas[::-1]


def transpose(tensor, pattern, **names):
    assert isinstance(tensor, torch.Tensor)
    left, right = pattern.split('->')
    # TODO check for commas, spaces, letter with digit, etc. Both can contain dots and brackets
    # checking that both have similar letters
    identifiers_left, seq_left = get_identifiers(left)
    identifiers_rght, seq_rght = get_identifiers(right)
    assert identifiers_left == identifiers_rght
    
    # parsing all dimenstions
    known_sizes = {}
    def add(k, v):
        assert v > 0
        v = int(v)
        if k in known_sizes:
            assert v == known_sizes[k]
        else:
            known_sizes[k] = v
    for name, value in names.items():
        # print(name, value)
        lemmas = get_lemmas(name)
        # TODO add working with letters
        if len(lemmas) == 1 and isinstance(value, int):
            assert 'a' <= name <= 'z'
            add(lemmas[0], value)
        else:
            assert len(lemmas) == len(value), [lemmas, value]
            for c, v in zip(name, value):
                if c != '_':
                    assert 'a' <= c <= 'z'
                    add(c, v)
#     print(known_sizes)
    # inferring rest of sizes
    assert len(seq_left) == len(tensor.shape)
    for group, size in zip(seq_left, tensor.shape):
        not_found = {name for name in group if name not in known_sizes}
        found_products = 1
        for name in group:
            if name in known_sizes:
                found_products *= known_sizes[name]
        if len(not_found) == 0:
            assert found_products == size
        else:
            assert len(not_found) == 1
            assert size % found_products == 0
            name, = not_found
            known_sizes[name] = size // found_products
    
    def compute_sizes_and_groups(groups_seqence, known_sizes):
        axes_sizes = []
        groups_sizes = []
        for group in groups_seqence:
            product = 1
            for name in group:
                axes_sizes.append(known_sizes[name])
                product *= known_sizes[name]
            groups_sizes.append(product)
        return axes_sizes, groups_sizes
    
    axes_sizes_left, group_sizes_left = compute_sizes_and_groups(seq_left, known_sizes=known_sizes)
    axes_sizes_rght, group_sizes_rght = compute_sizes_and_groups(seq_rght, known_sizes=known_sizes)
    
    def compute_matching(seq_left, seq_rght):
        l = list(itertools.chain(*seq_left))
        r = list(itertools.chain(*seq_rght))
        return [l.index(x) for x in r]
    
    matching = compute_matching(seq_left, seq_rght)    
    assert list(group_sizes_left) == list(tensor.shape)
    
    return tensor.reshape(axes_sizes_left).permute(matching).reshape(group_sizes_rght)


# tests below

x = torch.arange(8).reshape(2, 4)
result1 = transpose(x, 'a(bc)->bca', c=2)

x = torch.zeros(10, 20, 30, 40)

y = transpose(x, 'bhwc->bchw')
print(y.shape)


y = transpose(x, 'bhwc->bc(hw)')
print(y.shape)


y = transpose(x, 'bhw(ch1w1)->b(hh1)(ww1)c', h1=2, w1=2)
print(y.shape)


y = transpose(x, 'b(h,h1)(w,w1)c->bhw(h1w1c)', h1=2, w1=2)
print(y.shape)


y1, y2 = transpose(x, 'bhw(cg)->gbhwc', g=2)
print(y1.shape, y2.shape)


y = transpose(x, 'b1sb2t->b1b2st')
print(y.shape)


t = transpose(x, 'bchw->(bhw)c') @ torch.randn(20, 50) # @ это просто перемножение матриц
print(t.shape)

y = transpose(t, '(bhw)c2->bc2hw', b_hw=x.shape)
print(y.shape)


y = transpose(t, '(bhw)c2->bc2hw', b=30, h=10)
print(y.shape)